"""Serialization, shared by training and serving so both see identical
token sequences.

    prefix   = <|state|> state <|/state|> <|facts|> F1 ... <|/facts|>
    question = <|q|> instructions <|opt|> key: description <|mark|> ... <|abstain|> <|mark|> <|/q|>

The prefix and each question block are tokenized separately and concatenated as token ids, so
running [prefix + block] as one sequence and running the block against a cached prefix are the
same computation (prefix-fork).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import facts as fact_channel

SPECIAL = ["<|state|>", "<|/state|>", "<|facts|>", "<|/facts|>", "<|q|>", "<|opt|>", "<|mark|>",
           "<|abstain|>", "<|/q|>"]
NOUL_DEFAULT = {"true": "Yes", "false": "No"}


def state_text(state) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                         for k, v in state.items())
    return json.dumps(state, ensure_ascii=False)


def options(q: dict) -> tuple[list[str], list[str]]:
    """Option keys (the response keys) and descriptions, in the order given."""
    t, crit = q["type"], q.get("criteria")
    if t == "score":
        return [str(i) for i in range(len(crit))], [str(c) for c in crit]
    if t == "noul":
        crit = {**NOUL_DEFAULT, **(crit or {})}
        return ["true", "false"], [str(crit["true"]), str(crit["false"])]
    return list(crit), [str(v) for v in crit.values()]


@dataclass
class Prefix:
    ids: list[int]
    state_span: tuple[int, int]            # token range of the state text
    state_offsets: list[tuple[int, int]]   # char offsets of state tokens within the state text
    text: str
    truncated: bool = False


@dataclass
class Block:
    ids: list[int]
    marks: list[int]    # positions of <|mark|> within the block: one per option, last = abstain
    end: int            # position of <|/q|> within the block
    keys: list[str]
    qtype: str
    levels: list[int] = field(default_factory=list)  # ordinal index per option (score only)


class Serializer:
    def __init__(self, tokenizer, max_prefix_tokens: int = 3584, use_facts: bool = True):
        self.tok = tokenizer
        missing = [t for t in SPECIAL if t not in tokenizer.get_vocab()]
        if missing:
            tokenizer.add_tokens(missing, special_tokens=True)
        self.special = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL}
        self.max_prefix = max_prefix_tokens
        self.use_facts = use_facts

    def _enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def prefix(self, state, questions: list[dict], reference_time=None) -> Prefix:
        s = state_text(state)
        enc = self.tok(s, add_special_tokens=False, return_offsets_mapping=True)
        s_ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
        fl = []
        if self.use_facts:
            q_texts = [q.get("instructions", "") + " " + " ".join(options(q)[1]) for q in questions]
            fl = fact_channel.facts(s, q_texts, reference_time)
        f_ids = self._enc("\n".join(fl)) if fl else []
        budget = self.max_prefix - 4 - len(f_ids)
        truncated = len(s_ids) > budget
        if truncated:  # keep head and tail of the state; the middle is least often decisive
            head = budget * 2 // 3
            tail = budget - head
            s_ids = s_ids[:head] + s_ids[-tail:]
            offs = offs[:head] + offs[-tail:]
        ids = [self.special["<|state|>"]] + s_ids + [self.special["<|/state|>"],
                                                     self.special["<|facts|>"]]
        ids += f_ids + [self.special["<|/facts|>"]]
        return Prefix(ids, (1, 1 + len(s_ids)), offs, s, truncated)

    def block(self, q: dict) -> Block:
        keys, descs = options(q)
        sp = self.special
        ids = [sp["<|q|>"]] + self._enc(" " + q.get("instructions", "").strip())
        marks = []
        for k, d in zip(keys, descs):
            ids += [sp["<|opt|>"]] + self._enc(f" {k}: {d}") + [sp["<|mark|>"]]
            marks.append(len(ids) - 1)
        ids += [sp["<|abstain|>"], sp["<|mark|>"]]
        marks.append(len(ids) - 1)
        ids.append(sp["<|/q|>"])
        levels = list(range(len(keys))) if q["type"] == "score" else []
        return Block(ids, marks, len(ids) - 1, keys, q["type"], levels)


def evidence_mask(prefix: Prefix, quotes: list[str] | None) -> list[float] | None:
    """Per state token: 1 if the token lies inside any evidence quote found in the state."""
    if not quotes:
        return None
    spans = []
    for qt in quotes:
        i = prefix.text.find(qt)
        if i >= 0:
            spans.append((i, i + len(qt)))
    if not spans:
        return None
    return [float(any(a < e and b > s for s, e in spans)) for a, b in prefix.state_offsets]
