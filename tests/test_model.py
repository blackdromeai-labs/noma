"""CPU tests for the fact channel, serializer, bootstrap, and loss (no backbone needed)."""

import torch

from noma.model import facts
from noma.model.heads import Heads, gather_slots
from noma.model.serialize import SPECIAL, Prefix, Serializer, evidence_mask, options, state_text
from noma.train.data import Item, bootstrap_weight
from noma.train import losses


def test_facts_dates_and_amounts():
    f = facts.facts("Paid $620 on 2026-03-02; limit was $500. Filed 3 days ago.",
                    ["Is the amount over $500?"], "2026-03-20T09:00:00Z")
    text = "\n".join(f)
    assert "reference time 2026-03-20" in text
    assert "18 days before reference" in text
    assert "620 USD" in text and "> 500 USD" in text
    assert "2026-03-17" in text          # "3 days ago" resolved
    assert all(line.startswith(f"F{i}:") for i, line in enumerate(f, 1))
    assert len(f) <= facts.MAX_FACTS


def test_facts_v02_times_totals_units():
    s = ("B1 Cycle limit: USD 6,000.00 per statement cycle.\n15 Sep chair USD 620.00\n"
         "16 Sep taxi USD 1,240.00\n17 Sep hotel USD 2,450.50\n18 Sep books USD 1,900.00\n"
         "Previous dose: 3 September 2026 at 14:30. Next dose 01:30 on 4 September 2026.\n"
         "The plan includes 2.40 TiB.")
    text = "\n".join(facts.facts(s, ["Which purchase was declined first?"], None))
    assert "running total USD first reaches 6,000 at value #4 (1,900)" in text
    assert "11 h elapsed from 2026-09-03 14:30 to 2026-09-04 01:30" in text
    assert "2 days counting both ends" in text
    assert "2.40 TiB = 2638.83 GB" in text
    assert facts.quantities("EUR 692.00 and $40")[0] == (692.0, "EUR", "EUR 692.00")


def test_facts_percent_of_amount():
    f = "\n".join(facts.facts("Alert at 80% of the monthly budget USD 12,000.", [], None))
    assert "80% of 12,000 USD = 9,600 USD" in f


def test_facts_no_reference_no_relative():
    assert not any("reference" in x for x in facts.facts("due tomorrow", [], None))


def test_options_and_state_text():
    assert options({"type": "noul", "criteria": None}) == (["true", "false"], ["Yes", "No"])
    assert options({"type": "score", "criteria": ["a", "b", "c"]})[0] == ["0", "1", "2"]
    assert options({"type": "choice", "criteria": {"x": "X", "y": "Y"}}) == (["x", "y"], ["X", "Y"])
    assert state_text({"a": 1, "b": "two"}) == "a: 1\nb: two"


class FakeTok:
    """Whitespace tokenizer with offsets, enough to test the serializer's bookkeeping."""
    def __init__(self):
        self.vocab = {}

    def get_vocab(self):
        return self.vocab

    def add_tokens(self, toks, special_tokens=True):
        for t in toks:
            self.vocab.setdefault(t, len(self.vocab))

    def convert_tokens_to_ids(self, t):
        return self.vocab[t]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs, i = [], [], 0
        for w in text.split():
            j = text.index(w, i)
            self.vocab.setdefault(w, len(self.vocab))
            ids.append(self.vocab[w])
            offs.append((j, j + len(w)))
            i = j + len(w)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


def test_serializer_positions_and_evidence():
    ser = Serializer(FakeTok(), max_prefix_tokens=64, use_facts=False)
    q = {"type": "choice", "instructions": "Route it", "criteria": {"a": "Alpha", "b": "Beta"}}
    p = ser.prefix("the card was declined twice today", [q])
    b = ser.block(q)
    mark = ser.special["<|mark|>"]
    assert len(b.marks) == 3 and all(b.ids[m] == mark for m in b.marks)
    assert b.ids[b.end] == ser.special["<|/q|>"] and b.keys == ["a", "b"]
    assert p.ids[0] == ser.special["<|state|>"]
    ev = evidence_mask(p, ["declined twice"])
    assert ev == [0, 0, 0, 1, 1, 0]
    long = ser.prefix(" ".join(f"w{i}" for i in range(200)), [q])
    assert long.truncated and len(long.ids) <= 64


def test_bootstrap_is_poisson_like():
    ws = [bootstrap_weight(f"id{i}", h, 0) for i in range(4000) for h in range(2)]
    mean = sum(ws) / len(ws)
    assert 0.9 < mean < 1.1
    assert 0.3 < ws.count(0) / len(ws) < 0.44          # P(0) = e^-1 ~ 0.368
    assert bootstrap_weight("x", 0, 0) == bootstrap_weight("x", 0, 0)


def _item(target, qtype="choice", abstain=0.0):
    from noma.model.serialize import Block
    K = len(target) if target else 3
    blk = Block(list(range(K + 3)), list(range(1, K + 2)), K + 2, [str(i) for i in range(K)], qtype,
                list(range(K)) if qtype == "score" else [])
    pre = Prefix([0, 1, 2], (1, 2), [(0, 1)], "x")
    return Item("i", "fam", pre, blk, target, abstain, None, 6)


def test_loss_prefers_correct_answer_and_heads_shapes():
    H, d = 2, 16
    heads = Heads(d, n_heads=H, d=32, layers=1, heads=4, ff=64)
    hidden = torch.randn(1, 10, d)
    x, roles, ords, pad = gather_slots(hidden, [0], [[0, 1, 2, 3, 4]], [[]])
    opt, ab = heads(x, roles, ords, pad)
    assert opt.shape == (H, 1, 5) and ab.shape == (H, 1)
    it = _item([1.0, 0.0, 0.0])
    good = torch.tensor([[[0.0, 5.0, -5.0, -5.0, 0.0]]]).expand(H, 1, 5)
    bad = torch.tensor([[[0.0, -5.0, 5.0, -5.0, 0.0]]]).expand(H, 1, 5)
    ab0 = torch.full((H, 1), -5.0)
    lg, st = losses.compute(good, ab0, [torch.zeros(1)], [it], 0, bootstrap=False)
    lb, _ = losses.compute(bad, ab0, [torch.zeros(1)], [it], 0, bootstrap=False)
    assert lg < lb and st["acc"] == 1.0
    ab_item = _item(None, abstain=1.0)
    l_hi, _ = losses.compute(good, torch.full((H, 1), 5.0), [torch.zeros(1)], [ab_item], 0, False)
    l_lo, _ = losses.compute(good, ab0, [torch.zeros(1)], [ab_item], 0, False)
    assert l_hi < l_lo


def test_special_tokens_are_fixed():
    assert len(SPECIAL) == len(set(SPECIAL)) == 9
