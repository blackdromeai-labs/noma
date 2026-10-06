"""External baseline: the untrained base model, all layers, scoring options through its
language-model head.

    python -m noma.eval.lm_baseline --backbone Qwen/Qwen3.5-4B-Base --eval data/eval \
        --out baseline.json --dump baseline.jsonl

Each item becomes a plain prompt (state, question, numbered options, "Answer:"). The score of
an option is the log-probability of its number as the continuation; a softmax over the options
gives the distribution. Numbers are zero-padded to one width per item, so no option's label is
a prefix of another's. Nothing is trained or calibrated. The dump has the same columns as
noma.eval.quick --dump (no item text).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from noma.eval.quick import SETS
from noma.model.serialize import options, state_text

MAX_STATE_TOKENS = 4096


def rows(path: Path) -> list[dict]:
    """The items noma.probe.capacity.eval_items scores: an expected answer that is an option."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t.get("expected") is None:
            continue
        q = t["question"]
        exp = str(t["expected"])
        if q["type"] == "noul":
            exp = {"yes": "true", "no": "false"}.get(exp, exp)
        keys, descs = options(q)
        if exp in keys:
            out.append({"id": t["id"], "family": t["family"], "qtype": q["type"], "keys": keys,
                        "descs": descs, "expected": exp, "state": state_text(t["state"]),
                        "instructions": q.get("instructions") or ""})
    return out


def labels(n: int) -> list[str]:
    w = len(str(n))
    return [str(i + 1).zfill(w) for i in range(n)]


def prompt(tok, r: dict) -> tuple[list[int], bool]:
    ids = tok(r["state"], add_special_tokens=False)["input_ids"]
    cut = len(ids) > MAX_STATE_TOKENS
    state = r["state"]
    if cut:   # keep the head and the tail, as the Noma serialiser does
        h = MAX_STATE_TOKENS // 2
        state = tok.decode(ids[:h]) + "\n[...]\n" + tok.decode(ids[-h:])
    lab = labels(len(r["keys"]))
    kind = {"noul": "Answer the question about the state.",
            "score": "Pick the level that fits the state.",
            "choice": "Pick the option that fits the state."}[r["qtype"]]
    lines = []
    for l, k, d in zip(lab, r["keys"], r["descs"]):
        lines.append(f"{l}. {d}" if r["qtype"] != "choice" or k == d else f"{l}. {k}: {d}")
    text = (f"State:\n{state}\n\n{kind}\nQuestion: {r['instructions']}\nOptions:\n"
            + "\n".join(lines) + "\n\nReply with the number of one option.\nAnswer:")
    return tok(text)["input_ids"], cut


@torch.no_grad()
def option_logprobs(lm, tok, ids: list[int], n: int, max_tokens: int) -> torch.Tensor:
    """log P(label_k | prompt) for the n labels, exact over their token sequences."""
    conts = [tok(" " + l, add_special_tokens=False)["input_ids"] for l in labels(n)]
    prefixes = sorted({tuple(c[:i]) for c in conts for i in range(len(c))}, key=lambda p: (len(p), p))
    nxt = {}
    by_len = {}
    for p in prefixes:
        by_len.setdefault(len(p), []).append(p)
    for L, group in by_len.items():
        step = max(1, max_tokens // (len(ids) + L))
        for i in range(0, len(group), step):
            chunk = group[i:i + step]
            x = torch.tensor([ids + list(p) for p in chunk], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = lm(input_ids=x).logits[:, -1].float()
            lp = torch.log_softmax(logits, -1).cpu()
            for p, row in zip(chunk, lp):
                nxt[p] = row
    return torch.tensor([sum(float(nxt[tuple(c[:i])][c[i]]) for i in range(len(c))) for c in conts])


def score(res: list[dict]) -> dict:
    n = len(res)
    hits = [r["correct"] for r in res]
    confs = [max(r["probs"]) for r in res]
    fam = {}
    for r in res:
        f = fam.setdefault(r["family"], [0, 0])
        f[0] += r["correct"]
        f[1] += 1
    ece = 0.0
    for lo in range(10):
        sel = [i for i, c in enumerate(confs) if lo / 10 < c <= (lo + 1) / 10]
        if sel:
            ece += len(sel) / n * abs(sum(hits[i] for i in sel) / len(sel) - sum(confs[i] for i in sel) / len(sel))
    return {"n": n, "acc": round(sum(hits) / n, 4), "ece": round(ece, 4),
            "per_family": {k: f"{a}/{b}" for k, (a, b) in sorted(fam.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--max-tokens", type=int, default=24000, help="tokens per forward batch")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    lm = AutoModelForCausalLM.from_pretrained(args.backbone, dtype=torch.bfloat16).cuda().eval()
    out = {"backbone": args.backbone, "method": "zero-shot option scoring through the LM head, all layers"}
    with open(args.dump, "w", encoding="utf-8") as dump:
        for name in [s for s in args.sets.split(",") if s]:
            path = Path(args.eval) / f"{name}.jsonl"
            if not path.exists():
                continue
            res, cut = [], 0
            for r in rows(path):
                ids, c = prompt(tok, r)
                cut += c
                p = torch.softmax(option_logprobs(lm, tok, ids, len(r["keys"]), args.max_tokens), -1)
                k = int(p.argmax())
                row = {"set": name, "id": r["id"], "family": r["family"], "qtype": r["qtype"],
                       "keys": r["keys"], "expected": r["expected"], "scored": True,
                       "probs": [float(x) for x in p], "predicted": r["keys"][k],
                       "correct": r["keys"][k] == r["expected"]}
                res.append(row)
                dump.write(json.dumps(row) + "\n")
                dump.flush()
            out[name] = {**score(res), "truncated_states": cut}
            print(json.dumps({name: {k: out[name][k] for k in ("n", "acc", "ece")}}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
