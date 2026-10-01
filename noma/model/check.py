"""prefix-fork equals running each question separately (exact in fp32; bf16 drift
reported).

    python -m noma.model.check [--ckpt runs/dry/overfit]

Compares backbone hidden states at every scored slot (<|/q|> and each <|mark|>) between
(1) prefix run once, cache forked across all question blocks, and (2) each [prefix + block] run
as its own sequence. A control (block run with NO prefix) shows what a real mismatch looks like.
"""

from __future__ import annotations

import argparse
import json

import torch

from .noma import Noma, NomaConfig, expand_cache

STATE = ("Invoice INV-2231 for $620 was issued on 2026-03-02 and is due 2026-03-16. The customer "
         "paid $500 on March 10, 2026 and asked for the remaining $120 to be waived because the "
         "card was declined twice during the outage. Account tier: Gold since 2019.")
QUESTIONS = {
    "late": {"type": "noul", "instructions": "Is the invoice overdue?",
             "criteria": {"true": "overdue", "false": "not overdue"}},
    "dept": {"type": "choice", "instructions": "Route this ticket.",
             "criteria": {"billing": "Billing team", "tech": "Technical support",
                          "sales": "Sales", "retention": "Retention desk"}},
    "sev": {"type": "score", "instructions": "How urgent is it?",
            "criteria": ["low", "medium", "high", "critical"]},
}


@torch.no_grad()
def compare(m: Noma, state=STATE, questions=QUESTIONS, ref="2026-03-20T09:00:00Z") -> dict:
    dev = m.heads.temperature.device
    qs = list(questions.values())
    prefix = m.ser.prefix(state, qs, ref)
    blocks = [m.ser.block(q) for q in qs]
    pad = m.tok.pad_token_id or 0

    def padded(seqs):
        L = max(len(s) for s in seqs)
        t = torch.full((len(seqs), L), pad, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            t[i, :len(s)] = torch.tensor(s, device=dev)
        return t

    _, cache = m.hidden(torch.tensor([prefix.ids], device=dev), use_cache=True)
    expand_cache(cache, len(blocks))
    h_fork, _ = m.hidden(padded([b.ids for b in blocks]), past_key_values=cache, use_cache=True)
    h_sep, _ = m.hidden(padded([prefix.ids + b.ids for b in blocks]))
    h_ctl, _ = m.hidden(padded([b.ids for b in blocks]))
    P = len(prefix.ids)
    rel, ctl = [], []
    for i, b in enumerate(blocks):
        for pos in [b.end] + b.marks:
            a, s, c = h_fork[i, pos].float(), h_sep[i, P + pos].float(), h_ctl[i, pos].float()
            rel.append(float((a - s).norm() / s.norm()))
            ctl.append(float((c - s).norm() / s.norm()))
    return {"slots": len(rel), "fork_vs_separate_max_rel": max(rel),
            "fork_vs_separate_mean_rel": sum(rel) / len(rel),
            "control_no_prefix_mean_rel": sum(ctl) / len(ctl)}


def prob_drift(m: Noma) -> dict:
    """Largest change in any output probability between fork and separate runs."""
    a, _ = m.decide(STATE, QUESTIONS, "2026-03-20T09:00:00Z", fork=True)
    b, _ = m.decide(STATE, QUESTIONS, "2026-03-20T09:00:00Z", fork=False)
    return {"max_prob_diff": max(abs(a[k][0][o] - b[k][0][o]) for k in a for o in a[k][0]),
            "max_abstain_diff": max(abs(a[k][1] - b[k][1]) for k in a)}


def main() -> None:
    """Exactness is judged in fp32 (bf16 rounding differs between one long pass and a split
    pass, so bf16 can only show closeness). bf16 drift is reported separately: it is what a
    served answer can differ from the training-path answer by."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    load = (lambda dt: Noma.load(args.ckpt, dtype=dt)) if args.ckpt else (
        lambda dt: Noma(NomaConfig(), dtype=dt))
    m = load(torch.float32).cuda().eval()
    r = {"fp32": compare(m)}
    del m
    torch.cuda.empty_cache()
    m = load(torch.bfloat16).cuda().eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r["bf16"] = compare(m) | prob_drift(m)
    r["pass"] = r["fp32"]["fork_vs_separate_max_rel"] < args.tol < r["fp32"]["control_no_prefix_mean_rel"]
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
