"""Training objective.

L = CE(y, p) + 0.5 Brier(y, p) + 0.5 BCE(a, p_abstain) + 0.1 EMD(y, p) [score] + 0.1 BCE(evidence)

Each ensemble head's per-item loss is weighted by that head's bootstrap count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import Item, bootstrap_weight

W_BRIER, W_ABSTAIN, W_EMD, W_EVIDENCE = 0.5, 0.5, 0.1, 0.1


def compute(opt, ab, ev, items: list[Item], seed: int, bootstrap: bool = True):
    """opt [H, B, T] (slot 0 = q summary, 1..K options, K+1 abstain); ab [H, B]; ev list of
    [n_state] logits. Returns (loss, stats dict)."""
    H = opt.size(0)
    dev = opt.device
    total = opt.new_zeros(())
    parts = {"ce": 0.0, "brier": 0.0, "abstain": 0.0, "emd": 0.0, "evidence": 0.0}
    n_correct = n_scored = 0
    for i, it in enumerate(items):
        K = len(it.block.keys)
        logits = opt[:, i, 1:1 + K].float()                          # [H, K]
        w = torch.tensor([bootstrap_weight(it.id, h, seed) if bootstrap else 1 for h in range(H)],
                         dtype=torch.float32, device=dev)
        a_target = torch.full((H,), float(it.abstain), device=dev)
        l_ab = F.binary_cross_entropy_with_logits(ab[:, i].float(), a_target, reduction="none")
        per_head = W_ABSTAIN * l_ab
        parts["abstain"] += l_ab.detach().mean().item()
        if it.target is not None:
            y = torch.tensor(it.target, device=dev).unsqueeze(0).expand(H, K)
            logp = F.log_softmax(logits, -1)
            p = logp.exp()
            ce = -(y * logp).sum(-1)
            brier = ((p - y) ** 2).sum(-1)
            per_head = per_head + ce + W_BRIER * brier
            parts["ce"] += ce.detach().mean().item()
            parts["brier"] += brier.detach().mean().item()
            if it.block.qtype == "score" and K > 1:
                emd = (p.cumsum(-1) - y.cumsum(-1)).abs().sum(-1) / (K - 1)
                per_head = per_head + W_EMD * emd
                parts["emd"] += emd.detach().mean().item()
            n_scored += 1
            n_correct += int(int(p.mean(0).argmax()) == max(range(K), key=lambda k: it.target[k]))
        total = total + (per_head * w).sum() / H
        if it.evidence is not None and ev[i].numel() == len(it.evidence):
            e = torch.tensor(it.evidence, device=dev)
            l_ev = F.binary_cross_entropy_with_logits(ev[i].float(), e)
            total = total + W_EVIDENCE * l_ev
            parts["evidence"] += l_ev.detach().item()
    n = len(items)
    stats = {k: v / n for k, v in parts.items()}
    stats["acc"] = n_correct / max(n_scored, 1)
    return total / n, stats
