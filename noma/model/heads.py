"""Listwise option scorer (and a pointwise baseline), its ensemble, and the evidence head."""

from __future__ import annotations

import torch
import torch.nn as nn

MAX_LEVELS = 64
ROLE_Q, ROLE_OPT, ROLE_ABSTAIN = 0, 1, 2


class ListwiseScorer(nn.Module):
    """A small bidirectional transformer over [q_summary, opt_1..opt_K, abstain], so options are
    scored against each other rather than one at a time."""

    def __init__(self, d_in: int, d: int = 512, layers: int = 2, heads: int = 8, ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
        self.role = nn.Embedding(3, d)
        self.ordinal = nn.Embedding(MAX_LEVELS + 1, d)  # 0 = not an ordinal option
        nn.init.zeros_(self.ordinal.weight)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.opt_out = nn.Linear(d, 1)
        self.abstain_out = nn.Linear(d, 1)

    def forward(self, x, roles, ordinals, pad):
        """x [B, T, d_in]; roles/ordinals [B, T]; pad [B, T] True where padding.
        Returns option logits [B, T] (only ROLE_OPT slots are meaningful) and abstain logit [B]."""
        h = self.proj(x.float()) + self.role(roles) + self.ordinal(ordinals)
        h = self.enc(h, src_key_padding_mask=pad)
        opt = self.opt_out(h).squeeze(-1)
        ab_idx = (roles == ROLE_ABSTAIN).float().argmax(1)
        ab = self.abstain_out(h[torch.arange(h.size(0), device=h.device), ab_idx]).squeeze(-1)
        return opt, ab


class PointwiseScorer(nn.Module):
    """Baseline: each option is scored alone from [its mark state, q_summary] by an MLP, with no
    attention across options. Same inputs and outputs as ListwiseScorer."""

    def __init__(self, d_in: int, d: int = 512, ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
        self.role = nn.Embedding(3, d)
        self.ordinal = nn.Embedding(MAX_LEVELS + 1, d)
        nn.init.zeros_(self.ordinal.weight)
        self.mlp = nn.Sequential(nn.Linear(2 * d, ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff, d), nn.GELU(), nn.Dropout(dropout))
        self.opt_out = nn.Linear(d, 1)
        self.abstain_out = nn.Linear(d, 1)

    def forward(self, x, roles, ordinals, pad):
        h = self.proj(x.float()) + self.role(roles) + self.ordinal(ordinals)
        h = self.mlp(torch.cat([h, h[:, :1].expand_as(h)], -1))     # slot 0 is q_summary
        opt = self.opt_out(h).squeeze(-1)
        ab_idx = (roles == ROLE_ABSTAIN).float().argmax(1)
        ab = self.abstain_out(h[torch.arange(h.size(0), device=h.device), ab_idx]).squeeze(-1)
        return opt, ab


SCORERS = {"listwise": ListwiseScorer, "pointwise": PointwiseScorer}


class Heads(nn.Module):
    def __init__(self, d_in: int, n_heads: int = 4, kind: str = "listwise", **kw):
        super().__init__()
        self.scorers = nn.ModuleList(SCORERS[kind](d_in, **kw) for _ in range(n_heads))
        self.evidence = nn.Linear(d_in, 1)
        # Post-hoc temperatures, fitted on the calibration split.
        self.register_buffer("temperature", torch.ones(3))   # choice, noul, score
        self.register_buffer("abstain_temperature", torch.ones(1))

    def forward(self, x, roles, ordinals, pad):
        """Returns option logits [H, B, T] and abstain logits [H, B]."""
        outs = [s(x, roles, ordinals, pad) for s in self.scorers]
        return torch.stack([o for o, _ in outs]), torch.stack([a for _, a in outs])


TYPE_INDEX = {"choice": 0, "noul": 1, "score": 2}


def gather_slots(hidden, rows, positions, orders):
    """Build scorer inputs from backbone hidden states.

    hidden [N, L, d]; for item i, rows[i] is its row in hidden, positions[i] = [q_end, mark_1..
    mark_K, abstain_mark] (absolute positions in that row), orders[i] = ordinal per option (score)
    or None. Returns x [B, T, d], roles [B, T], ordinals [B, T], pad [B, T]."""
    B = len(positions)
    T = max(len(p) for p in positions)
    d = hidden.size(-1)
    x = hidden.new_zeros(B, T, d)
    roles = torch.zeros(B, T, dtype=torch.long, device=hidden.device)
    ords = torch.zeros(B, T, dtype=torch.long, device=hidden.device)
    pad = torch.ones(B, T, dtype=torch.bool, device=hidden.device)
    for i, (r, pos) in enumerate(zip(rows, positions)):
        n = len(pos)
        x[i, :n] = hidden[r, torch.tensor(pos, device=hidden.device)]
        roles[i, 0] = ROLE_Q
        roles[i, 1:n - 1] = ROLE_OPT
        roles[i, n - 1] = ROLE_ABSTAIN
        if orders[i]:
            ords[i, 1:n - 1] = torch.tensor([min(o + 1, MAX_LEVELS) for o in orders[i]],
                                            device=hidden.device)
        pad[i, :n] = False
    return x, roles, ords, pad
