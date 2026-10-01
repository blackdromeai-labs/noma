"""Training records (data/built/*.jsonl) -> encoded items and length-bucketed batches."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from noma.model.serialize import Block, Prefix, Serializer, evidence_mask


@dataclass
class Item:
    id: str
    family: str
    prefix: Prefix
    block: Block
    target: list[float] | None   # soft distribution over block.keys; None for abstain items
    abstain: float
    evidence: list[float] | None
    n_tokens: int


def load_jsonl(path: Path, limit: int | None = None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def encode(ser: Serializer, r: dict) -> Item:
    q = r["question"]
    prefix = ser.prefix(r["state"], [q], r.get("reference_time"))
    block = ser.block(q)
    target = [float(r["target"].get(k, 0.0)) for k in block.keys] if r.get("target") else None
    if target is not None:
        s = sum(target)
        target = [t / s for t in target] if s > 0 else None
    ev = evidence_mask(prefix, r.get("evidence")) if not prefix.truncated else None
    return Item(r["id"], r["family"], prefix, block, target, float(r.get("abstain") or 0.0), ev,
                len(prefix.ids) + len(block.ids))


def batches(items: list[Item], max_tokens: int, max_items: int, seed: int,
            shuffle: bool = True) -> list[list[Item]]:
    """Length-bucketed: sort within random chunks, cut by padded token budget, shuffle batches."""
    rng = random.Random(seed)
    order = items[:]
    if shuffle:
        rng.shuffle(order)
    out = []
    chunk = 50 * max_items
    for i in range(0, len(order), chunk):
        part = sorted(order[i:i + chunk], key=lambda x: x.n_tokens)
        cur: list[Item] = []
        for it in part:
            L = max([x.n_tokens for x in cur] + [it.n_tokens])
            if cur and (L * (len(cur) + 1) > max_tokens or len(cur) >= max_items):
                out.append(cur)
                cur = []
            cur.append(it)
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def bootstrap_weight(item_id: str, head: int, seed: int) -> int:
    """Online bootstrap: a Poisson(1) count per (item, head), fixed by hashing, so each ensemble
    head trains on its own resample of the data."""
    u = int(hashlib.sha256(f"{seed}:{head}:{item_id}".encode()).hexdigest()[:12], 16) / 16 ** 12
    # Inverse CDF of Poisson(1).
    k, p, c = 0, 0.36787944117144233, 0.36787944117144233
    while u > c and k < 8:
        k += 1
        p /= k
        c += p
    return k
