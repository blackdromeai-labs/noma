"""Train Noma.

    # sanity check: overfit 200 examples
    python -m noma.train.train --overfit 200 --epochs 30 --out runs/dry/overfit
    # dry run on 2k examples
    python -m noma.train.train --limit 2000 --calib-limit 300 --out runs/dry/2k

Saves only LoRA, the new-token embeddings, and the heads (plus fitted temperatures).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from noma.model.heads import TYPE_INDEX
from noma.model.noma import Noma, NomaConfig

from . import losses
from .data import batches, encode, load_jsonl

ROOT = Path(__file__).resolve().parents[2]


def param_groups(m: Noma, lora_lr: float, head_lr: float):
    lora = [p for n, p in m.body.named_parameters() if p.requires_grad]
    # The new-token embeddings feed every scored position; moving them at the head rate shifts
    # the features under the heads, so they follow the backbone (LoRA) rate.
    groups = [{"params": [m.embed.table], "lr": lora_lr if lora_lr > 0 else head_lr * 0.1,
               "weight_decay": 0.0},
              {"params": [p for p in m.heads.parameters() if p.requires_grad], "lr": head_lr,
               "weight_decay": 0.01}]
    if lora_lr > 0:
        groups.insert(0, {"params": lora, "lr": lora_lr, "weight_decay": 0.0})
    else:
        for p in lora:
            p.requires_grad_(False)
    return groups


@torch.no_grad()
def collect(m: Noma, items, max_tokens, max_items):
    """Per item: head logits over its options [H, K] and abstain logits [H]."""
    m.eval()
    out = []
    for b in batches(items, max_tokens, max_items, seed=0, shuffle=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            opt, ab, _ = m.forward_items([(it.prefix, it.block) for it in b])
        for i, it in enumerate(b):
            K = len(it.block.keys)
            out.append((it, opt[:, i, 1:1 + K].float().cpu(), ab[:, i].float().cpu()))
    return out


def fit_temperatures(m: Noma, collected) -> dict:
    """Per question type, the temperature minimizing log loss of the head-averaged probabilities
    against the soft targets; one abstain temperature by BCE. Grid search in log space."""
    grid = [math.exp(x / 20) for x in range(-30, 41)]   # ~0.22 .. 7.4
    res = {}
    for qtype, ti in TYPE_INDEX.items():
        rows = [(lg, torch.tensor(it.target)) for it, lg, _ in collected
                if it.block.qtype == qtype and it.target is not None]
        if len(rows) < 20:
            continue
        def nll(t):
            return sum(float(-(y * torch.softmax(lg / t, -1).mean(0).clamp_min(1e-9).log()).sum())
                       for lg, y in rows) / len(rows)
        best = min(grid, key=nll)
        m.heads.temperature[ti] = best
        res[qtype] = {"T": best, "nll_before": nll(1.0), "nll_after": nll(best), "n": len(rows)}
    rows = [(a, it.abstain) for it, _, a in collected]
    def bce(t):
        return sum(float(torch.nn.functional.binary_cross_entropy(
            torch.sigmoid(a / t).mean().clamp(1e-6, 1 - 1e-6), torch.tensor(y)))
            for a, y in rows) / len(rows)
    best = min(grid, key=bce)
    m.heads.abstain_temperature[0] = best
    res["abstain"] = {"T": best, "bce_before": bce(1.0), "bce_after": bce(best)}
    return res


def evaluate(collected, temps_applied: bool = True) -> dict:
    """Accuracy (argmax vs target argmax), top-label ECE (10 bins), Brier, abstain accuracy."""
    import collections
    per_fam = collections.defaultdict(lambda: [0, 0])
    confs, hits, briers, ab_ok, ab_n = [], [], [], 0, 0
    for it, lg, a in collected:
        pa = float(torch.sigmoid(a).mean())
        ab_ok += int((pa >= 0.5) == (it.abstain >= 0.5))
        ab_n += 1
        if it.target is None:
            continue
        p = torch.softmax(lg, -1).mean(0)
        y = torch.tensor(it.target)
        k = int(p.argmax())
        hit = int(k == int(y.argmax()))
        confs.append(float(p[k]))
        hits.append(hit)
        briers.append(float(((p - y) ** 2).sum()))
        per_fam[it.family][0] += hit
        per_fam[it.family][1] += 1
    n = len(hits)
    ece = 0.0
    for b in range(10):
        idx = [i for i, c in enumerate(confs) if b / 10 < c <= (b + 1) / 10 or (b == 0 and c == 0)]
        if idx:
            ece += len(idx) / n * abs(sum(hits[i] for i in idx) / len(idx)
                                      - sum(confs[i] for i in idx) / len(idx))
    return {"n": n, "acc": sum(hits) / max(n, 1), "ece": ece, "brier": sum(briers) / max(n, 1),
            "abstain_acc": ab_ok / max(ab_n, 1),
            "per_family": {f: round(c / t, 3) for f, (c, t) in sorted(per_fam.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--cut", type=int, default=14)
    ap.add_argument("--train", default=str(ROOT / "data/built/train.jsonl"))
    ap.add_argument("--calib", default=str(ROOT / "data/built/calib.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="random subset of train")
    ap.add_argument("--calib-limit", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=0,
                    help="sanity check: train and evaluate on the same N examples, no dropout/bootstrap")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate (0 = freeze LoRA)")
    ap.add_argument("--head-lr", type=float, default=1e-3,
                    help="learning rate of the heads and new-token embeddings")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-tokens", type=int, default=4096, help="padded tokens per micro-batch")
    ap.add_argument("--max-items", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-prefix", type=int, default=1536)
    ap.add_argument("--no-facts", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-bad", type=int, default=10,
                    help="stop after this many skipped non-finite batches")
    ap.add_argument("--save-every", type=int, default=200, help="save a partial checkpoint every N steps")
    ap.add_argument("--encoded", default=None,
                    help="pre-encoded cache from noma.train.encode (skips encoding on the GPU box)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = NomaConfig(backbone=args.backbone, cut=args.cut, max_prefix_tokens=args.max_prefix,
                     use_facts=not args.no_facts,
                     lora_dropout=0.0 if args.overfit else 0.05)
    m = Noma(cfg).cuda()
    if args.overfit:
        for mod in m.heads.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
    if args.grad_ckpt:
        m.body.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    t0 = time.time()
    if args.encoded:
        import pickle
        from .encode import fingerprint, settings
        cache = pickle.loads(Path(args.encoded).read_bytes())
        if cache["fingerprint"] != fingerprint(m.tok) or cache["settings"] != settings(m.ser):
            raise SystemExit(f"encoded cache does not match this model: {cache['fingerprint']} "
                             f"{cache['settings']} vs {fingerprint(m.tok)} {settings(m.ser)}")
        train_items, calib_items = cache["train"], cache["calib"]
        n = args.overfit or args.limit
        if n and n < len(train_items):
            train_items = random.Random(args.seed).sample(train_items, n)
        if args.calib_limit and args.calib_limit < len(calib_items):
            calib_items = random.Random(1).sample(calib_items, args.calib_limit)
        if args.overfit:
            calib_items = train_items
    else:
        rows = load_jsonl(args.train, args.overfit or args.limit, args.seed)
        train_items = [encode(m.ser, r) for r in rows]
        if args.overfit:
            calib_items = train_items
        else:
            calib_items = [encode(m.ser, r) for r in load_jsonl(args.calib, args.calib_limit, 1)]
    ntok = sum(it.n_tokens for it in train_items)
    print(f"encoded {len(train_items)} train / {len(calib_items)} calib items in "
          f"{time.time() - t0:.0f}s; {ntok / len(train_items):.0f} tokens/item, "
          f"{sum(it.prefix.truncated for it in train_items)} truncated, "
          f"{sum(it.evidence is not None for it in train_items)} with evidence", flush=True)

    n_epochs = math.ceil(args.epochs)
    plan = []
    for e in range(n_epochs):
        bs = batches(train_items, args.max_tokens, args.max_items, seed=args.seed + e)
        if e == n_epochs - 1 and args.epochs < n_epochs:
            bs = bs[:max(1, int(len(bs) * (args.epochs - e)))]
        plan += bs
    steps = math.ceil(len(plan) / args.accum)
    opt = torch.optim.AdamW(param_groups(m, args.lr, args.head_lr), betas=(0.9, 0.98))
    base_lrs = [g["lr"] for g in opt.param_groups]
    warm = max(1, int(args.warmup * steps))

    def set_lr(step):
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(f, 0.02)

    params = [p for g in opt.param_groups for p in g["params"]]
    log = []
    m.train()
    t0 = time.time()
    step = 0
    bad_batches: list[dict] = []
    run = {"loss": 0.0, "acc": 0.0, "n": 0}
    for i, b in enumerate(plan):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, a, ev = m.forward_items([(it.prefix, it.block) for it in b])
        loss, st = losses.compute(o, a, ev, b, seed=args.seed, bootstrap=not args.overfit)
        if not torch.isfinite(loss):
            # An earlier run went NaN at one batch and the NaN gradient then poisoned every weight for
            # the rest of the run. A non-finite batch is now skipped and recorded instead.
            bad_batches.append({"batch": i, "ids": [it.id for it in b], "why": "loss"})
            opt.zero_grad(set_to_none=True)
            if len(bad_batches) > args.max_bad:
                raise SystemExit(f"{len(bad_batches)} non-finite batches; stopping: {bad_batches[-1]}")
            continue
        (loss / args.accum).backward()
        run["loss"] += float(loss)
        run["acc"] += st["acc"]
        run["n"] += 1
        if (i + 1) % args.accum == 0 or i == len(plan) - 1:
            norms = [torch.nn.utils.clip_grad_norm_(g["params"], 1.0)  # clip per group: large
                     for g in opt.param_groups]                         # head grads must not
            if not all(torch.isfinite(n) for n in norms):                # shrink the LoRA step
                bad_batches.append({"batch": i, "ids": [it.id for it in b], "why": "grad"})
                opt.zero_grad(set_to_none=True)
                if len(bad_batches) > args.max_bad:
                    raise SystemExit(f"{len(bad_batches)} non-finite batches; stopping")
                continue
            set_lr(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if args.save_every and step % args.save_every == 0:
                m.save(out / "partial")
                (out / "partial" / "step.json").write_text(json.dumps(
                    {"step": step, "bad_batches": bad_batches}), encoding="utf-8")
            if step % args.log_every == 0 or step == steps:
                rec = {"step": step, "of": steps, "loss": run["loss"] / run["n"],
                       "acc": run["acc"] / run["n"], "lr": opt.param_groups[0]["lr"],
                       "s": round(time.time() - t0), "mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                log.append(rec)
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in rec.items()}), flush=True)
                run = {"loss": 0.0, "acc": 0.0, "n": 0}

    coll = collect(m, calib_items, args.max_tokens, args.max_items)
    before = evaluate(coll)
    temps = fit_temperatures(m, coll) if not args.overfit else {}
    coll = collect(m, calib_items, args.max_tokens, args.max_items)
    for i, (it, lg, a) in enumerate(coll):  # apply fitted temperatures for the report
        t = float(m.heads.temperature[TYPE_INDEX[it.block.qtype]])
        coll[i] = (it, lg / t, a / float(m.heads.abstain_temperature))
    after = evaluate(coll)
    m.save(out)
    summary = {"args": vars(args), "train_items": len(train_items), "steps": steps,
               "bad_batches": bad_batches,
               "seconds": round(time.time() - t0), "eval_before_temps": before,
               "temperatures": temps, "eval_after_temps": after, "log": log}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"eval_before_temps": before, "temperatures": temps,
                      "eval_after_temps": after}, indent=1))


if __name__ == "__main__":
    main()
