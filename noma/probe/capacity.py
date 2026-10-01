"""Capacity test: does a deeper / bigger frozen backbone beat our cut? (more data alone
did not move the hard tier, so depth and size were tested directly).

    python -m noma.probe.capacity --backbone Qwen/Qwen3.5-4B-Base --layers 18,24,32 \
        --encoded data/built/encoded.pkl --eval data/eval --n-train 8000 --out cap_4b.json

The backbone is frozen (no LoRA). Features at the scored slots (<|/q|> and each <|mark|>) are
extracted once per layer; then the same listwise ensemble heads (noma.model.heads) are trained
on each layer's features with the training loss (noma.train.losses) and evaluated on the sealed
set and JevBench hard. Same data, heads, loss and budget for every layer, so the only variable
is which layer (and which model) the heads read. Evaluation items are never trained on.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path

import torch

from noma.model.heads import Heads, TYPE_INDEX, gather_slots
from noma.model.serialize import SPECIAL, Serializer
from noma.train import losses
from noma.train.data import Item, batches, encode


def eval_items(ser: Serializer, path: Path) -> list[tuple[Item, str]]:
    """JevBench Task rows -> (Item, expected option key). Items without an expected answer
    (abstain cases) are skipped: this test measures answer accuracy."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t.get("expected") is None:
            continue
        q = {k: v for k, v in t["question"].items() if k in ("type", "instructions", "criteria")}
        exp = str(t["expected"])
        if q["type"] == "noul":
            exp = {"yes": "true", "no": "false"}.get(exp, exp)
        rec = {"id": t["id"], "family": t["family"], "state": t["state"], "question": q,
               "target": None, "abstain": 0.0}
        it = encode(ser, rec)
        if exp in it.block.keys:
            out.append((it, exp))
    return out


@torch.no_grad()
def extract(model, embed, items: list[Item], layers: list[int], max_tokens: int) -> dict:
    """{layer: [tensor(n_slots, d) fp16 on CPU] aligned with items}."""
    feats = {L: [None] * len(items) for L in layers}
    index = {id(it): i for i, it in enumerate(items)}
    dev = "cuda"
    for b in batches(items, max_tokens, 48, seed=0, shuffle=False):
        seqs = [it.prefix.ids + it.block.ids for it in b]
        T = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), T), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inputs_embeds=embed(ids), output_hidden_states=True, use_cache=False,
                        attention_mask={"full_attention": None, "linear_attention": None})
        hs = out.hidden_states
        for i, it in enumerate(b):
            P = len(it.prefix.ids)
            pos = torch.tensor([P + it.block.end] + [P + m for m in it.block.marks], device=dev)
            for L in layers:
                feats[L][index[id(it)]] = hs[L][i, pos].to(torch.float16).cpu()
    return feats


def _batch(feats, idx, dev):
    F = [feats[i].to(dev, torch.float32) for i in idx]
    T = max(f.shape[0] for f in F)
    hidden = torch.zeros(len(F), T, F[0].shape[1], device=dev)
    for i, f in enumerate(F):
        hidden[i, :f.shape[0]] = f
    return hidden, [list(range(f.shape[0])) for f in F]


def train_heads(feats, items, d, epochs: int, seed: int, dev="cuda") -> Heads:
    torch.manual_seed(seed)
    heads = Heads(d).to(dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.01)
    order = list(range(len(items)))
    steps = epochs * math.ceil(len(order) / 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    rng = random.Random(seed)
    heads.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for k in range(0, len(order), 64):
            idx = order[k:k + 64]
            hidden, pos = _batch(feats, idx, dev)
            b = [items[i] for i in idx]
            x, roles, ords, pad = gather_slots(hidden, list(range(len(idx))), pos,
                                               [it.block.levels for it in b])
            o, a = heads(x, roles, ords, pad)
            loss, _ = losses.compute(o, a, [torch.zeros(0)] * len(b), b, seed=seed)
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                opt.step()
            sched.step()
    return heads.eval()


@torch.no_grad()
def predict(heads, feats, items, dev="cuda"):
    """Per item: head-averaged probabilities over its option keys (T=1)."""
    out = []
    for k in range(0, len(items), 128):
        idx = list(range(k, min(k + 128, len(items))))
        hidden, pos = _batch(feats, idx, dev)
        b = [items[i] for i in idx]
        x, roles, ords, pad = gather_slots(hidden, list(range(len(idx))), pos,
                                           [it.block.levels for it in b])
        o, _ = heads(x, roles, ords, pad)
        for j, it in enumerate(b):
            K = len(it.block.keys)
            out.append(torch.softmax(o[:, j, 1:1 + K].float(), -1).mean(0).cpu())
    return out


def score(probs, items, expected) -> dict:
    hits, confs, fam = [], [], {}
    for p, it, e in zip(probs, items, expected):
        k = int(p.argmax())
        h = it.block.keys[k] == e
        hits.append(h)
        confs.append(float(p[k]))
        f = fam.setdefault(it.family, [0, 0])
        f[0] += h
        f[1] += 1
    n = len(hits)
    ece = 0.0
    for lo in range(10):
        sel = [i for i, c in enumerate(confs) if lo / 10 < c <= (lo + 1) / 10]
        if sel:
            ece += len(sel) / n * abs(sum(hits[i] for i in sel) / len(sel)
                                      - sum(confs[i] for i in sel) / len(sel))
    return {"n": n, "acc": round(sum(hits) / n, 4), "ece": round(ece, 4),
            "per_family": {k: f"{a}/{b}" for k, (a, b) in sorted(fam.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--layers", required=True, help="comma-separated hidden_states indices")
    ap.add_argument("--encoded", default="data/built/encoded.pkl")
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from noma.model.noma import NewTokenEmbedding
    from noma.train.encode import fingerprint

    tok = AutoTokenizer.from_pretrained(args.backbone)
    cache = pickle.loads(Path(args.encoded).read_bytes())
    ser = Serializer(tok, cache["settings"]["max_prefix"], cache["settings"]["use_facts"])
    if cache["fingerprint"] != fingerprint(tok):
        raise SystemExit("encoded cache was built with a different tokenizer")
    train = random.Random(0).sample(cache["train"], min(args.n_train, len(cache["train"])))
    evals = {name: eval_items(ser, Path(args.eval) / f"{name}.jsonl")
             for name in ("sealed", "jevbench_hard")}

    lm = AutoModelForCausalLM.from_pretrained(args.backbone, dtype=torch.bfloat16).cuda().eval()
    body = lm.model
    emb = body.embed_tokens
    new_ids = [ser.special[t] for t in SPECIAL]
    seeds = ["state", "end", "facts", "end", "question", "option", ":", "none", "answer"]
    init = torch.stack([emb.weight[tok(" " + w, add_special_tokens=False)["input_ids"]]
                        .float().mean(0) for w in seeds])
    embed = NewTokenEmbedding(emb, new_ids, init).cuda()
    n_layers = body.config.num_hidden_layers
    print(f"{args.backbone}: {n_layers} layers, hidden {body.config.hidden_size}; "
          f"train {len(train)}, eval " + ", ".join(f"{k} {len(v)}" for k, v in evals.items()),
          flush=True)

    t = time.time()
    all_items = train + [it for v in evals.values() for it, _ in v]
    feats = extract(body, embed, all_items, layers, args.max_tokens)
    print(f"features for {len(all_items)} items x {len(layers)} layers in {time.time() - t:.0f}s",
          flush=True)
    del lm, body
    torch.cuda.empty_cache()

    res = {"backbone": args.backbone, "n_layers": n_layers, "n_train": len(train),
           "epochs": args.epochs, "layers": {}}
    d = feats[layers[0]][0].shape[1]
    for L in layers:
        tr = feats[L][:len(train)]
        t = time.time()
        heads = train_heads(tr, train, d, args.epochs, seed=0)
        r = {"train_s": round(time.time() - t)}
        off = len(train)
        for name, v in evals.items():
            its = [it for it, _ in v]
            ef = feats[L][off:off + len(its)]
            off += len(its)
            r[name] = score(predict(heads, ef, its), its, [e for _, e in v])
        res["layers"][str(L)] = r
        print(json.dumps({f"layer {L}/{n_layers}": {k: (v["acc"] if isinstance(v, dict) else v)
                                                   for k, v in r.items()}}), flush=True)
    res["seconds"] = round(time.time() - t0)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
