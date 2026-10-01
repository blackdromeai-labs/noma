"""Serving latency benchmark (model time only, GPU-synchronized).

    python -m noma.serve.bench --backbone Qwen/Qwen3.5-4B-Base --cut 18 --tasks data/eval

Modes, each on a fresh process state per mode is not needed because kernels are keyed by
shape: the fork mode runs first (cold compile per new length), then bucketed modes with a
start-up warmup. Reports first-pass (cold) and second-pass (warm) p50/p95 on JevBench hard
(long) and original (short) states, plus the largest probability difference between modes
(the bucketed path must give the same answers).
Latency does not depend on trained weights, so an untrained model times the same as a
trained one of the same architecture.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from noma.model.noma import Noma, NomaConfig


def _reqs(path: Path, n: int):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[:n]:
        t = json.loads(line)
        q = {k: v for k, v in t["question"].items() if k in ("type", "instructions", "criteria")}
        out.append((t["state"], {"decision": q}))
    return out


def _pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1)


def run_mode(m: Noma, reqs, fork):
    def one(state, qs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, n = m.decide(state, qs, "2026-09-29T10:00:00Z", fork=fork)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000, n, out
    first = [one(*r) for r in reqs]
    second = [one(*r) for r in reqs]
    return first, second


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--cut", type=int, default=18)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks", default="data/eval")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-prefix", type=int, default=4096)
    ap.add_argument("--compile", action="store_true", help="also try torch.compile reduce-overhead")
    args = ap.parse_args()
    if args.ckpt and (Path(args.ckpt) / "model.safetensors").exists():   # an exported model
        m = Noma.from_pretrained(args.ckpt, device="cuda")
    else:
        m = (Noma.load(args.ckpt) if args.ckpt else
             Noma(NomaConfig(backbone=args.backbone, cut=args.cut,
                             max_prefix_tokens=args.max_prefix))).cuda().eval()
    sets = {"hard": _reqs(Path(args.tasks) / "jevbench_hard.jsonl", args.n),
            "short": _reqs(Path(args.tasks) / "jevbench_original.jsonl", args.n)}
    res = {"gpu": torch.cuda.get_device_name(0), "modes": {}}
    ref_out = {}

    plans = [("fork_nobucket", None, True, False), ("flat_bucket128", 128, "auto", False),
             ("flat_bucket128_graphs", 128, "auto", True)]
    for name, bucket, fork, graphs in plans:
        m.bucket = bucket
        m._graphs = None
        t = time.perf_counter()
        lens = m.warmup(graphs=graphs) if bucket else []
        warm_s = time.perf_counter() - t
        r = {"bucket": bucket, "warmup_s": round(warm_s, 1), "warmup_lengths": len(lens)}
        for sname, reqs in sets.items():
            first, second = run_mode(m, reqs, fork)
            r[sname] = {"cold_p50": _pct([x[0] for x in first], .5),
                        "cold_p95": _pct([x[0] for x in first], .95),
                        "warm_p50": _pct([x[0] for x in second], .5),
                        "warm_p95": _pct([x[0] for x in second], .95),
                        "tokens_p50": _pct([x[1] for x in first], .5)}
            outs = [x[2] for x in second]
            if sname not in ref_out:
                ref_out[sname] = outs
            else:
                r[sname]["max_prob_diff_vs_fork"] = round(max(
                    abs(a["decision"][0][k] - b["decision"][0][k])
                    for a, b in zip(ref_out[sname], outs) for k in a["decision"][0]), 5)
        res["modes"][name] = r
        print(json.dumps({name: r}), flush=True)

    if args.compile:
        m.bucket = 256
        try:
            m.body = torch.compile(m.body, mode="reduce-overhead", dynamic=False)
            t = time.perf_counter()
            m.warmup()
            r = {"bucket": 256, "warmup_s": round(time.perf_counter() - t, 1)}
            for sname, reqs in sets.items():
                first, second = run_mode(m, reqs, "auto")
                r[sname] = {"cold_p50": _pct([x[0] for x in first], .5),
                            "warm_p50": _pct([x[0] for x in second], .5),
                            "warm_p95": _pct([x[0] for x in second], .95)}
        except Exception as e:  # compile support for the hybrid model is not guaranteed
            r = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        res["modes"]["flat_bucket256_compiled"] = r
        print(json.dumps({"flat_bucket256_compiled": r}), flush=True)
    Path("bench_result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
