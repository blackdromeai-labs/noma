"""In-process evaluation of a checkpoint (release numbers come from JevBench's own
runner over HTTP against `noma serve`).

    python -m noma.eval.quick --ckpt /path/to/noma --eval /path/to/tasks --out quick.json

Same model path as serving (bucketed single pass, fitted temperatures). Items without an
expected answer (abstain cases) are skipped: this reports answer accuracy and top-label ECE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from noma.model.heads import TYPE_INDEX
from noma.model.noma import Noma
from noma.probe.capacity import eval_items, score
from noma.train.data import batches

SETS = ("sealed", "jevbench_hard_heldout", "jevbench_hard", "jevbench_original", "jevbench_easy")


@torch.no_grad()
def probabilities(m: Noma, items) -> list:
    out = {}
    for b in batches(items, 16384, 32, seed=0, shuffle=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            opt, _, _ = m.forward_items([(it.prefix, it.block) for it in b], pad_multiple=128,
                                        evidence=False)
        for j, it in enumerate(b):
            K = len(it.block.keys)
            t = m.heads.temperature[TYPE_INDEX[it.block.qtype]]
            out[id(it)] = torch.softmax(opt[:, j, 1:1 + K].float() / t, -1).mean(0).cpu()
    return [out[id(it)] for it in items]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    exported = (Path(args.ckpt) / "model.safetensors").exists()   # a noma.export folder
    m = (Noma.from_pretrained(args.ckpt) if exported else Noma.load(args.ckpt).cuda().eval())
    cfg_file = Path(args.ckpt) / ("noma_config.json" if exported else "config.json")
    res = {"ckpt": args.ckpt, "config": json.loads(cfg_file.read_text())}
    for name in SETS:
        path = Path(args.eval) / f"{name}.jsonl"
        if not path.exists():
            continue
        v = eval_items(m.ser, path)
        its = [it for it, _ in v]
        res[name] = score(probabilities(m, its), its, [e for _, e in v])
        print(json.dumps({name: {k: res[name][k] for k in ("n", "acc", "ece")}}), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
