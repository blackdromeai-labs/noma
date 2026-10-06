"""In-process evaluation of a checkpoint (release numbers come from JevBench's own
runner over HTTP against `noma serve`).

    python -m noma.eval.quick --ckpt /path/to/noma --eval /path/to/tasks --out quick.json

Same model path as serving (bucketed single pass, fitted temperatures). Items without an
expected answer (abstain cases) are skipped: this reports answer accuracy and top-label ECE.

Optional:
    --sets sealed,zeroshot/clinc150     which files under --eval to score (default: SETS)
    --max-tokens N --max-items N        batch budget (defaults 16384 / 32, as before)
    --dump items.jsonl                  one JSON line per item: ids, option keys, probabilities
                                        (ensemble mean and per head), abstain probability,
                                        prediction. Never any item text. Items without a usable
                                        expected answer are written too, with "scored": false
                                        (they are not part of the aggregates); they come from a
                                        separate forward pass so the scored numbers do not move.
                                        Rows are flushed batch by batch, and a run started again
                                        with the same --dump continues where the file stops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from noma.model.heads import TYPE_INDEX
from noma.model.noma import Noma
from noma.probe.capacity import eval_items, score
from noma.train.data import batches, encode

SETS = ("sealed", "jevbench_hard_heldout", "jevbench_hard", "jevbench_original", "jevbench_easy")


@torch.no_grad()
def details(m: Noma, items, cached: dict | None = None, sink=None,
            max_tokens: int = 16384, max_items: int = 32) -> list[dict]:
    """Per item: {"p": ensemble-mean probabilities [K], "heads": per-head probabilities [H, K],
    "abstain": mean abstain probability} (all after the fitted temperatures).

    cached: {item id: detail} from an earlier, interrupted run. A batch whose items are all
    cached is not run again; the batches themselves are always cut from the full item list, so
    a resumed run sees exactly the batches (and paddings) an uninterrupted one would.
    sink(item, detail) is called for every newly computed item, batch by batch."""
    out = {}
    for b in batches(items, max_tokens, max_items, seed=0, shuffle=False):
        if cached is not None and all(it.id in cached for it in b):
            for it in b:
                out[id(it)] = cached[it.id]
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            opt, ab, _ = m.forward_items([(it.prefix, it.block) for it in b], pad_multiple=128,
                                         evidence=False)
        for j, it in enumerate(b):
            K = len(it.block.keys)
            t = m.heads.temperature[TYPE_INDEX[it.block.qtype]]
            ph = torch.softmax(opt[:, j, 1:1 + K].float() / t, -1)
            a = torch.sigmoid(ab[:, j].float() / m.heads.abstain_temperature).mean()
            out[id(it)] = {"p": ph.mean(0).cpu(), "heads": ph.cpu(), "abstain": float(a)}
            if sink:
                sink(it, out[id(it)])
    return [out[id(it)] for it in items]


def probabilities(m: Noma, items) -> list:
    return [d["p"] for d in details(m, items)]


def unscored_items(ser, path: Path, scored_ids: set) -> list:
    """The rows eval_items leaves out (no expected answer, or one that is not an option key),
    encoded the same way. Only used for the dump."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if t["id"] in scored_ids:
            continue
        q = {k: v for k, v in t["question"].items() if k in ("type", "instructions", "criteria")}
        rec = {"id": t["id"], "family": t["family"], "state": t["state"], "question": q,
               "target": None, "abstain": 0.0}
        out.append((encode(ser, rec), None if t.get("expected") is None else str(t["expected"])))
    return out


def dump_row(name: str, it, exp, d: dict, scored: bool) -> dict:
    """Probabilities are written at full float32 precision so a resumed run reproduces the
    aggregates exactly."""
    k = int(d["p"].argmax())
    return {"set": name, "id": it.id, "family": it.family, "qtype": it.block.qtype,
            "keys": list(it.block.keys), "expected": exp, "scored": scored,
            "probs": [float(x) for x in d["p"]],
            "head_probs": [[float(x) for x in h] for h in d["heads"]],
            "abstain": d["abstain"],
            "predicted": it.block.keys[k],
            "correct": (it.block.keys[k] == exp) if scored else None}


def read_dump(path: Path) -> dict:
    """{(set, id): row} from a (possibly interrupted) dump; a torn last line is dropped. The
    file is rewritten with the complete rows so appending to it is safe."""
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                rows[(r["set"], r["id"])] = r
            except (ValueError, KeyError):
                continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
    return rows


def row_detail(r: dict) -> dict:
    return {"p": torch.tensor(r["probs"], dtype=torch.float32),
            "heads": torch.tensor(r["head_probs"], dtype=torch.float32), "abstain": r["abstain"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=None,
                    help="comma-separated file stems under --eval (subfolders allowed, e.g. "
                         "zeroshot/clinc150); default: " + ",".join(SETS))
    ap.add_argument("--dump", default=None, help="write per-item results (no item text) here")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="padded tokens per batch; lower it on a small GPU (results can move "
                         "in the last decimals, because the batch padding changes)")
    ap.add_argument("--max-items", type=int, default=32, help="items per batch")
    args = ap.parse_args()
    sets = tuple(s.strip() for s in args.sets.split(",") if s.strip()) if args.sets else SETS
    exported = (Path(args.ckpt) / "model.safetensors").exists()   # a noma.export folder
    m = (Noma.from_pretrained(args.ckpt) if exported else Noma.load(args.ckpt).cuda().eval())
    cfg_file = Path(args.ckpt) / ("noma_config.json" if exported else "config.json")
    res = {"ckpt": args.ckpt, "config": json.loads(cfg_file.read_text())}
    done = read_dump(Path(args.dump)) if args.dump else {}     # resume: rows already on disk
    dump = open(args.dump, "a", encoding="utf-8") if args.dump else None

    def run(name, pairs, scored):
        """Forward the (item, expected) pairs; with --dump, each new item is written and flushed
        as soon as its batch is done, and items already in the dump are not run again."""
        if not dump:
            return details(m, [it for it, _ in pairs], max_tokens=args.max_tokens,
                           max_items=args.max_items)
        exp = {it.id: e for it, e in pairs}
        cached = {i: row_detail(r) for (s, i), r in done.items() if s == name and i in exp}

        def sink(it, d):
            dump.write(json.dumps(dump_row(name, it, exp[it.id], d, scored)) + "\n")
            dump.flush()

        return details(m, [it for it, _ in pairs], cached, sink, args.max_tokens, args.max_items)

    for name in sets:
        path = Path(args.eval) / f"{name}.jsonl"
        if not path.exists():
            continue
        v = eval_items(m.ser, path)
        its = [it for it, _ in v]
        det = run(name, v, True)
        res[name] = score([d["p"] for d in det], its, [e for _, e in v])
        print(json.dumps({name: {k: res[name][k] for k in ("n", "acc", "ece")}}), flush=True)
        if dump:
            rest = unscored_items(m.ser, path, {it.id for it in its})
            if rest:
                run(name, rest, False)
    if dump:
        dump.close()
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
