"""JevBench's own TypeSafe client talks to the local Noma server unchanged.

    python -m noma.serve.jev_client_check --endpoint http://127.0.0.1:8000 --n 30

Runs the unmodified ``jevbench.adapters.TypeSafeAdapter`` (the same code JevBench uses against
api.typesafe.ai) on public JevBench tasks of every question type, and passes only if every
response parses. Only the wire format is tested here.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "jevbench"))

from jevbench.adapters import TypeSafeAdapter  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--tasks", default=str(ROOT / "data/eval/jevbench_original.jsonl"))
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    tasks = load_jsonl(args.tasks)
    by_type = collections.defaultdict(list)
    for t in tasks:
        by_type[t.question["type"]].append(t)
    per = max(1, args.n // max(1, len(by_type)))
    picked = [t for ts in by_type.values() for t in ts[:per]]
    ad = TypeSafeAdapter(endpoint=args.endpoint, model="noma-1", key_env=None)
    ok = collections.Counter()
    fails = []
    lat = []
    for t in picked:
        r = ad.run(t)
        ok[(t.question["type"], r.ok)] += 1
        if r.ok:
            lat.append(r.latency_s)
            s = sum(r.probs.values())
            if abs(s - 1) > 1e-3:
                fails.append((t.id, f"probabilities sum {s}"))
        else:
            fails.append((t.id, r.error))
    print(dict(ok))
    for f in fails[:10]:
        print("FAIL", f)
    if lat:
        lat.sort()
        print(f"end-to-end latency p50 {lat[len(lat) // 2] * 1000:.0f} ms (local 4050, dry run)")
    print("PASS" if not fails else "FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
