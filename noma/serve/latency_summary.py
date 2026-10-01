"""Summarize JevBench result files: n, correct, end-to-end latency p50/p95 (ms).

    python -m noma.serve.latency_summary h100/*.results.jsonl
"""

import json
import sys


def main() -> None:
    for f in sorted(sys.argv[1:]):
        rows = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        lat = sorted(r["latency_s"] for r in rows if r.get("latency_s"))
        ok = sum(bool(r.get("correct")) for r in rows)
        print(f"{f}: n {len(rows)} correct {ok} p50 {lat[len(lat) // 2] * 1000:.1f} ms "
              f"p95 {lat[int(0.95 * len(lat))] * 1000:.1f} ms")


if __name__ == "__main__":
    main()
