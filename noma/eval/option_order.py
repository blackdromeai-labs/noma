"""Option-order robustness: does the predicted option change when the options are listed in a
different order?

    python -m noma.eval.option_order --ckpt <exported folder> --eval data/eval \
        --out-dir paper/analysis                      # GPU pass + report
    python -m noma.eval.option_order --analyze-only --out-dir paper/analysis

Choice questions only (score questions are ordered scales and yes/no questions have a fixed
order in the serializer). Each scored item is run three times: options as given, reversed, and
in one seeded random permutation (never the identity; never the reverse when there are more
than two options). Everything else is the normal pipeline, so the fact channel sees the
reordered options as well.

Writes option_order_items.jsonl (ids, option keys, probabilities; never item text; flushed
batch by batch and resumable), option_order.json and option_order.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from pathlib import Path

import numpy as np

SETS = ("sealed", "jevbench_original", "jevbench_easy")
ORDERS = ("original", "reversed", "random")
SEED = 20261006
B = 10_000
Z = 1.959963984540054


def permutation(order: str, item_id: str, k: int) -> list[int]:
    idx = list(range(k))
    if order == "original" or k < 2:
        return idx
    if order == "reversed":
        return idx[::-1]
    rng = random.Random(f"{SEED}:{item_id}")
    while True:
        p = idx[:]
        rng.shuffle(p)
        if p != idx and (k == 2 or p != idx[::-1]):
            return p


def variants(ser, path: Path) -> dict:
    """{order: [(Item, expected key)]} for the scored choice questions of one file."""
    from noma.train.data import encode
    out = {o: [] for o in ORDERS}
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        q = t["question"]
        if q["type"] != "choice" or t.get("expected") is None or not isinstance(q.get("criteria"), dict):
            continue
        keys = list(q["criteria"])
        if str(t["expected"]) not in keys:
            continue
        for o in ORDERS:
            crit = {keys[i]: q["criteria"][keys[i]] for i in permutation(o, t["id"], len(keys))}
            rec = {"id": t["id"], "family": t["family"], "state": t["state"],
                   "question": {"type": "choice", "instructions": q.get("instructions", ""),
                                "criteria": crit},
                   "target": None, "abstain": 0.0}
            out[o].append((encode(ser, rec), str(t["expected"])))
    return out


def run(args) -> None:
    from noma.eval.quick import details
    from noma.model.noma import Noma
    items_path = Path(args.out_dir) / "option_order_items.jsonl"
    done = {}
    if items_path.exists():
        for line in items_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[(r["set"], r["order"], r["id"])] = r
            except (ValueError, KeyError):
                continue
    items_path.write_text("".join(json.dumps(r) + "\n" for r in done.values()), encoding="utf-8")
    m = Noma.from_pretrained(args.ckpt)
    import torch
    with open(items_path, "a", encoding="utf-8") as fh:
        for name in SETS:
            path = Path(args.eval) / f"{name}.jsonl"
            if not path.exists():
                continue
            for order, pairs in variants(m.ser, path).items():
                exp = {it.id: e for it, e in pairs}
                cached = {i: {"p": torch.tensor(r["probs"]), "heads": None, "abstain": r["abstain"]}
                          for (s, o, i), r in done.items() if s == name and o == order}

                def sink(it, d, name=name, order=order, exp=exp):
                    k = int(d["p"].argmax())
                    fh.write(json.dumps({
                        "set": name, "order": order, "id": it.id, "family": it.family,
                        "keys": list(it.block.keys), "expected": exp[it.id],
                        "probs": [float(x) for x in d["p"]], "abstain": d["abstain"],
                        "predicted": it.block.keys[k],
                        "correct": it.block.keys[k] == exp[it.id]}) + "\n")
                    fh.flush()

                details(m, [it for it, _ in pairs], cached, sink, args.max_tokens, args.max_items)
                print(f"{name} {order}: {len(pairs)} items", flush=True)


# ----------------------------------------------------------------------------- analysis (numpy only)
def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** (n - 1))


def boot_mean(x: np.ndarray, tag: str) -> list[float]:
    if len(x) == 0:
        return [float("nan"), float("nan")]
    g = np.random.default_rng([SEED, zlib.crc32(tag.encode())])
    v = x[g.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def compare(base: dict, other: dict, tag: str) -> dict:
    """base, other: {id: row}. Statistics over the ids present in both."""
    ids = sorted(set(base) & set(other))
    n = len(ids)
    flip = np.array([base[i]["predicted"] != other[i]["predicted"] for i in ids])
    a = np.array([bool(base[i]["correct"]) for i in ids])
    b = np.array([bool(other[i]["correct"]) for i in ids])
    dp, dps = [], []
    for i in ids:
        top = base[i]["predicted"]
        p0 = base[i]["probs"][base[i]["keys"].index(top)]
        p1 = other[i]["probs"][other[i]["keys"].index(top)]
        dp.append(abs(p1 - p0))
        dps.append(p1 - p0)
    dp, dps = np.array(dp), np.array(dps)
    lost, gained = int((a & ~b).sum()), int((~a & b).sum())
    d = (gained - lost) / n
    se = math.sqrt(max(gained + lost - (gained - lost) ** 2 / n, 0)) / n
    first = lambda rows: sum(rows[i]["predicted"] == rows[i]["keys"][0] for i in ids)
    k2 = np.array([len(base[i]["keys"]) == 2 for i in ids])
    return {"n": n, "flips": int(flip.sum()), "flip_rate": float(flip.mean()),
            "flip_rate_wilson95": wilson(int(flip.sum()), n),
            "flips_two_options": [int(flip[k2].sum()), int(k2.sum())],
            "flips_more_options": [int(flip[~k2].sum()), int((~k2).sum())],
            "acc_original": float(a.mean()), "acc_original_wilson95": wilson(int(a.sum()), n),
            "acc_reordered": float(b.mean()), "acc_reordered_wilson95": wilson(int(b.sum()), n),
            "correct_original": int(a.sum()), "correct_reordered": int(b.sum()),
            "lost": lost, "gained": gained, "acc_diff": d,
            "acc_diff_ci95_wald": [d - Z * se, d + Z * se],
            "mcnemar_exact_p": mcnemar_exact(lost, gained),
            "mean_abs_dp_top": float(dp.mean()), "mean_abs_dp_top_boot95": boot_mean(dp, tag + "abs"),
            "median_abs_dp_top": float(np.median(dp)), "p95_abs_dp_top": float(np.percentile(dp, 95)),
            "max_abs_dp_top": float(dp.max()),
            "mean_signed_dp_top": float(dps.mean()), "mean_signed_dp_top_boot95": boot_mean(dps, tag + "sgn"),
            "predicted_first_listed_original": first(base), "predicted_first_listed_reordered": first(other)}


def analyze(out_dir: Path) -> None:
    rows = [json.loads(l) for l in (out_dir / "option_order_items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["set"], r["order"]), {})[r["id"]] = r
    pct = lambda x: f"{100 * x:.1f}%"
    ci = lambda c: f"[{100 * c[0]:.1f}%, {100 * c[1]:.1f}%]"
    J = {"seed": SEED, "bootstrap_resamples": B, "sets": {}, "pooled": {}, "either_order": {}}
    groups = [(s, [s]) for s in SETS if (s, "original") in by] + [("pooled", [s for s in SETS if (s, "original") in by])]
    t1, t2, t3 = [], [], []
    for g, members in groups:
        merged = {o: {f"{s}|{i}": r for s in members for i, r in by.get((s, o), {}).items()} for o in ORDERS}
        res = {}
        for o in ("reversed", "random"):
            c = compare(merged["original"], merged[o], f"{g}|{o}")
            res[o] = c
            t1.append([g, o, c["n"], f"{c['flips']}/{c['n']}", pct(c["flip_rate"]), ci(c["flip_rate_wilson95"]),
                       f"{c['flips_two_options'][0]}/{c['flips_two_options'][1]}",
                       f"{c['flips_more_options'][0]}/{c['flips_more_options'][1]}"])
            t2.append([g, o, f"{c['correct_original']}/{c['n']} = {pct(c['acc_original'])} {ci(c['acc_original_wilson95'])}",
                       f"{c['correct_reordered']}/{c['n']} = {pct(c['acc_reordered'])} {ci(c['acc_reordered_wilson95'])}",
                       c["lost"], c["gained"], f"{100 * c['acc_diff']:+.1f} [{100 * c['acc_diff_ci95_wald'][0]:+.1f}, {100 * c['acc_diff_ci95_wald'][1]:+.1f}]",
                       f"{c['mcnemar_exact_p']:.3f}"])
            t3.append([g, o, f"{c['mean_abs_dp_top']:.4f} [{c['mean_abs_dp_top_boot95'][0]:.4f}, {c['mean_abs_dp_top_boot95'][1]:.4f}]",
                       f"{c['median_abs_dp_top']:.4f}", f"{c['p95_abs_dp_top']:.3f}", f"{c['max_abs_dp_top']:.3f}",
                       f"{c['mean_signed_dp_top']:+.4f} [{c['mean_signed_dp_top_boot95'][0]:+.4f}, {c['mean_signed_dp_top_boot95'][1]:+.4f}]",
                       f"{c['predicted_first_listed_original']} -> {c['predicted_first_listed_reordered']}"])
        ids = sorted(set(merged["original"]) & set(merged["reversed"]) & set(merged["random"]))
        e = sum(merged["original"][i]["predicted"] != merged["reversed"][i]["predicted"]
                or merged["original"][i]["predicted"] != merged["random"][i]["predicted"] for i in ids)
        res["either"] = {"n": len(ids), "flips": e, "flip_rate": e / len(ids), "flip_rate_wilson95": wilson(e, len(ids))}
        (J["pooled"] if g == "pooled" else J["sets"]).update({g: res} if g != "pooled" else res)

    # numeric noise floor: the same original order, scored in the main dump with other batches
    noise = None
    dump = out_dir / "dump_release.jsonl"
    if dump.exists():
        main_rows = {}
        for l in dump.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                main_rows[f"{r['set']}|{r['id']}"] = r
        here = {f"{s}|{i}": r for s in SETS for i, r in by.get((s, "original"), {}).items()}
        common = {k: v for k, v in main_rows.items() if k in here}
        if common:
            noise = compare(common, here, "noise")
            J["noise_floor_same_order_other_batches"] = {k: noise[k] for k in ("n", "flips", "mean_abs_dp_top", "max_abs_dp_top")}

    def table(head, body):
        return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
                         + ["| " + " | ".join(str(c) for c in r) + " |" for r in body]) + "\n"

    P = J["pooled"]
    md = ["# Noma: option-order robustness\n",
          "Generated by `python -m noma.eval.option_order` (released weights, in-process). "
          "Scored choice questions only, from sealed, original and easy; score (ordered scale) and "
          "yes/no questions are excluded. Each item is run with its options as given, reversed, and "
          f"in one random permutation (seed {SEED}; never the identity, and never the reverse when "
          "there are more than two options; with exactly two options the random order is the "
          "reverse). A *flip* is a change of the predicted option key. Intervals are 95%: Wilson for "
          f"rates, percentile bootstrap ({B:,} resamples over items) for the probability change, "
          "exact McNemar for the accuracy difference.\n",
          "## Flip rate of the predicted option\n",
          table(["set", "order", "n", "flips", "flip rate", "Wilson 95%", "flips, 2 options", "flips, 3+ options"], t1)]
    if "either" in P:
        e = P["either"]
        md.append(f"Pooled, an item's prediction changes under at least one of the two reorderings in "
                  f"{e['flips']}/{e['n']} cases = {pct(e['flip_rate'])} {ci(e['flip_rate_wilson95'])}.\n")
    md += ["## Accuracy under each order\n",
           table(["set", "order", "original order", "reordered", "lost", "gained", "diff, pts [95%]", "McNemar p"], t2),
           "## Change in the probability of the originally predicted option\n",
           table(["set", "order", "mean abs change [95%]", "median", "p95", "max", "mean signed change [95%]",
                  "predictions on first-listed option (orig -> reordered)"], t3)]
    if noise:
        md.append(f"Noise floor: the same items in the same original order, as scored in the main dump "
                  f"with different batch composition and padding, differ by {noise['flips']}/{noise['n']} "
                  f"predictions and a mean absolute probability change of {noise['mean_abs_dp_top']:.5f} "
                  f"(max {noise['max_abs_dp_top']:.4f}). Reordering effects of that size are numerical, not behavioural.\n")
    md.append("## Reading\n")
    for o in ("reversed", "random"):
        if o in P:
            c = P[o]
            md.append(f"- Pooled, {o}: {c['flips']}/{c['n']} predictions flip ({pct(c['flip_rate'])}, 95% CI "
                      f"{ci(c['flip_rate_wilson95'])}); accuracy {pct(c['acc_original'])} -> {pct(c['acc_reordered'])} "
                      f"(McNemar p = {c['mcnemar_exact_p']:.3f}).")
    md.append("- The upper ends of the flip-rate intervals are the honest statement of what this test "
              "can rule out; the per-set intervals for original and easy (36 items each) are wide.")
    md.append("- Scope: only two reorderings per item, only choice questions with a few options, only "
              "sets on which the model is already accurate. Order sensitivity on the hard sets, on "
              "long option lists (the zero-shot sets) and for yes/no polarity was not tested.\n")
    (out_dir / "option_order.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
    (out_dir / "option_order.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_dir / 'option_order.md'} and option_order.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--out-dir", default="paper/analysis")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-items", type=int, default=32)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        if not args.ckpt:
            ap.error("--ckpt is required unless --analyze-only")
        run(args)
    analyze(Path(args.out_dir))


if __name__ == "__main__":
    main()
