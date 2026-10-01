"""Pre-encode training items once, off the GPU (encoding on the GPU cost minutes of paid time per run).

    python -m noma.train.encode --out data/built/encoded.pkl --max-prefix 4096

The cache stores a tokenizer fingerprint and the serializer settings; the trainer refuses a
cache whose fingerprint or settings differ from its own, so a cache built with one tokenizer
can never be silently used with another.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import time
from pathlib import Path

from noma.model import facts
from noma.model.serialize import SPECIAL, Serializer

from .data import encode, load_jsonl

ROOT = Path(__file__).resolve().parents[2]
PROBE = ("Invoice INV-2231 for $620 was issued on 2026-03-02; the claim (§4.1) was filed "
         "3 days ago at 14:30. Überprüfung: naïve café, 東京, emoji 🙂, TiB vs GB.")


def fingerprint(tok) -> str:
    ids = tok(PROBE, add_special_tokens=False)["input_ids"]
    sp = [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    return hashlib.sha256(repr((len(tok), ids, sp)).encode()).hexdigest()[:16]


def settings(ser: Serializer) -> dict:
    return {"max_prefix": ser.max_prefix, "use_facts": ser.use_facts,
            "facts_version": facts.VERSION if ser.use_facts else "off"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--train", default=str(ROOT / "data/built/train.jsonl"))
    ap.add_argument("--calib", default=str(ROOT / "data/built/calib.jsonl"))
    ap.add_argument("--max-prefix", type=int, default=4096)
    ap.add_argument("--no-facts", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/built/encoded.pkl"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ser = Serializer(tok, args.max_prefix, not args.no_facts)
    t0 = time.time()
    out = {"fingerprint": fingerprint(tok), "settings": settings(ser),
           "train_sha": hashlib.sha256(Path(args.train).read_bytes()).hexdigest()[:16],
           "train": [encode(ser, r) for r in load_jsonl(args.train)],
           "calib": [encode(ser, r) for r in load_jsonl(args.calib)]}
    # A truncated prefix loses its middle, which can hold the clause that decides the label, so
    # truncated items are dropped rather than trained on (the serving path still truncates).
    for split in ("train", "calib"):
        n0 = len(out[split])
        out[split] = [i for i in out[split] if not i.prefix.truncated]
        out[f"{split}_dropped_truncated"] = n0 - len(out[split])
    Path(args.out).write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    tr = out["train"]
    print(f"encoded {len(tr)} train / {len(out['calib'])} calib in {time.time() - t0:.0f}s; "
          f"{sum(i.n_tokens for i in tr) / len(tr):.0f} tokens/item, "
          f"{out['train_dropped_truncated']} truncated items dropped; fingerprint {out['fingerprint']}; "
          f"{Path(args.out).stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
