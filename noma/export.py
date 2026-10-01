"""Export a trained checkpoint as one self-contained model folder, and load it back.

    python -m noma.export --ckpt runs/noma --out release/noma

The folder holds everything inference needs, so users download one repo and need no base
model and no PEFT:
    model.safetensors   cut backbone with LoRA merged in, new-token embeddings, heads
    noma_config.json    Noma settings (cut depth, prefix length, facts version, ...)
    backbone_config.json  the cut backbone's transformers config
    tokenizer files     with Noma's special tokens already added

    from noma import Noma
    m = Noma.from_pretrained("BlackdromeAILabs/noma")
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch


def export(ckpt: str, out: str) -> None:
    from safetensors.torch import save_file
    from .model.noma import Noma

    m = Noma.load(ckpt, dtype=torch.bfloat16)
    body = m.body.merge_and_unload()          # LoRA folded into the base weights
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    tensors = {f"body.{k}": v.contiguous() for k, v in body.state_dict().items()}
    tensors["embed.table"] = m.embed.table.detach().to(torch.float32).contiguous()
    tensors.update({f"heads.{k}": v.contiguous() for k, v in m.heads.state_dict().items()})
    save_file(tensors, str(out_p / "model.safetensors"), metadata={"format": "pt"})
    (out_p / "noma_config.json").write_text(json.dumps(asdict(m.cfg), indent=1), encoding="utf-8")
    (out_p / "backbone_config.json").write_text(body.config.to_json_string(), encoding="utf-8")
    m.tok.save_pretrained(str(out_p))
    size = (out_p / "model.safetensors").stat().st_size / 2**30
    print(f"exported {out_p} ({size:.2f} GB)")


def from_pretrained(path_or_repo: str, device: str | None = None, dtype=torch.bfloat16):
    """Build a ready-to-use Noma from an exported folder or a Hugging Face repo id."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from .model.heads import Heads
    from .model.noma import Noma, NomaConfig, NewTokenEmbedding
    from .model.serialize import SPECIAL, Serializer

    p = Path(path_or_repo)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(path_or_repo))
    cfg = NomaConfig(**json.loads((p / "noma_config.json").read_text(encoding="utf-8")))
    bcfg = Qwen3_5TextConfig(**json.loads((p / "backbone_config.json").read_text(encoding="utf-8")))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Build the backbone without allocating weights, then hand it the saved tensors directly:
    # peak memory stays at one copy of the model, on the target device.
    with torch.device("meta"):
        body = Qwen3_5TextModel(bcfg)
    body.norm = torch.nn.Identity()
    state = load_file(str(p / "model.safetensors"), device=dev)
    weights = {k[5:]: v.to(dtype) for k, v in state.items() if k.startswith("body.")}
    missing, unexpected = body.load_state_dict(weights, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"model.safetensors is missing backbone weights: {missing[:5]}")
    del weights
    body.rotary_emb = type(body.rotary_emb)(bcfg).to(dev)   # its buffers are computed, not saved
    body.eval()

    m = Noma.__new__(Noma)
    torch.nn.Module.__init__(m)
    m.cfg = cfg
    m.tok = AutoTokenizer.from_pretrained(str(p))
    m.ser = Serializer(m.tok, cfg.max_prefix_tokens, cfg.use_facts)
    m.body = body
    new_ids = [m.ser.special[t] for t in SPECIAL]
    table = state["embed.table"]
    m.embed = NewTokenEmbedding(body.embed_tokens, new_ids, table)
    m.heads = Heads(bcfg.hidden_size, n_heads=cfg.n_heads, d=cfg.d_head)
    m.heads.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("heads.")})
    m.bucket, m._graphs = None, None
    return m.to(dev).eval()


def verify(ckpt: str, out: str) -> float:
    """Largest probability difference between the training checkpoint and its export."""
    from .model.check import QUESTIONS, STATE
    from .model.noma import Noma

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref, res = "2026-03-20T09:00:00Z", []
    for m in (Noma.load(ckpt, dtype=torch.bfloat16).to(dev).eval(), from_pretrained(out, device=dev)):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            res.append(m.decide(STATE, QUESTIONS, ref)[0])
        del m
    a, b = res
    diff = max(abs(a[k][0][o] - b[k][0][o]) for k in a for o in a[k][0])
    print(f"max probability difference, checkpoint vs export: {diff:.5f}")
    print({k: max(v[0], key=v[0].get) for k, v in b.items()})
    return diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", help="compare the export with the checkpoint")
    args = ap.parse_args()
    export(args.ckpt, args.out)
    if args.verify and verify(args.ckpt, args.out) > 0.02:
        raise SystemExit("export does not match the checkpoint")


if __name__ == "__main__":
    main()
