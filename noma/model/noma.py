"""Noma model: a cut decoder backbone with LoRA, trained embeddings for the new special tokens,
and the listwise ensemble heads.

Training runs each question as one sequence [prefix + block]. Serving runs the prefix once and
forks its cache across all question blocks (``decide``); the two are the same computation, which
tests/test_model.py and ``python -m noma.model.check`` verify.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .heads import TYPE_INDEX, Heads, gather_slots
from .serialize import SPECIAL, Block, Prefix, Serializer

# Rows are single causal sequences padded on the right, so no mask is needed: SDPA applies its
# own causal mask and the linear-attention layers take none. Passing the per-type mapping skips
# transformers' mask builder, whose checks read GPU values on the host (a sync that breaks CUDA
# graph capture). Not valid with a cache (fork path), which keeps the default.
NO_MASK = {"full_attention": None, "linear_attention": None}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj"]


@dataclass
class NomaConfig:
    backbone: str = "Qwen/Qwen3.5-0.8B-Base"
    cut: int = 14
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    n_heads: int = 4
    d_head: int = 512
    max_prefix_tokens: int = 3584
    use_facts: bool = True
    facts_version: str = ""
    head_kind: str = "listwise"   # "pointwise" = the per-option MLP baseline (heads.SCORERS)

    def to_dict(self) -> dict:
        """Keys added after the first release are written only when they differ from their
        default, so configs of default models stay loadable by the released package."""
        d = asdict(self)
        if d["head_kind"] == "listwise":
            del d["head_kind"]
        return d


def expand_cache(cache, n: int) -> None:
    """Repeat a batch-1 prefix cache n times along the batch dimension, in place. Attention
    layers have a built-in for this; linear-attention layers keep conv/recurrent state tensors
    in per-index containers, which are expanded here."""
    def rep(container):
        items = container.items() if isinstance(container, dict) else enumerate(container)
        for k, v in list(items):
            if isinstance(v, torch.Tensor):
                container[k] = v.repeat_interleave(n, dim=0)
    for layer in cache.layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(n)
        for name in ("conv_states", "recurrent_states"):
            if isinstance(getattr(layer, name, None), (dict, list)):
                rep(getattr(layer, name))
        for name in ("keys", "values"):   # combined linear+full layers
            v = getattr(layer, name, None)
            if isinstance(v, torch.Tensor) and not hasattr(layer, "batch_repeat_interleave"):
                setattr(layer, name, v.repeat_interleave(n, dim=0))


class NewTokenEmbedding(nn.Module):
    """The backbone's embedding stays frozen; the new special tokens get their own small table,
    so the (250k x d) matrix never needs gradients."""

    def __init__(self, base: nn.Embedding, new_ids: list[int], init: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("ids", torch.tensor(new_ids), persistent=False)
        self.table = nn.Parameter(init.float().clone())

    def forward(self, input_ids):
        emb = self.base(input_ids)
        match = input_ids.unsqueeze(-1) == self.ids          # [..., n_new]
        hit = match.any(-1)
        # Branch-free: `if hit.any()` read a GPU value on the host, which syncs and made the
        # forward impossible to capture in a CUDA graph.
        idx = match.float().argmax(-1)
        return torch.where(hit.unsqueeze(-1), self.table[idx].to(emb.dtype), emb)


class Noma(nn.Module):
    def __init__(self, cfg: NomaConfig, dtype=torch.bfloat16, load_weights: bool = True):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from . import facts

        cfg.facts_version = facts.VERSION if cfg.use_facts else "off"
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.backbone)
        self.ser = Serializer(self.tok, cfg.max_prefix_tokens, cfg.use_facts)
        lm = AutoModelForCausalLM.from_pretrained(cfg.backbone, dtype=dtype)
        body = lm.model
        body.layers = body.layers[:cfg.cut]
        body.norm = nn.Identity()          # read the residual stream at the cut, as the probe did
        body.config.num_hidden_layers = cfg.cut
        if getattr(body.config, "layer_types", None):
            body.config.layer_types = body.config.layer_types[:cfg.cut]
        del lm
        emb = body.embed_tokens
        new_ids = [self.ser.special[t] for t in SPECIAL]
        if max(new_ids) >= emb.num_embeddings:
            raise RuntimeError("special token ids exceed the embedding table; resize needed")
        # Initialize each new token from the mean embedding of a few ordinary words.
        seeds = ["state", "end", "facts", "end", "question", "option", ":", "none", "answer"]
        init = torch.stack([emb.weight[self.tok(" " + w, add_special_tokens=False)["input_ids"]]
                            .float().mean(0) for w in seeds])
        for p in body.parameters():
            p.requires_grad_(False)
        from peft import LoraConfig, get_peft_model
        body = get_peft_model(body, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                               lora_dropout=cfg.lora_dropout,
                                               target_modules=LORA_TARGETS, bias="none"))
        for n, p in body.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
        self.body = body
        self.embed = NewTokenEmbedding(emb, new_ids, init)
        d = body.base_model.model.config.hidden_size
        self.heads = Heads(d, n_heads=cfg.n_heads, kind=cfg.head_kind, d=cfg.d_head)
        self.bucket: int | None = None  # serving: pad lengths to multiples of this (see warmup)
        self._graphs: dict | None = None  # serving: CUDA graphs per bucket length (see warmup)

    # ------------------------------------------------------------------ backbone
    def hidden(self, input_ids, **kw):
        out = self.body(inputs_embeds=self.embed(input_ids), use_cache=kw.pop("use_cache", False),
                        **kw)
        return out.last_hidden_state, getattr(out, "past_key_values", None)

    # ------------------------------------------------------------------ training forward
    def forward_items(self, items: list[tuple[Prefix, Block]], pad_multiple: int | None = None,
                      evidence: bool = True):
        """One row per (prefix, block). Returns option logits [H, B, T], abstain logits [H, B],
        and evidence logits per item (state tokens; empty when evidence=False). pad_multiple
        rounds the padded length up so the kernels see a small set of shapes."""
        dev = self.heads.temperature.device
        seqs = [p.ids + b.ids for p, b in items]
        L = max(len(s) for s in seqs)
        if pad_multiple:
            L = -(-L // pad_multiple) * pad_multiple
        pad_id = self.tok.pad_token_id or 0
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=dev)
        # Right padding: causal layers never let a real token see a pad, so no mask is needed
        # (and the linear-attention layers would ignore one).
        if self._graphs is not None and len(items) == 1 and not self.training:
            h = self._graph_hidden(ids)
        else:
            h, _ = self.hidden(ids, attention_mask=NO_MASK)
        positions = [[len(p.ids) + b.end] + [len(p.ids) + m for m in b.marks] for p, b in items]
        orders = [b.levels for _, b in items]
        x, roles, ords, pad = gather_slots(h, list(range(len(items))), positions, orders)
        opt, ab = self.heads(x, roles, ords, pad)
        ev = [self.heads.evidence(h[i, p.state_span[0]:p.state_span[1]].float()).squeeze(-1)
              for i, (p, _) in enumerate(items)] if evidence else []
        return opt, ab, ev

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def warmup(self, max_len: int | None = None, graphs: bool = False) -> list[int]:
        """Compile the kernels for every bucket length once (serving start-up). With
        graphs=True, also capture one CUDA graph per bucket for single-question requests."""
        if not self.bucket:
            return []
        dev = self.heads.temperature.device
        top = max_len or (self.cfg.max_prefix_tokens + 512)
        lens = list(range(self.bucket, top + self.bucket, self.bucket))
        if graphs:
            self._graphs, self._pool = {}, torch.cuda.graph_pool_handle()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for L in lens:
                ids = torch.full((1, L), self.tok.pad_token_id or 0, dtype=torch.long, device=dev)
                if graphs:
                    self._graph_hidden(ids)
                else:
                    self.hidden(ids, attention_mask=NO_MASK)
        torch.cuda.synchronize()
        return lens

    def _graph_hidden(self, ids):
        """Backbone forward through a captured CUDA graph for this (1, L) shape. Serving was
        ~55 ms of kernel-launch overhead for a 100-token request; one replay replaces the
        thousands of launches. The returned tensor is the graph's static output buffer, valid
        until the next call."""
        L = ids.shape[1]
        g = self._graphs.get(L)
        if g is None:
            static_in = torch.zeros_like(ids)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.hidden(static_in, attention_mask=NO_MASK)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                static_out, _ = self.hidden(static_in, attention_mask=NO_MASK)
            g = self._graphs[L] = (graph, static_in, static_out)
        graph, static_in, static_out = g
        static_in.copy_(ids)
        graph.replay()
        return static_out

    @torch.no_grad()
    def decide(self, state, questions: dict, reference_time=None, fork="auto") -> dict:
        """questions: {key: {type, instructions, criteria}} -> {key: (probs dict, abstain,
        uncertainty)}. With fork=True the prefix runs once and its cache is shared."""
        qs = list(questions.values())
        prefix = self.ser.prefix(state, qs, reference_time)
        blocks = [self.ser.block(q) for q in qs]
        dev = self.heads.temperature.device
        if fork == "auto":
            # Bucketed flat path when a bucket is set: every question runs as [prefix + block],
            # right-padded to a multiple of `bucket`, so the Triton kernels only ever see a few
            # sequence lengths (they recompile for each new length: ~2 s per new length on an
            # H100). The fork is kept for multi-question requests with a long shared prefix.
            fork = self.bucket is None or (len(blocks) > 2 and len(prefix.ids) > 1024)
        if fork:
            p_ids = torch.tensor([prefix.ids], device=dev)
            _, cache = self.hidden(p_ids, use_cache=True)
            n = len(blocks)
            if n > 1:
                expand_cache(cache, n)
            Lb = max(len(b.ids) for b in blocks)
            ids = torch.full((n, Lb), self.tok.pad_token_id or 0, dtype=torch.long, device=dev)
            for i, b in enumerate(blocks):
                ids[i, :len(b.ids)] = torch.tensor(b.ids, device=dev)
            h, _ = self.hidden(ids, past_key_values=cache, use_cache=True)
            positions = [[b.end] + b.marks for b in blocks]
            x, roles, ords, pad = gather_slots(h, list(range(n)), positions,
                                               [b.levels for b in blocks])
            opt, ab = self.heads(x, roles, ords, pad)
        else:
            opt, ab, _ = self.forward_items([(prefix, b) for b in blocks],
                                            pad_multiple=self.bucket, evidence=False)
        return self._readout(list(questions), blocks, opt, ab), len(prefix.ids) + sum(
            len(b.ids) for b in blocks)

    def _readout(self, keys, blocks, opt, ab):
        out = {}
        for i, (k, b) in enumerate(zip(keys, blocks)):
            K = len(b.keys)
            logits = opt[:, i, 1:1 + K].float()                    # [H, K]
            t = self.heads.temperature[TYPE_INDEX[b.qtype]]
            p_heads = torch.softmax(logits / t, -1)
            p = p_heads.mean(0)
            ent = lambda q: -(q * q.clamp_min(1e-12).log()).sum(-1)
            mi = float(ent(p) - ent(p_heads).mean())               # ensemble disagreement
            a = float(torch.sigmoid(ab[:, i].float() / self.heads.abstain_temperature).mean())
            out[k] = ({key: float(v) for key, v in zip(b.keys, p)}, a, max(mi, 0.0))
        return out

    # ------------------------------------------------------------------ checkpoints
    def trainable_state(self) -> dict:
        from peft import get_peft_model_state_dict
        return {"lora": get_peft_model_state_dict(self.body),
                "new_tokens": self.embed.table.detach().cpu(),
                "heads": {k: v.cpu() for k, v in self.heads.state_dict().items()}}

    def save(self, out: Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state(), out / "noma.pt")
        (out / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, dtype=torch.bfloat16) -> "Noma":
        from peft import set_peft_model_state_dict
        path = Path(path)
        cfg = NomaConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        m = cls(cfg, dtype=dtype)
        st = torch.load(path / "noma.pt", map_location="cpu")
        set_peft_model_state_dict(m.body, st["lora"])
        m.embed.table.data.copy_(st["new_tokens"])
        m.heads.load_state_dict(st["heads"])
        return m


def _from_pretrained(path_or_repo: str, device: str | None = None, dtype=torch.bfloat16) -> "Noma":
    """One-download loading of an exported Noma (see noma.export)."""
    from noma.export import from_pretrained
    return from_pretrained(path_or_repo, device=device, dtype=dtype)


Noma.from_pretrained = staticmethod(_from_pretrained)
