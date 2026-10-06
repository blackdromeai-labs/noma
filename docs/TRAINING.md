# Training

This is the recipe that produced the released weights, and the failures that shaped it.

## Recipe

| | |
|---|---|
| Backbone | Qwen3.5-4B-Base, layers 1 to 18, frozen except LoRA |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05 |
| New parameters | embeddings for 9 special tokens; 4 listwise scorers; abstain and evidence outputs |
| Data | about 34,000 training decisions, 1,800 calibration decisions |
| Epochs | 1 |
| Learning rates | 1e-4 for LoRA and the new embeddings, 1e-3 for the heads, 3% warm-up |
| Batching | length-bucketed micro-batches of up to 16,384 padded tokens |
| Memory | gradient checkpointing; bfloat16 autocast |
| Hardware | one GPU with 40 GB; about 75 minutes on an A100 for the full set |

```bash
pip install "blackdrome-noma[train]"

# encode once, off the GPU
python -m noma.train.encode --tokenizer Qwen/Qwen3.5-4B-Base --train train.jsonl --calib calib.jsonl --out encoded.pkl --max-prefix 4096

python -m noma.train.train --backbone Qwen/Qwen3.5-4B-Base --cut 18 \
    --encoded encoded.pkl --epochs 1 --max-prefix 4096 --max-tokens 16384 --max-items 48 \
    --accum 1 --grad-ckpt --save-every 100 --out runs/noma

python -m noma.export --ckpt runs/noma --out release/noma --verify
```

A training record is one decision:

```json
{"id": "...", "family": "routing", "state": "...", "reference_time": "2026-03-20T09:00:00Z",
 "question": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}},
 "target": {"a": 0.9, "b": 0.1}, "abstain": 0.0, "evidence": ["a quote from the state"]}
```

`target` is a distribution, not a single label. `abstain` is 1 for items where no option is
supported. `evidence` is optional.

## Objective

```
loss = cross-entropy(soft target)
     + 0.5 * Brier
     + 0.5 * abstain binary cross-entropy
     + 0.1 * ordinal earth mover's distance   (score questions)
     + 0.1 * evidence binary cross-entropy
```

Cross-entropy alone trains a model to be right. Brier and the earth
mover's term are meant to help calibration; after temperature scaling we could not measure a
benefit from them. The soft targets carry real disagreement (between labellers, or between
human annotators in the source data) into the model instead of rounding it away. The earth
mover's term makes "one level off" cheaper than "three levels off" on ordered scales.

Each of the four heads sees the same batches with its own Poisson(1) weights per item, so
the heads end up as a bootstrap ensemble without training four models. The paper's ablations
found that the bootstrap makes no measurable difference and that one head is as accurate as
four.

After training, temperatures (one per question type, one for abstain) are fitted on the
calibration split.

## What went wrong, and the fixes

**The first overfit test failed (37% on 200 memorised items).** Four separate causes: LoRA
learning rate too high, micro-batches too small, one gradient clip shared between adapter
and heads, and the new embeddings training at the heads' rate. Fixes: the learning rates in
the table above, per-group gradient clipping, and embeddings on the adapter's rate.

**A full run went to NaN around step 700 and was lost.** One batch produced a non-finite
loss and the gradient poisoned every weight after it. The trainer now skips non-finite
batches, records which items were in them, stops if more than `--max-bad` occur, and writes
partial checkpoints every `--save-every` steps. The rerun skipped 5 batches and finished.

**Encoding on the GPU wasted paid time.** Tokenising and building fact lines for the whole
set took minutes of GPU billing per run. Encoding is now a separate step with a cache that
stores a tokenizer fingerprint and the serialiser settings; the trainer refuses a cache that
does not match.

**One 80,000-character item took over an hour in the fact channel.** A number pattern
backtracked catastrophically. The patterns are now bounded.

## Controlled experiments

All under the same recipe and data, changing one thing at a time. Details in
[EVALUATION.md](EVALUATION.md#controlled-experiments).

- **Depth.** Layer 18 of 32 matches the full depth for decisions (79.4% at both, three seeds).
- **Size.** A 9B backbone scored no better than the 4B one (one seed, earlier data build).
- **Data volume.** Not settled. One early run showed no gain from more data; the paper's runs
  give 79.4% at 6,000 items and 81.3% to 82.6% on the full set.
- **Targeted synthetic data.** Several thousand generated multi-step items did not improve
  held-out multi-step questions. More data of the same kind is not the lever for that tier;
  a single pass is.

## Ablation flags

The trainer and evaluator expose the switches used for the paper's ablations.

| Flag | Default | What it changes |
|---|---|---|
| `--head-kind listwise\|pointwise` | `listwise` | the released listwise scorer, or a small MLP that scores each option on its own |
| `--n-heads` | 4 | ensemble size |
| `--no-bootstrap` | off | every head sees every item with weight 1 |
| `--w-brier`, `--w-emd`, `--w-abstain`, `--w-evidence` | 0.5, 0.1, 0.5, 0.1 | loss weights; 0 removes a term |
| `--data-seed` | `--seed` | seed of the `--limit` subset only, so runs with different seeds can share one subset |

`python -m noma.eval.quick` takes `--sets` to choose evaluation sets and `--dump` to write
per-item probabilities (no item text). `python -m noma.eval.option_order` reruns choice
questions with the options reversed and shuffled.

The smallest configuration the paper tested, one pointwise head with cross-entropy, abstain
and evidence terms, scored within a point of the released design on the sealed set:

```bash
python -m noma.train.train --backbone Qwen/Qwen3.5-4B-Base --cut 18 --encoded encoded.pkl \
    --head-kind pointwise --n-heads 1 --no-bootstrap --w-brier 0 --w-emd 0 --out runs/minimal
```

## Reproducing

The training data is not part of this repository (see [DATA.md](DATA.md)). The trainer,
losses, encoder and export are complete, and any dataset in the record format above will
train. `python -m noma.train.train --overfit 200 --epochs 30 --out runs/overfit` is the
quickest check that a setup works: it should memorise 200 items.
