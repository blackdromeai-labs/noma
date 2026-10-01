# Evaluation

Every number published for Noma, how it was measured, and what we learned from the
experiments behind it.

## Headline

| | Result |
|---|---|
| Latency, end to end (H100, JevBench client, 231 public tasks) | **16 ms median** |
| Latency, model only (H100) | 14 ms short states, 41 ms long states |
| JevBench easy | **48/48** (100%), ECE 0.011 |
| JevBench original | **71/72** (98.6%), ECE 0.084 |
| Sealed set, 12 families | **319/386** (82.6%), ECE 0.042 |
| JevBench, all public tasks | 176/231 (76.2%) |

## How latency was measured

The released weights were served with `noma serve` on one NVIDIA H100 with the fast path on
(length buckets of 128 tokens, CUDA graph per bucket). JevBench's own runner then sent the
public tasks over HTTP, one at a time, from the same machine. This is the method JevBench
uses for hosted models, minus the internet round trip.

| Tier | Correct | Median | p95 |
|---|---|---|---|
| Easy (48) | 48 | 15.6 ms | |
| Original (72) | 71 | 15.5 ms | |
| Hard (111) | 57 | 34.9 ms | 106 ms |
| All public (231) | 176 | about 16 ms | |

Hard-tier states are long, which is why that tier is slower. Model-only time on other
hardware: 21 ms (short) and 80 ms (long) on an A100 40GB.

## JevBench, full results

<p align="center"><img src="../media/latency.svg" alt="Latency comparison" width="80%"></p>

| Model | Open weights | Public tasks | Hard tier | Median latency | Cost per 1k |
|---|---|---|---|---|---|
| **Noma** | yes | 76.2% | 51.4% public, 46.4% held-out | **16 ms** | about $0.01 to $0.03, self-hosted |
| Jev 1.13.0 | no | 86.6% | 74.1% | 652 ms | $0.040 |
| decider-4b v2 | | | 67.3% | 17 ms | $0.020 |
| Cygnet | | | 75.5% | 35 ms | |
| Nimble 9B | yes | | 65.5% | 389 ms | |
| OpenJev (thinking) | yes | | 78.2% | | |

Figures for other models are as published on the JevBench leaderboard at the time of
release; blank cells are numbers we do not have. Noma's are our own measurements with the
JevBench runner and have not yet been submitted to the leaderboard.

Noma's cost is an estimate: at 16 ms per decision one GPU serves about 225,000 decisions an
hour in a single serial stream, and on-demand H100 prices currently run from about $2 to $6
an hour.

### Reading the hard tier

The hard tier is made of questions that need several chained steps: date arithmetic across
rules, long policies with amendments, multi-hop lookups, probability. Noma answers in one
forward pass with no scratchpad, and this is the tier where that design shows.

| Hard-tier family | Correct (public, 111) |
|---|---|
| Adversarial | 6/6 |
| Routing (hard) | 5/5 |
| Trap | 6/8 |
| Trade-off | 4/6 |
| Judge (hard) | 11/17 |
| Ambiguous | 4/7 |
| Multi-hop | 9/18 |
| Probability | 5/10 |
| Long policy | 5/19 |
| Temporal / numeric | 3/15 |

(Per-family counts are from the in-process evaluator, which scored 58/111; the HTTP run
scored 57/111. The one-item difference is numerical noise between the two paths.)

Two things to know about the hard-tier number:

1. **55 of the 111 public hard items were used as structure templates** for synthetic
   training data (the structure, never the text). The clean number is the one on the 56
   items that no generator or training run ever saw: **26/56 (46.4%)**. We report both and
   treat the held-out figure as the real one.
2. The families split cleanly. Where the question is a judgement (adversarial, routing,
   trap, trade-off), Noma does well on the hard tier too. Where it is a calculation chain
   (temporal and numeric, long policy), it does not, and that is the part to send to a
   reasoning model.

In practice: use `abstain` and `uncertainty`, plus what you know about the question, to
route multi-step questions to a model that can reason step by step, and keep Noma on the
decisions around it.

## Sealed set

386 scored decisions across 12 families, human reviewed, never used in training or by any
data generator. The set is private.

<p align="center"><img src="../media/families.svg" alt="Sealed set accuracy by family" width="80%"></p>

| Family | Correct | Family | Correct |
|---|---|---|---|
| Ambiguous / out of scope | 16/17 | Policy | 29/34 |
| Intent | 32/34 | Routing | 28/33 |
| Enum pick | 31/33 | Adequacy | 27/34 |
| Tone and safety | 30/32 | Relevance | 25/33 |
| Severity | 31/34 | Trade-off | 24/34 |
| Adversarial | 29/34 | Temporal / numeric | 17/34 |

## Calibration

Expected calibration error (top label, lower is better):

| Set | ECE |
|---|---|
| JevBench easy | 0.011 |
| Sealed set | 0.042 |
| JevBench original | 0.084 |
| JevBench hard (public) | 0.186 |
| JevBench hard (held-out) | 0.263 |

On the kinds of decisions Noma is built for, stated confidence tracks accuracy closely.
On multi-step questions it is overconfident, which is one more reason to route those by
question type and not by confidence alone.

## Controlled experiments

Each row changes one thing and holds the recipe, data and budget fixed. Differences of a
point or two on these set sizes are inside the noise.

**Depth and size** (same LoRA recipe, same 6,000-item subset):

| Backbone | Layers used | Hard tier (public) | Sealed set |
|---|---|---|---|
| 4B | 18 of 32 | 50.5% | 79.5% |
| 4B | 32 of 32 | 53.2% | 80.1% |
| 9B | 16 of 32 | 53.2% | 76.2% |

**Findings**

- **Mid-depth matches full depth for decisions.** Full depth is within noise of layer 18
  and costs nearly twice the compute.
- **4B matches 9B** under identical fine-tuning.
- **Accuracy saturates near 6,000 training items.** The 6,000-item runs above land within
  noise of the runs on the full set.
- **Targeted synthetic data does not transfer to held-out multi-step reasoning.** Adding
  about 3,500 generated multi-step items left the held-out hard items unchanged (26/56
  before and after) while the sealed set moved from 79.8% to 82.6%. Data, depth and size
  all fail to move that tier. We read this as a limit of single-pass decision models, and
  it is reproducible with the code here.

## Reproducing

```bash
# in-process accuracy and ECE on a folder of JevBench-format task files
python -m noma.eval.quick --ckpt /path/to/noma --eval /path/to/tasks --out results.json

# model-time latency, plain path against the fast path
python -m noma.serve.bench --ckpt /path/to/noma --tasks /path/to/tasks

# end to end: serve, then point the JevBench runner at http://127.0.0.1:8000
noma serve
```

The public JevBench tasks come from the JevBench repository. The sealed set is not released.
