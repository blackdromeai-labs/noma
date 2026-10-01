# Notes for coding agents

Guidance for AI coding agents (and people) changing this repository.

## What this is

Noma is a decision model: a cut Qwen3.5-4B backbone with listwise decision heads, served
over `/v1/systemone`. It does not generate text. Read `docs/ARCHITECTURE.md` first.

## Layout

```
noma/model/     the model, heads, serialiser, fact channel
noma/serve/     FastAPI server, playground (static/), latency benchmark
noma/train/     trainer, losses, encoding cache
noma/eval/      in-process evaluator
noma/export.py  single-file export and from_pretrained
scripts/        figure and recording scripts for the README
tests/          unit tests; none need weights or a GPU
```

## Commands

```bash
pip install -e ".[dev]"
python -m pytest -q tests            # must pass before any change is proposed
noma serve --model /path/to/noma     # server and playground
python scripts/make_figures.py       # regenerate README figures
```

## Rules

- **Never train on evaluation data.** JevBench tasks and the sealed set are for scoring
  only. New training data must pass 13-gram decontamination against every evaluation item.
- **Do not report a number you did not measure.** Every figure in the README and
  `docs/EVALUATION.md` traces to a run. If you change the model or the fact channel, the
  numbers must be re-measured, not carried over.
- **The fact channel is versioned.** Any change to `noma/model/facts.py` that alters its
  output changes the model's inputs: bump the version and retrain or re-evaluate.
- **Keep the forward pass free of host-device syncs.** No `.item()`, no Python branching on
  tensor values inside `Noma.hidden` or the heads. A sync breaks CUDA-graph capture and
  costs milliseconds per call. `python -m noma.serve.bench` shows the effect.
- **The wire format is a contract.** `probabilities` sums to 1 over the caller's options;
  Noma-specific fields live under `noma`. Jev clients must keep working unchanged.
- **Patterns in the fact channel must be bounded.** An unbounded number regex once took an
  hour on one long input.
- Match the surrounding code: short modules, comments that say why, no new dependencies
  without a reason.

## Tests

`tests/test_serve.py` exercises the HTTP layer with a stand-in model. `tests/test_model.py`
covers serialisation, the fact channel, the heads and the losses. Add a test with any change
to request validation, serialisation or the fact channel.
