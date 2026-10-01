# Usage

Noma answers typed questions about a state. This page covers the request, the response, and
how to act on the numbers.

## The request

```json
{
  "state": "any text, or a JSON object",
  "questions": {
    "<your key>": {"type": "choice | noul | score", "instructions": "...", "criteria": ...}
  },
  "reference_time": "2026-10-01T09:00:00Z"
}
```

- **`state`** is what Noma decides over. Plain text or a JSON object; objects are serialised
  for you. Up to 4,096 tokens.
- **`questions`** is a map from your own keys to questions. All questions in one request share
  one pass over the state, so asking five costs little more than asking one.
- **`reference_time`** (optional) is "now" for relative dates such as "3 days ago" or
  "by Friday". If you leave it out the server clock is used and the response says so.

The wrapped shape `{"model": "...", "input": {...}}` is accepted too.

### Question types

| Type | `criteria` | Answer field |
|---|---|---|
| `choice` | object: option key to description, 1 to 255 options | `choice`: the top option key |
| `noul` | optional object describing `true` and `false` | `noul`: probability of yes |
| `score` | list of level descriptions, lowest first, at least 2 | `score`: expected level (0-based) |

Write `instructions` as the question you would ask a careful colleague. Write `criteria`
descriptions so that each option could be recognised without seeing the others.

## The response

```json
{
  "model": "noma",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.93,
      "probabilities": {"billing": 0.93, "identity": 0.05, "shipping": 0.02},
      "noma": {"abstain": 0.02, "uncertainty": 0.004}
    }
  },
  "usage": {"input_tokens": 212, "output_tokens": 0},
  "noma": {"model_ms": 14.2}
}
```

- `probabilities` always sums to 1 over the options you gave.
- `confidence` is the top probability (for `noul`, the larger of yes and no).
- `noma.abstain` is the probability that none of your options is supported by the state.
- `noma.uncertainty` is the disagreement between Noma's four heads.
- `output_tokens` is always 0. Noma does not generate.

Everything outside the `noma` objects matches the Jev wire format, so an existing Jev client
works by changing the base URL.

## Acting on the numbers

The usual pattern is three bands:

```python
a = response["answers"]["safe_next"]
if a["noma"]["abstain"] > 0.5 or a["noma"]["uncertainty"] > 0.1:
    escalate()                      # the question does not fit the state, or the input is unfamiliar
elif a["confidence"] >= 0.9:
    act(a)                          # automatic
else:
    send_to_bigger_model_or_person()
```

Pick the thresholds on your own traffic: run a few hundred labelled examples, plot accuracy
against confidence, and choose the confidence above which the error rate is acceptable.
Because Noma is calibrated, that curve is close to the diagonal and the threshold transfers.

Things worth knowing:

- A high `abstain` with a confident top option means "if forced to choose, this one, but the
  state does not really say". Treat it as no answer.
- `uncertainty` is small in absolute terms. Values above about 0.1 are unusual.
- For `score` questions, read the whole distribution. A flat distribution over the levels is
  a real answer: the state does not pin the level down.
- Questions that need several chained calculation steps are outside what one pass does well.
  Route those to a reasoning model (see [EVALUATION.md](EVALUATION.md)).

## Python API

```python
from noma import Noma

model = Noma.from_pretrained("BlackdromeAILabs/noma")        # or a local folder
answers, n_tokens = model.decide(state, questions, reference_time="2026-10-01T09:00:00Z")
probs, abstain, uncertainty = answers["team"]
```

`from_pretrained(path_or_repo, device=None)` picks CUDA when it is available and falls back
to CPU. Pass `device="mps"` on Apple silicon. On CUDA, wrap calls in
`torch.autocast("cuda", dtype=torch.bfloat16)` as the server does.

For the latencies quoted in the README you need the fast path, which the server sets up for
you:

```python
model.bucket = 128
model.warmup(graphs=True)      # NVIDIA GPUs only; takes a minute at start-up
```

## Patterns

**Routing.** One `choice` question per routing decision, plus a `noul` for "does this need a
human". Use the keys your code already switches on.

**Agent step verification.** After each tool call, pass the tool output as the state and ask
`step_ok`, `safe_next`, `done`. This is the cheapest place to catch a failed step before the
agent builds on it.

**Model routing.** Ask which tier should serve a request and whether tools are needed, then
send only the hard ones to the expensive model.

**Moderation and scoring.** `score` questions with ordered levels give you a distribution
and an expected level; threshold on the probability mass above a level rather than on the
single top level.
