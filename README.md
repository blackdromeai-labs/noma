<p align="center">
  <img src="media/banner.svg" alt="Noma: decisions in milliseconds. An open decision model by Blackdrome AI Labs." width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MPL-2.0" src="https://img.shields.io/badge/license-MPL--2.0-7DD0C0?labelColor=08243B"></a>
  <a href="https://huggingface.co/BlackdromeAILabs/noma"><img alt="Weights on Hugging Face" src="https://img.shields.io/badge/weights-Hugging%20Face-7DD0C0?labelColor=08243B"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-7DD0C0?labelColor=08243B">
  <img alt="API: /v1/systemone" src="https://img.shields.io/badge/API-%2Fv1%2Fsystemone-7DD0C0?labelColor=08243B">
</p>

Noma is an open decision model. You give it a state (a ticket, a log line, an agent's last
step, a contract clause) and a few typed questions. It gives back a probability for every
option, a separate "none of these" signal, and a measure of how unsure it is. It never
writes text, so there is nothing to parse and nothing to wait for.

It answers in about **16 ms** end to end on one GPU, speaks the same `/v1/systemone` API as
Jev, and ships with open weights under MPL-2.0.

<p align="center">
  <img src="media/playground-presets.gif" alt="The Noma playground answering ticket triage, agent step checks, model routing and contract questions" width="100%">
</p>
<p align="center"><sub>The local playground, recorded against the released weights on one NVIDIA L40S. Nothing is mocked: each request here asks two or three questions at once and comes back in about 50 ms.</sub></p>

## Try it

```bash
pip install git+https://github.com/blackdromeai-labs/noma
noma serve
```

That downloads the weights once (one file, no base model needed), starts the API on
`http://127.0.0.1:8000/v1/systemone`, and opens a playground at `http://127.0.0.1:8000/`.

Ask it something:

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Hi, my card was charged twice for order #4471. I also cannot log in since yesterday.",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this first?",
               "criteria": {"billing": "Billing and refunds", "identity": "Login and account access",
                            "shipping": "Shipping and delivery"}},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Not urgent", "Low", "High", "Critical"]}
  }
}'
```

Or skip the server and call the model from Python:

```python
from noma import Noma

model = Noma.from_pretrained("BlackdromeAILabs/noma")
answers, _ = model.decide(
    state="Deploy 4/6 finished. Health checks: 3 of 12 pods failing readiness after rollout.",
    questions={
        "step_ok":   {"type": "noul", "instructions": "Did the rollout succeed?"},
        "next_step": {"type": "choice", "instructions": "What should the agent do next?",
                      "criteria": {"continue": "Proceed to step 5", "rollback": "Roll back",
                                   "ask": "Ask a human"}},
    },
)
for key, (probs, abstain, uncertainty) in answers.items():
    print(key, max(probs, key=probs.get), probs, abstain, uncertainty)
```

More in [docs/USAGE.md](docs/USAGE.md).

## Why a decision model

Agents and pipelines spend most of their calls on small questions: which queue, which model,
did that step work, is this safe to run, are we done. Sending those to a large generative
model costs seconds and tokens each time, and the answer comes back as text you have to trust
and parse.

Noma is built for that layer only. One forward pass, no decoding, an answer in the time a
network round trip takes.

<p align="center">
  <img src="media/agent-loop.svg" alt="Noma routes requests before an agent runs and verifies each step after" width="100%">
</p>

## Fast

<p align="center">
  <img src="media/latency.svg" alt="Median latency per decision: Noma 16 ms, decider-4b v2 17 ms, Cygnet 35 ms, Nimble 9B 389 ms, Jev 1.13.0 652 ms" width="82%">
</p>

Measured with JevBench's own client over HTTP, one question per request, on the 231 public
tasks: **16 ms median** on an H100 (14 ms of that is the model). Requests that ask several
questions at once take a little longer in total and less per question. Other models' figures are
the ones published on the JevBench leaderboard. Details and hardware in
[docs/EVALUATION.md](docs/EVALUATION.md).

At that speed one GPU handles roughly 225,000 decisions an hour in a single serial stream,
which works out to about **$0.01 to $0.03 per 1,000 decisions** at current on-demand GPU
prices. You run it yourself, so there is no per-call fee.

## Calibrated

When Noma says 90%, it is right close to 90% of the time. That is what makes the probabilities
usable: you can set a threshold, act automatically above it, and send the rest to a person or
a bigger model.

<p align="center">
  <img src="media/calibration.svg" alt="Expected calibration error: 0.011 on JevBench easy, 0.042 on the sealed set, 0.084 on JevBench original" width="82%">
</p>

Two more signals come with every answer:

- **`abstain`**: the probability that none of the options is supported by the state. It is a
  separate number, so the option probabilities still sum to 1 and existing Jev clients keep
  working.
- **`uncertainty`**: disagreement between four independently trained heads. It rises on
  inputs unlike anything Noma was trained on.

## Accurate where one pass is enough

<p align="center">
  <img src="media/accuracy.svg" alt="Accuracy: 100% on JevBench easy, 98.6% on JevBench original, 82.6% on a sealed human-reviewed set" width="82%">
</p>

<p align="center">
  <img src="media/families.svg" alt="Sealed set accuracy by decision family" width="82%">
</p>

The sealed set is 386 human-reviewed decisions across 12 families that no training run or
data generator ever saw. Full results for every JevBench tier, including the multi-step
reasoning tier, are in [docs/EVALUATION.md](docs/EVALUATION.md).

## The playground

`noma serve` includes a local playground: paste a state, build questions, watch the
probabilities, and copy the request as curl or Python.

<p align="center">
  <img src="media/playground-light.png" alt="Playground, light theme: a contract clause with two questions answered" width="100%">
</p>

<table>
<tr>
<td width="50%"><img src="media/playground-custom.gif" alt="Typing a new state and question into the playground"></td>
<td width="50%"><img src="media/playground-code.gif" alt="Copying the request as code and switching theme"></td>
</tr>
<tr>
<td align="center"><sub>Write your own state and questions</sub></td>
<td align="center"><sub>Copy the request as code; light and dark themes</sub></td>
</tr>
<tr>
<td><img src="media/playground-agent.png" alt="Agent step check example"></td>
<td><img src="media/playground-routing.png" alt="Model routing example"></td>
</tr>
<tr>
<td align="center"><sub>Agent step check</sub></td>
<td align="center"><sub>Model routing</sub></td>
</tr>
</table>

## How it works

<p align="center">
  <img src="media/architecture.svg" alt="Request, fact channel, cut backbone, decision heads, answer" width="100%">
</p>

Noma keeps the first 18 of 32 layers of Qwen3.5-4B and replaces the language-model head with
small decision heads. What is new in it:

1. **A decision head instead of token scoring.** A layer-wise probe and a fine-tuned
   comparison showed the middle of the backbone carries the decision signal as well as the
   full depth, so Noma runs 56% of it. A listwise scorer reads all options together. Abstain
   is its own calibrated output. Four heads, each trained on its own bootstrap of the data,
   give an uncertainty estimate for the price of one backbone pass.
2. **Fast serving for a hybrid linear-attention backbone.** The state is processed once and
   its cache, including the recurrent and convolution state of the linear-attention layers,
   is forked across all questions. Requests are padded to length buckets and each bucket runs
   as a captured CUDA graph. This took serving from about 2.3 s to 16 ms.
3. **A fact channel.** Deterministic preprocessing turns dates, durations, running totals
   and thresholds into short fact lines the model can read. They are hints, never overrides.
4. **Blind, agreement-gated labelling.** Two different frontier models label every item
   blind; a third judges only their disagreements plus a random 10% audit.

The full list, with the supporting methods and the controlled experiments behind each
choice, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/TRAINING.md](docs/TRAINING.md).

## Scope

Noma makes single-pass decisions: classify, route, score, check, verify. It is the fast
layer in a system, and it is built to know when to hand off.

- Questions that need several chained steps of arithmetic or date reasoning, or tracing a
  long policy through its amendments, belong with a reasoning model. Use `abstain` and
  `uncertainty` to route them there.
- Text only. States up to 4,096 tokens.
- Trained and evaluated in English.

## Documentation

| | |
|---|---|
| [Usage](docs/USAGE.md) | Question types, the response, thresholds, Python API |
| [Serving](docs/SERVING.md) | `noma serve`, hardware, the fast path, Jev clients, recording the playground |
| [Architecture](docs/ARCHITECTURE.md) | The model and what it introduces |
| [Training](docs/TRAINING.md) | Recipe, losses, what failed and what fixed it |
| [Data](docs/DATA.md) | How the training and evaluation data were built |
| [Evaluation](docs/EVALUATION.md) | Every number, how it was measured, and the findings |
| [AGENTS.md](AGENTS.md) | Notes for coding agents working in this repository |

## Citation

```bibtex
@software{noma2026,
  title  = {Noma: an open, calibrated decision model},
  author = {{Blackdrome AI Labs}},
  year   = {2026},
  url    = {https://github.com/blackdromeai-labs/noma}
}
```

## License

Code and weights are released under [MPL-2.0](LICENSE), © Blackdrome AI Labs. Noma is built
on Qwen3.5-4B-Base (Apache-2.0); see [NOTICE](NOTICE).

Questions, results, or something Noma got wrong: **hello@blackdrome.tech**

<p align="center"><sub>Blackdrome AI Labs · <a href="https://blackdromeai.vercel.app/">blackdromeai.vercel.app</a></sub></p>
