# Serving

## Start the server

```bash
pip install git+https://github.com/blackdromeai-labs/noma
noma serve                                  # weights from Hugging Face
noma serve --model /path/to/noma            # a local copy of the weights
noma serve --host 0.0.0.0 --port 9000       # reachable from other machines
noma serve --device cpu --no-fast-path      # plain path, any device
```

| Route | What it does |
|---|---|
| `POST /v1/systemone` | Decisions. Same wire format as Jev. |
| `GET /` | The playground. |
| `GET /v1/info` | Model name, readiness, device, whether the fast path is on. |
| `GET /health` | `{"ok": true}` once the model is loaded. |

The model loads in the background at start-up, so the playground opens at once and shows
"loading" until `/v1/info` reports `ready`.

Settings can also be given as environment variables: `NOMA_MODEL`, `NOMA_DEVICE`,
`NOMA_BUCKET` (0 turns bucketing off), `NOMA_GRAPHS` (0 turns graph capture off),
`NOMA_MODEL_NAME` (the name reported in responses).

The server has no authentication. Run it on localhost or behind your own gateway.

## Hardware

| Device | Works | Notes |
|---|---|---|
| NVIDIA GPU, 8 GB or more, Linux | yes | Fast path. All published latencies use this setup. |
| NVIDIA GPU, 6 GB | yes | The model takes 5.2 GB. Tested on a laptop RTX 4050 on Windows: about 0.4 s per request with the reference kernels. Use `--no-fast-path`. |
| Apple silicon (MPS) | yes | Plain path. |
| CPU | yes | Plain path. Loads in about 6 GB of RAM. Tens of seconds per request: enough to check an answer, not to serve. |

The weights are one 5.4 GB `model.safetensors` file in bfloat16. Nothing else is downloaded: the
backbone is already inside it with the adapter merged.

For full speed on Linux with an NVIDIA GPU, install the optimised kernels for the
linear-attention layers:

```bash
pip install "noma[fast] @ git+https://github.com/blackdromeai-labs/noma"
```

Without them the backbone uses a reference implementation that gives the same answers more
slowly. That is what runs on Windows and macOS.

## The fast path

Three things make the 16 ms figure:

1. **Prefix fork.** The state is run once. Its cache is copied across every question, so
   questions are scored in parallel without re-reading the state. For the hybrid backbone
   this includes the recurrent and convolution state of the linear-attention layers.
2. **Length buckets.** Inputs are padded to a multiple of 128 tokens, so the server sees a
   small set of shapes.
3. **CUDA graphs.** Each bucket is captured as a CUDA graph at start-up and replayed. This
   only became possible after removing every host-to-device sync from the forward pass.

The bucketed path gives the same probabilities as the plain path to within numerical noise.
`python -m noma.serve.bench` measures both and reports the largest difference.

| Hardware | Short states | Long states |
|---|---|---|
| H100, model time | 14 ms | 41 ms |
| A100 40GB, model time | 21 ms | 80 ms |
| H100, end to end over HTTP (JevBench client) | 16 ms median across all public tasks | |

Two things to expect in practice:

- **Several questions in one request** run as one batch rather than through a captured
  graph. On an L40S a request with two or three questions takes about 50 ms in total.
- **The first request of a new shape is slower** (around 180 ms on an L40S) while its kernel
  is prepared. Start-up warm-up covers the single-question shapes; send a few representative
  requests after start-up if you need every shape warm.

## Using a Jev client

Any client written for Jev's `/v1/systemone` works unchanged. Point it at your server:

```python
import requests

r = requests.post("http://127.0.0.1:8000/v1/systemone", json={
    "model": "noma",
    "state": "...",
    "questions": {"ok": {"type": "noul", "instructions": "Is this request safe to run?"}},
})
print(r.json()["answers"]["ok"])
```

`python -m noma.serve.jev_client_check --endpoint http://127.0.0.1:8000` runs JevBench's
unmodified adapter against your server and checks that every response parses. It needs a
checkout of JevBench on the Python path.

Noma's custom heads are not a language-model head, so the weights cannot be converted to
GGUF or run through runtimes that expect a generative model. Use `noma serve`.

## Recording the playground

The GIFs and screenshots in the README were produced by `scripts/record_playground.py`,
which drives a real browser against a running server:

```bash
pip install playwright && playwright install chromium
noma serve --port 8000 &
python scripts/record_playground.py --url http://127.0.0.1:8000 --out media
ffmpeg -i media/playground-presets.webm -vf "fps=12,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" media/playground-presets.gif
```

Nothing in the recordings is mocked: the answers and latencies are whatever the server
returned.
