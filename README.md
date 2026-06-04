# platform-ai-service

OpenAI-compatible local LLM inference, powered by [vLLM](https://docs.vllm.ai).
Hosts a single configurable model and exposes it over HTTP so other applications
can consume it with the standard OpenAI SDK — just change `base_url`.

## Why vLLM (not nano-vllm)

nano-vllm is an educational ~1,200-line reimplementation: single maintainer, no
PyPI releases, Qwen3-only, no HTTP server, no streaming, no async API, and
Linux/CUDA-only. vLLM is production-grade — continuous batching + PagedAttention,
a natively-async engine, a built-in OpenAI-compatible API, broad model support,
quantization, tensor parallelism, LoRA, and Prometheus metrics — which is why it
is the platform here.

## Architecture

Our own FastAPI app embeds vLLM's `AsyncLLMEngine` and **delegates the OpenAI
endpoints to vLLM's own serving handlers**, so responses are byte-for-byte
OpenAI-compatible while we keep ownership of auth, CORS, health, logging, and
config. Single process, no proxy hop.

```
client ─▶ server/Init.py        FastAPI app, CORS, lifespan (loads model once)
        ─▶ server/ChatEndpoint   /v1/chat/completions, /v1/completions
        ─▶ engine/ServingClient  vLLM OpenAIServing{Chat,Completion,Models}  ← only version-sensitive file
        ─▶ engine/EngineClient   AsyncLLMEngine (built from ai/llm/VLLM.py config)
```

All `vllm.entrypoints.openai.*` imports are confined to `engine/ServingClient.py`
(plus the request schemas in `ChatEndpoint.py`). A vLLM upgrade only needs that
file reconciled — see the version note at the top of it.

## Requirements

- **Linux** host with an **NVIDIA GPU** (CUDA). vLLM does not run on native
  Windows — use the Linux host, WSL2, or a container. The Windows checkout is for
  editing only.
- Python 3.10–3.12 (transformers 5 drops 3.9).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit: set MODEL, API_KEYS, and HF_TOKEN if gated
```

## Run

```bash
python main.py
```

The model loads during startup (FastAPI lifespan); `GET /health/ready` returns
200 once it is serving. Run **one process per GPU group** — scale out with more
instances behind a load balancer, never with extra uvicorn workers.

## Running Gemma 4 31B (unsloth bnb-4bit)

`unsloth/gemma-4-31B-unsloth-bnb-4bit` is a **multimodal** (`gemma4` arch),
**bitsandbytes 4-bit** checkpoint with a **256K** max context. To serve it, set
in `.env` (an example block is included there):

```env
MODEL=unsloth/gemma-4-31B-unsloth-bnb-4bit
DTYPE=bfloat16
QUANTIZATION=bitsandbytes
LOAD_FORMAT=bitsandbytes      # required for pre-quantized bnb weights
ENFORCE_EAGER=true            # recommended for bnb
MAX_MODEL_LEN=8192            # model max is 262144 — cap it or the KV cache OOMs
TENSOR_PARALLEL_SIZE=1        # bnb is happiest on a single GPU
HF_TOKEN=hf_...               # Gemma is gated
```

Requirements specific to this model (already in `requirements.txt`):
`vllm>=0.19` (gemma4 support; pinned to `0.22.0`), `transformers>=5.5.0`, and the
`bitsandbytes` package. Plan for **~40GB+ VRAM** — the vision/audio towers,
embeddings, and `lm_head` stay in bf16 (only the transformer weights are 4-bit),
so it won't fit on a 24GB card with useful context.

Text chat works through `/v1/chat/completions` as usual. Note bitsandbytes is
convenient for testing but slower than AWQ/GPTQ/FP8; for a production deployment
prefer an AWQ/FP8 build or full bf16 if VRAM allows.

## Endpoints

| Method | Path                    | Auth | Notes                                  |
|--------|-------------------------|------|----------------------------------------|
| POST   | `/v1/chat/completions`  | yes  | OpenAI-compatible, `stream` supported  |
| POST   | `/v1/completions`       | yes  | OpenAI-compatible, `stream` supported  |
| GET    | `/v1/models`            | yes  | OpenAI-compatible model listing        |
| POST   | `/api/v1/infer-schema`  | yes  | LLM column type/date inference (`Result` shape) |
| GET    | `/info`                 | yes  | Service/config summary (`Result` shape)|
| GET    | `/health`               | no   | Liveness                               |
| GET    | `/health/ready`         | no   | Readiness (503 until model loaded)     |
| GET    | `/metrics`              | no   | Prometheus (vLLM stats)                |

Auth: send `Authorization: Bearer <key>` or `X-API-Key: <key>` (keys from
`API_KEYS`). Empty `API_KEYS` disables auth — dev only.

## Consume from another app (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://YOUR_HOST:8000/v1", api_key="YOUR_KEY")

# Non-streaming
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)
print(resp.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Column type / date inference

`POST /api/v1/infer-schema` takes a file's columns (with sample values) and asks
the hosted model to infer each column's true type — in particular **whether a
string column is really a date/datetime, and its format**. Numbers stored as
strings are corrected too. The original fields are never mutated; the model only
contributes `InferredDataType` / `isDate` / `DateFormat`, which are merged back
in code.

- `InferredDataType` ∈ `string | integer | float | boolean | date | datetime`
  (`datetime` = has a time component; `date` = calendar date only).
- `DateFormat` uses Unicode/LDML tokens (`yyyy/MM/dd`,
  `yyyy-MM-dd HH:mm:ss OOOO (zzzz)`). Empty when `isDate` is false.

Prompts live in-repo at `ai/template/SchemaInferenceTemplate.py`. The call goes
to the already-loaded model in-process (no extra hop); when this feature moves to
its own service, only `engine/LLMClient.py` changes.

```bash
curl -s -H "Content-Type: application/json" -d '{
  "FileName":"customer.csv",
  "Columns":[
    {"ColumnName":"Joining Date","DataType":"string","SampleValues":["2025/09/13","2026/03/15"]}
  ]
}' localhost:8000/api/v1/infer-schema
```

```json
{
  "Data": {
    "FileName": "customer.csv",
    "Columns": [
      {
        "ColumnName": "Joining Date", "DataType": "string",
        "SampleValues": ["2025/09/13", "2026/03/15"],
        "InferredDataType": "date", "isDate": true, "DateFormat": "yyyy/MM/dd"
      }
    ]
  },
  "Status": 1,
  "Message": "JSON updated successfully!"
}
```

## Deploy with systemd

`/etc/systemd/system/platform-ai-service.service`:

```ini
[Unit]
Description=platform-ai-service (vLLM inference)
After=network-online.target

[Service]
User=aiservice
WorkingDirectory=/opt/platform-ai-service
EnvironmentFile=/opt/platform-ai-service/.env
Environment=HF_HOME=/opt/platform-ai-service/.cache/huggingface
ExecStart=/opt/platform-ai-service/.venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now platform-ai-service
```

## Verify end-to-end

```bash
curl -s localhost:8000/health/ready
curl -s -H "Authorization: Bearer YOUR_KEY" localhost:8000/v1/models
curl -s -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"hi"}]}' \
  localhost:8000/v1/chat/completions
# streaming:
curl -N -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"hi"}],"stream":true}' \
  localhost:8000/v1/chat/completions
```

## Scaling to multiple models (future)

This service hosts one model per process. To serve several, run one instance per
model on different ports and put a router/gateway in front that maps the request
`model` field to the right upstream.

## Notes / maintenance

- Bumping vLLM: `engine/ServingClient.py` self-adapts to minor kwarg drift
  (it passes only the kwargs each handler accepts). For a major bump, reconcile
  it against the new `vllm/entrypoints/openai/api_server.py::init_app_state`.
  Nothing else should need changes.
- OOM at load: lower `GPU_MEMORY_UTILIZATION` or `MAX_MODEL_LEN`, or set
  `QUANTIZATION`.
