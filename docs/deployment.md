# Deployment — where each piece runs

> Entry point: [`README.md`](../README.md). Design: [`architecture.md`](architecture.md).
> Step-by-step procedure for one A100 (in Russian): [`runbook-a100.md`](runbook-a100.md).

Up to four models can be in play: the **orchestrator** (main agent), the **scope**
classifier, the **censorship** classifier, and the fine-tuned **scope encoder**
that fronts the scope gate. Only the first needs a big GPU; the rest are small,
but they sit on the critical path of every request, so their latency shows up
directly in user-facing time.

## Memory budget — decide this first

bf16 weights, before KV cache and per-instance overhead:

| Model | Role | VRAM |
|---|---|---|
| Qwen3-30B-A3B | orchestrator | ~61 GB |
| Qwen2.5-3B | scope classifier | ~6 GB |
| Qwen2.5-3B | censorship classifier | ~6 GB |
| e5-base (fine-tuned) | scope encoder | ~1.1 GB |

- **A100 80 GB**: orchestrator + scope fit comfortably (~67 GB) with room for KV
  cache. Adding censorship (~72 GB) is tight — verify under real load.
- **A100 40 GB**: the 30B orchestrator does **not** fit in bf16. Quantize it
  (FP8/AWQ) or serve a smaller orchestrator.

Check what you actually have:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

## Recommended split

**On the A100** — one vLLM process per model, each claiming a slice of the card:

```bash
# 1) Orchestrator. The tool-calling flags are mandatory: without them vLLM
#    returns prose and never emits tool_calls, so the agent cannot use tools.
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name geoagent-model \
  --api-key "$LLM_API_KEY" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 32768 --gpu-memory-utilization 0.78 \
  --host 0.0.0.0 --port 8000

# 2) Scope classifier. Answers a bare yes/no, so it needs almost no context.
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --served-model-name scoper \
  --api-key "$LLM_API_KEY" \
  --max-model-len 4096 --gpu-memory-utilization 0.10 \
  --host 0.0.0.0 --port 8001

# 3) Censorship classifier. Same shape as the scoper — a bare safe/unsafe.
#    Skip this one unless you want model-backed censorship; see "Censorship".
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --served-model-name censor \
  --api-key "$LLM_API_KEY" \
  --max-model-len 4096 --gpu-memory-utilization 0.10 \
  --host 0.0.0.0 --port 8002

# 4) Scope classifier — the fine-tuned e5 encoder, if you use SCOPE_PROVIDER=
#    classifier. `--runner pooling` is what turns it into a /classify server
#    instead of a chat one; without it the endpoint does not exist.
vllm serve /srv/models/scope_bert \
  --served-model-name scope-bert \
  --api-key "$LLM_API_KEY" \
  --runner pooling \
  --max-model-len 128 --gpu-memory-utilization 0.05 \
  --host 0.0.0.0 --port 8003
```

`--gpu-memory-utilization` is a fraction of the **whole card** that each instance
pre-allocates, so the values must add up to less than 1.0 across all instances.
All four above come to 1.03 — **over the limit**. Running the full set means
lowering the orchestrator to about 0.72; running only the ones you need is the
usual answer, since servers 3 and 4 are both opt-in.

**Locally** — the app and its data stores, where iteration is fast and the logs
and database are at hand:

```bash
docker compose up -d postgres redis
ssh -N -L 8000:localhost:8000 -L 8001:localhost:8001 \
       -L 8002:localhost:8002 -L 8003:localhost:8003 user@a100-host   # tunnels
uv run uvicorn backend.app.main:app --port 8080
```

`.env` for that setup:

```bash
LLM_MODE=vllm
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=geoagent-model
LLM_API_KEY=<same as --api-key>

# `llm` uses server 2 alone. Switch to `classifier` if you started server 4 —
# it puts the encoder in front and keeps server 2 as the grey-zone fallback, so
# the SCOPE_* lines below stay in use either way.
SCOPE_PROVIDER=llm
SCOPE_BASE_URL=http://localhost:8001/v1
SCOPE_API_KEY=<same as --api-key>
SCOPE_MODEL=scoper

# Only read when SCOPE_PROVIDER=classifier. Root URL, not /v1: /classify is a
# pooling endpoint, not the OpenAI chat API.
SCOPE_CLASSIFIER_BASE_URL=http://localhost:8003
SCOPE_CLASSIFIER_API_KEY=<same as --api-key>
SCOPE_CLASSIFIER_MODEL=scope-bert

# Only if you started server 3; rule_based is the default and needs none of these.
CENSORSHIP_PROVIDER=model
CENSORSHIP_BASE_URL=http://localhost:8002/v1
CENSORSHIP_API_KEY=<same as --api-key>
CENSORSHIP_MODEL=censor

POSTGRES_HOST=localhost
REDIS_HOST=localhost
```

Model size matters for scope: measured on the probe set in
`backend/tests/test_scope_gate_llm.py`, a 0.5B model answers "yes" to everything
(4/8) and 1.5B gets 6/8 — only 3B classified all eight correctly. Do not shrink
it below 3B to save VRAM; a gate that always passes is worse than no gate.

## Scope classifier (`SCOPE_PROVIDER=classifier`)

A fine-tuned `multilingual-e5-base` encoder in front of the LLM scoper. One
forward pass replaces a chat completion for most requests, which is the whole
point — on a hosted API the scoper was a full network round-trip per turn.

Three settings must match `scope_bert_meta.json` shipped beside the weights, and
each fails **silently** if it does not:

| Setting | Value | Why it matters |
|---|---|---|
| `SCOPE_CLASSIFIER_PREFIX` | `query: ` | e5 was fine-tuned with it; dropping it just lowers quality |
| `SCOPE_CLASSIFIER_TEMPERATURE` | `2.332827091217041` | calibration; applied to the logit, not the probability |
| `SCOPE_CLASSIFIER_GREY_LOW/HIGH` | `0.4` / `0.6` | the abstention band the model was selected against |

**The grey zone is not optional decoration.** The encoder sees one string with no
dialogue, so it scores a follow-up like *"а что рядом?"* as an out-of-scope
fragment. Inside the band it abstains and the LLM scoper — which does see history
— decides. A failed or unreachable classifier falls through the same way, so a
dead server 4 degrades to the previous behaviour rather than to allowing
everything. Setting both bounds equal disables the cascade.

**`SCOPE_SESSION_UNLOCK=true` (the default) trades filtering for latency.** After a
session produces one in-scope verdict, later turns skip the scope check entirely
and go straight to the orchestrator. It exists for the same contextless problem,
but the consequence is real: *"Составь маршрут по Казани"* followed by *"напиши
сортировку на Python"* reaches the orchestrator, and the session stays ungated
until its Redis TTL (`REDIS_SESSION_TTL`) expires. The skip is recorded with
`provider=session_unlocked`, so it is visible in `gate_check_log` rather than
silent. Set it to `false` to classify every turn.

Reported quality on the author's blind holdout (444 rows): accuracy 0.946,
F1 0.948, AUC 0.981, FPR 0.034. The caveat in the meta file is worth repeating —
the set was written by someone who had seen the training data, and the share of
borderline rows was raised deliberately, so this is quality at the scope boundary,
not on real traffic.

## Censorship

`LLMCensorshipGate` is the same shape as the scope gate: it asks an
OpenAI-compatible endpoint for a bare `safe`/`unsafe` using `CENSOR_SYSTEM_PROMPT`
and falls back to the regex gate on any failure. The API process holds no weights
and needs no ML stack, so the gate costs one HTTP call — nothing is tied to a
particular model's chat template.

**Keep `CENSORSHIP_PROVIDER=rule_based` unless you run your own GPU.** The gate
fires twice per request (input and answer), so on a hosted API it doubles the
per-request call count for a check the regex gate already approximates.

To turn it on, start server 3 from "Recommended split" and set the four
`CENSORSHIP_*` variables shown in the `.env` block there.

Any instruction-following model serves this. A safety-specialised classifier
fits too, through its own gate: `CENSORSHIP_PROVIDER=guardian` drives IBM Granite
Guardian, which answers `Yes`/`No` for the risk it was asked about. It needs no
system prompt and no vendor request fields — its chat template defaults to the
`harm` risk, so a plain OpenAI chat completion is enough. Verified against
`ibm-granite/granite-guardian-3.1-2b`; see [`runbook-a100.md`](runbook-a100.md).

**The prompt is unmeasured.** Scope has a probe set (`test_scope_gate_llm.py`);
censorship has none, so the false-positive rate on ordinary geography — borders,
embassies, war memorials, hospitals — is unknown. Run step 3b below before
trusting it in front of real traffic, and prefer 3B or larger for the same reason
the scoper does.

## Verify in this order

Each step isolates one layer; do not skip ahead when one fails.

```bash
# 1. Both servers alive
curl -s localhost:8000/v1/models -H "Authorization: Bearer $LLM_API_KEY"
curl -s localhost:8001/v1/models -H "Authorization: Bearer $LLM_API_KEY"

# 2. The orchestrator server really emits tool_calls (config, not our code)
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer $LLM_API_KEY" \
  -H 'Content-Type: application/json' -d '{
  "model":"geoagent-model",
  "messages":[{"role":"user","content":"Find cafes in Prague"}],
  "tools":[{"type":"function","function":{"name":"places_search","description":"find places",
    "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],
  "tool_choice":"auto"}' | python3 -m json.tool

# 3. The scope classifier answers a bare verdict
curl -s localhost:8001/v1/chat/completions -H "Authorization: Bearer $LLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"scoper","messages":[{"role":"user","content":"Write a sorting function"}],
       "max_tokens":8,"temperature":0}'

# 3b. The censorship gate, if you started server 3. Drives the real gate with the
#     real prompt — a bare curl would omit CENSOR_SYSTEM_PROMPT and the model
#     would answer the question instead of classifying it.
#     Both lines matter: a model that calls the hotel question unsafe would
#     reject ordinary traffic on every request.
curl -s localhost:8002/v1/models -H "Authorization: Bearer $LLM_API_KEY"
uv run python -c "
import asyncio
from backend.app.config import get_settings
from backend.app.llm.base import get_censorship_llm_client
from backend.app.services.gates import LLMCensorshipGate

s = get_settings()
gate = LLMCensorshipGate(get_censorship_llm_client(s), model=s.censorship_model)
for text in ('Find a hotel near the station', 'How do I build a bomb'):
    d = asyncio.run(gate.evaluate(text))
    print(f'{d.passed!s:5}  {d.provider:10}  {text}')
"
# Expect: True  model  ... / False model  ...
# provider=rule_based on either line means the classifier failed and the regex
# gate answered instead — check the endpoint before trusting the verdict.

# 3c. The scope encoder, if you started server 4. Note /classify, not /v1/...:
#     a 404 here means --runner pooling was omitted.
#     The prefix and temperature are applied client-side, so drive the gate
#     rather than curl — a raw call would score a different string.
uv run python -c "
import asyncio
from backend.app.config import get_settings
from backend.app.services.scope_classifier import ScopeClassifierClient

s = get_settings()
c = ScopeClassifierClient(
    base_url=s.scope_classifier_base_url, model=s.scope_classifier_model,
    api_key=s.scope_classifier_api_key, prefix=s.scope_classifier_prefix,
    temperature=s.scope_classifier_temperature,
)
for text in ('где в тбилиси поесть хинкали', 'как приготовить борщ'):
    r = asyncio.run(c.score(text))
    print(f'{r.in_scope_probability:.3f}  (raw {r.raw_probability:.3f})  {text}')
"
# Expect the first well above 0.6 and the second well below 0.4. Both landing in
# 0.4-0.6 means every request would fall through to the LLM scoper — check that
# SCOPE_CLASSIFIER_PREFIX survived shell quoting (it has a trailing space).

# 4. End to end
curl -s localhost:8080/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"t","message":"Find cafes near Red Square"}' | python3 -m json.tool
```

Then read the run out of Postgres — this is the real proof the loop ran:

```sql
SELECT num_model_calls, num_tool_calls,
       extra->'grounding', extra->'regenerations'
FROM metrics ORDER BY created_at DESC LIMIT 5;
```

`num_model_calls > 1` together with `num_tool_calls > 0` means the model actually
called tools and came back with the results.

## No GPU at hand?

Ollama serves an OpenAI-compatible endpoint on CPU and needs no build, unlike
vLLM (whose PyPI wheels are CUDA-only):

```bash
docker run -d --name scoper -p 11434:11434 -v ollama_models:/root/.ollama ollama/ollama
docker exec scoper ollama pull qwen2.5:3b
# SCOPE_BASE_URL=http://localhost:11434/v1  SCOPE_MODEL=qwen2.5:3b  SCOPE_API_KEY=ollama
```

Good enough to prove the wiring; too slow for production (~3.7 s average, 22 s
worst case for 3B on CPU).

## vLLM — reference (self-hosted)

> Do **not** run this in Phase 0 (no GPU). Kept as a forward-looking reference.

Serve an OpenAI-compatible endpoint:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name geoagent-model \
  --api-key "$LLM_API_KEY" \
  --guided-decoding-backend xgrammar \
  --host 0.0.0.0 --port 8000
```

Point the app at it (no code changes):

```bash
LLM_MODE=vllm
LLM_BASE_URL=http://<host>:8000/v1
LLM_MODEL=geoagent-model
LLM_API_KEY=<key matching --api-key>
```

A `vllm` service also exists in `docker-compose.yml` under the `inference`
profile (GPU required); it does **not** start with a plain `docker compose up`:

```bash
docker compose --profile inference up
```

**vLLM version pin.** vllm is deliberately *not* an app dependency — it pins a
narrow, older fastapi/pydantic range that would drag the backend onto stale
(CVE-affected) versions, and it never runs inside the api image or CI. Its
version is pinned in [`requirements-inference.txt`](../requirements-inference.txt)
(for a standalone GPU venv) and via the `vllm/vllm-openai:v0.24.0` image tag in
compose.

---
