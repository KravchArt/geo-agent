# GeoAgent evaluation

## 1. Fill model answers

`fill_answers` sends each query to GeoAgent and checkpoints the answer after
every successful row:

```powershell
uv run --extra eval python -m eval.fill_answers `
  "eval\auto eval (only web_search).xlsx" `
  --query-column user_query `
  --answer-column model_answer
```

Use `eval\auto eval (web + places_search).xlsx` instead when both tool sets
are enabled. GeoAgent must already be reachable at
`http://localhost:8000/api/v1/chat`; use `--agent-url` to point the runner at a
different deployment. Existing non-empty answers are skipped unless
`--overwrite` is passed.

## 2. Start the judge on the GPU host

Recommended default: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`. It is not
from the Qwen family and requires about 55 GB VRAM in BF16/FP16, so it fits on a
single A100 80 GB. Run it separately from the main GeoAgent model unless the
combined vLLM memory reservations are known to fit:

```bash
export JUDGE_API_KEY='replace-me'

vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --served-model-name geo-judge \
  --api-key "$JUDGE_API_KEY" \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 \
  --port 8002
```

Expose only a local SSH tunnel:

```powershell
ssh -N -L 1234:localhost:8002 user@a100-host
```

Set the local client variables:

```powershell
$env:JUDGE_BASE_URL = "http://localhost:1234/v1"
$env:JUDGE_MODEL = "geo-judge"
$env:JUDGE_API_KEY = "replace-me"
```

## 3. Score answers

In-place mode mirrors `fill_answers` and checkpoints each row:

```powershell
uv run --extra eval python -m eval.judge_answers `
  "eval\auto eval (only web_search).xlsx" --dry-run
uv run --extra eval python -m eval.judge_answers `
  "eval\auto eval (only web_search).xlsx"
```

For the configured A100 server, the complete PowerShell wrapper checks SSH,
loads the API key, opens a temporary tunnel, verifies vLLM, and evaluates every
remaining row:

```powershell
.\eval\run_judge_all.ps1
```

Rows that already have scores are skipped. Pass `-Overwrite` only when every
answer must be evaluated again.

The full workbook template is also supported. `judge_answers` selects its
`Results` worksheet and joins reference criteria by `query_id` from
`Eval Dataset`:

```powershell
uv run --extra eval python -m eval.judge_answers `
  "eval\geoagent eval.xlsx" --dry-run
uv run --extra eval python -m eval.judge_answers `
  "eval\geoagent eval.xlsx"
```

Useful flags:

- `--limit N` evaluates only the first N eligible rows.
- `--overwrite` re-evaluates rows that already have scores.
- `--base-url`, `--model`, and `--api-key` override environment variables.
- `--temperature 0` is the default for reproducible scoring.

The compact workbook may contain only `user_query` and `model_answer`; in that
case the judge classifies route vs non-route requests. When a `scenario` column
exists, it is authoritative. When available, `tool_outputs` are passed as
evidence.

## Rubric and overall score

All component scores are integers from 0 to 5.

| Scenario | Intent | Relevance | Factuality | Feasibility | Format |
|---|---:|---:|---:|---:|---:|
| Non-route | 0.25 | 0.25 | 0.35 | — | 0.15 |
| Route | 0.20 | 0.20 | 0.25 | 0.25 | 0.10 |

`critical_error=true` is reserved for an answer-invalidating problem such as a
wrong city, fabricated key place, unsafe advice, impossible itinerary, or a
hard `must_not_have` violation. It caps `overall_score` at 2.0. The workbook
keeps the overall score as an auditable Excel formula.
