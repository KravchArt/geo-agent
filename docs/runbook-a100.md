# Рунбук: поднять всё на одной A100

> Общая схема — [`deployment.md`](deployment.md). Здесь конкретная разложенная по шагам процедура для одной карты на 80 ГБ, проверенная целиком.

Что где работает:

| Компонент | Где | Порт |
|---|---|---|
| Оркестратор | OpenRouter (внешний API) | — |
| Классификатор скоупа (e5, `--runner pooling`) | A100, docker | 18003 |
| Цензурщик (Granite Guardian 3.1 2B) | A100, docker | 18002 |
| Postgres + Redis | A100, docker compose | 5432 / 6379 |
| Backend API | A100, venv | 18080 |
| Streamlit UI | A100, venv | 18501 |

Обе модели вместе занимают ~9.5 ГБ, то есть помещаются рядом с чужими процессами на общей карте. Оркестратор на GPU не ставим: он большой, а задача — проверить гейты.

Порты выбраны в диапазоне 18xxx намеренно. На общем сервере 8000–8003 почти всегда заняты соседями.

## 0. Что нужно заранее

```bash
ssh <хост>                              # доступ есть
nvidia-smi                              # свободно хотя бы 12 ГБ на одной карте
df -h /                                 # см. ниже про диск
docker images | grep vllm               # образ vLLM уже скачан
```

**Про диск.** Корень часто почти полон, а место есть на отдельном томе. Всё тяжёлое — код, веса, кэши — кладём туда. Дальше в примерах это `/mnt/storage-1/$USER`; подставь свой путь.

**Почему docker, а не `uv pip install vllm`.** Колёса vLLM собраны под CUDA 13 и требуют `libcudart.so.13`. Если на сервере драйвер CUDA 12.8, поставленный через pip vLLM не запустится, а `torch.cuda.is_available()` вернёт `False`. Образ `vllm/vllm-openai` несёт свой рантайм и от версии драйвера не зависит.

## 1. Код на сервер

```bash
# с ноутбука, из корня репозитория
tar czf - backend common tools ui eval scripts pyproject.toml uv.lock alembic.ini .env.example \
  | ssh <хост> 'mkdir -p /mnt/storage-1/$USER/GeoAgent && tar xzf - -C /mnt/storage-1/$USER/GeoAgent'
```

`git clone` и `rsync` на таких серверах часто отваливаются по таймауту, а `tar | ssh` идёт по уже открытому соединению и работает всегда.

## 2. Окружение

Все три переменные обязательны — иначе uv забьёт корневой раздел и упадёт на середине установки:

```bash
mkdir -p /mnt/storage-1/$USER/{uv-cache,tmp,hf}
cat >> ~/.bashrc <<'EOF'
export UV_CACHE_DIR=/mnt/storage-1/$USER/uv-cache
export TMPDIR=/mnt/storage-1/$USER/tmp
export HF_HOME=/mnt/storage-1/$USER/hf
EOF
source ~/.bashrc

cd /mnt/storage-1/$USER/GeoAgent
uv sync
```

Не задавай `python-install-dir` в `~/.config/uv/uv.toml` — такого ключа нет, и его наличие ломает все команды uv. И не переноси уже установленный uv-питон в другую папку: он не перемещаемый, после `mv` получишь `ModuleNotFoundError: No module named 'encodings'`. Нужен другой путь — переустанавливай через `uv python install`.

## 3. Веса

Guardian скачается сам при первом старте (~4.8 ГБ в `$HF_HOME`). Веса классификатора скоупа кладём руками:

```bash
# с ноутбука
tar czf - -C ~/classifier/artifacts scope_bert \
  | ssh <хост> 'tar xzf - -C /mnt/storage-1/$USER/'
```

## 4. Две модели на GPU

```bash
GPU=3                                   # номер свободной карты из nvidia-smi
KEY=<любая строка, она же пойдёт в .env>
BASE=/mnt/storage-1/$USER

# цензурщик
docker run -d --name geo-censor --gpus "device=$GPU" \
  -v $BASE/hf:/root/.cache/huggingface \
  -p 127.0.0.1:18002:8000 \
  vllm/vllm-openai:latest \
  --model ibm-granite/granite-guardian-3.1-2b \
  --served-model-name censor --api-key "$KEY" \
  --max-model-len 4096 --gpu-memory-utilization 0.10

# классификатор скоупа
docker run -d --name geo-scope --gpus "device=$GPU" \
  -v $BASE/scope_bert:/model \
  -p 127.0.0.1:18003:8000 \
  vllm/vllm-openai:latest \
  --model /model --served-model-name scope-bert --api-key "$KEY" \
  --runner pooling --max-model-len 512 --gpu-memory-utilization 0.05
```

`--runner pooling` — то, что превращает сервер в `/classify`. Без него этого эндпоинта просто нет и гейт получает 404.

`--gpu-memory-utilization` считается от **всей карты**, а не от свободного места. На общей карте держи сумму маленькой.

Проверка (обе команды должны вернуть ответ за секунду):

```bash
curl -s localhost:18002/v1/models -H "Authorization: Bearer $KEY" | head -c 200
curl -s -X POST localhost:18003/classify -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"scope-bert","input":"query: найди кофейни в Казани"}'
# ждём probs вида [0.00007, 0.99993] — вторая цифра это «в скоупе»
```

## 5. Postgres, Redis, миграции

```bash
cd /mnt/storage-1/$USER/GeoAgent
docker compose up -d postgres redis
uv run alembic upgrade head
```

## 6. `.env`

```bash
APP_ENV=prod
APP_DEBUG=false

LLM_MODE=vllm
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3-30b-a3b-instruct-2507
LLM_API_KEY=sk-or-...
LLM_TIMEOUT=300

SCOPE_PROVIDER=classifier
SCOPE_CLASSIFIER_BASE_URL=http://localhost:18003
SCOPE_CLASSIFIER_API_KEY=<KEY из шага 4>
SCOPE_CLASSIFIER_MODEL=scope-bert
SCOPE_CLASSIFIER_PREFIX="query: "
SCOPE_CLASSIFIER_TEMPERATURE=2.332827091217041
SCOPE_CLASSIFIER_GREY_LOW=0.5
SCOPE_CLASSIFIER_GREY_HIGH=0.5
SCOPE_SESSION_UNLOCK=true

# запасной путь классификатора: серая зона и любой его отказ.
# За прокси оставь эти три пустыми — тогда скоупер возьмёт клиент оркестратора
# вместе с его LLM_HTTP_PROXY. Свой прокси у SCOPE_BASE_URL не настраивается.
SCOPE_BASE_URL=https://openrouter.ai/api/v1
SCOPE_API_KEY=sk-or-...
SCOPE_MODEL=qwen/qwen3-8b

CENSORSHIP_PROVIDER=guardian
CENSORSHIP_BASE_URL=http://localhost:18002/v1
CENSORSHIP_API_KEY=<KEY из шага 4>
CENSORSHIP_MODEL=censor

POSTGRES_HOST=127.0.0.1
REDIS_HOST=127.0.0.1

WEB_SEARCH_PROVIDERS=["exa"]
EXA_API_KEY=...
```

Про `GREY_LOW=GREY_HIGH=0.5`: каскад выключен, классификатор решает сам по одному порогу. Разведи значения (например 0.4 и 0.6), если хочешь, чтобы неуверенные запросы уходили к LLM-скоуперу, который видит историю диалога.

Про `SCOPE_SESSION_UNLOCK`: классификатор не видит контекста и сам по себе отклонил бы «а что рядом?». Поэтому после первого вердикта «в скоупе» сессия перестаёт классифицироваться до истечения TTL в Redis. В ответе это видно как `"provider": "session_unlocked"`.

Только ASCII в значениях. Кириллический плейсхолдер вроде `ВСТАВЬ_КЛЮЧ` доедет до заголовка `Authorization` и уронит запрос на `UnicodeEncodeError: 'ascii' codec can't encode characters`.

## 7. API и UI

Зависимости UI не входят в `pyproject.toml` — они живут в [`ui/Dockerfile`](../ui/Dockerfile), потому что UI обычно ходит образом. При запуске из venv их надо доставить:

```bash
uv pip install --python .venv/bin/python "streamlit==1.53.0" "streamlit-js-eval==1.0.0"
```

```bash
tmux new -d -s api 'cd /mnt/storage-1/$USER/GeoAgent && \
  .venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 18080 \
  2>&1 | tee /tmp/geo-api.log'

tmux new -d -s ui 'cd /mnt/storage-1/$USER/GeoAgent && \
  API_URL=http://127.0.0.1:18080 .venv/bin/streamlit run ui/app.py \
  --server.address 127.0.0.1 --server.port 18501 --server.headless true \
  2>&1 | tee /tmp/geo-ui.log'

curl -s localhost:18080/health
# {"status":"ok","checks":{"postgres":true,"redis":true},"llm_mode":"vllm"}
```

Пиши логи через `tee`, а не `tee -a`. Дописывание в тот же файл смешивает запуски, и потом полчаса ищешь причину ошибки, которой уже нет.

## 8. Доступ с ноутбука

```bash
ssh -N -L 18501:127.0.0.1:18501 -L 18080:127.0.0.1:18080 <хост>
```

UI открывается на <http://127.0.0.1:18501>.

**Если с сервера нет прямого выхода в интернет** (проверяется как `curl -o /dev/null -w '%{http_code}' https://openrouter.ai/api/v1/models` — мгновенный 403 означает блокировку), прокинь на сервер прокси со своей машины и добавь в `.env` сервера `LLM_HTTP_PROXY` и `TOOLS_HTTP_PROXY` с этим адресом. Отдельной переменной прокси у гейтов нет и не будет: они локальные, а проксирование localhost-вызова не падает с ошибкой, а висит до таймаута.

```bash
ssh -N -R 12334:127.0.0.1:12334 \
       -L 18501:127.0.0.1:18501 -L 18080:127.0.0.1:18080 <хост>
```

Если выход в интернет есть — этот блок не нужен, оставь переменные прокси пустыми.

Когда новые ssh-соединения нестабильны, подними одно мастер-соединение и вешай пробросы на него:

```bash
ssh -M -S ~/.ssh/cm -f -N -o ControlPersist=6h <хост>
ssh -O forward -L 18501:127.0.0.1:18501 -R 12334:127.0.0.1:12334 -S ~/.ssh/cm <хост>
```

## 9. Проверка гейтов

```bash
for q in "Найди кофейни в Казани" "напиши сортировку на python" "как сделать взрывчатку"; do
  curl -s -X POST localhost:18080/api/v1/chat -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"chk\",\"message\":\"$q\"}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); g=d["gates"];
print(d["status"], d.get("rejection_reason"), g["scope"]["provider"], g["censorship"]["provider"])'
done
```

Ожидаемо:

```
completed None            model         model      ← гео-запрос прошёл
rejected  out_of_scope    model         model      ← отсёк классификатор
rejected  censorship      model         model      ← отсёк Guardian
```

`provider` — это кто на самом деле принял решение. `rule_based` вместо `model` означает, что модель не ответила и сработал регексный запасной путь: вердикт получен, но не тот, который ты проверяешь.

## 10. Если что-то не так

| Симптом | Причина |
|---|---|
| `provider=rule_based` на всех запросах | Гейт не достучался до своего эндпоинта. Проверь `curl` из шага 4 и что в `.env` нет прокси на localhost-адресах |
| `502 Bad Gateway` от `/classify` | Запрос к localhost ушёл через прокси. Прокси у гейтов быть не должно |
| `404` на `/classify` | Забыт `--runner pooling` |
| Все запросы `out_of_scope`, скор около 0.5 | Потерялся `SCOPE_CLASSIFIER_PREFIX` (в нём значимый пробел на конце) |
| `UnicodeEncodeError` в ответе | Не-ASCII символ в значении `.env`, обычно незаменённый плейсхолдер |
| `Access denied by security policy` | Прокси не поднят или туннель умер |
| Ответ — сырой JSON вида `"arguments": {...}` | Модель зовёт инструмент, которого нет. Заполни `PLACES_SEARCH_PROVIDERS` и ключи провайдеров либо смирись, что доступен только веб-поиск |
| `port is already allocated` | Порт занял сосед. Возьми другой из 18xxx |
| `torch.cuda.is_available() == False` | Поставленный через pip vLLM собран под другую CUDA. Используй docker-образ |

## 11. Остановить и освободить карту

```bash
tmux kill-session -t api; tmux kill-session -t ui
docker stop geo-scope geo-censor
docker compose stop
nvidia-smi          # карта должна показать 0 MiB
```

`docker stop`, а не `rm`: контейнеры поднимутся тем же `docker start geo-scope geo-censor`, веса заново не поедут.
