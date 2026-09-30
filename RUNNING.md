# Running the platform locally (Submission 1, single machine)

| Service | Port | Start |
|---|---|---|
| Neo4j | 7474 / 7687 | `neo4j start` (user `neo4j`, password `fincrime-dev-2026`) |
| llama-server (Qwen3.5-9B) | 8080 | from `~/Ddrive/llamacpp_root/llama.cpp`, see below |
| ML engine | 8001 | `cd ml-engine && .venv/bin/python service.py` |
| Orchestrator gateway | 8000 | `cd orchestrator-service && .venv/bin/python gateway.py` |
| Dashboard | 5173 | `cd frontend-dashboard && npm run dev` |

## One-time setup

```bash
# Neo4j (already done on Dhruv's Mac)
brew install neo4j
neo4j-admin dbms set-initial-password fincrime-dev-2026   # before first start
neo4j start

# Python envs
cd ml-engine && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
cd orchestrator-service && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
cd frontend-dashboard && npm install && cd ..

# Train the model (writes ml-engine/artifacts/model.pkl)
cd ml-engine && .venv/bin/python train.py && cd ..

# Load the graph (idempotent)
cd orchestrator-service && .venv/bin/python db_loader.py && cd ..
```

## Start order

1. `neo4j start`
2. llama-server:
   ```bash
   cd ~/Ddrive/llamacpp_root/llama.cpp
   ./build/bin/llama-server -m models/qwen3.5-9b-q4_k_m.gguf --alias qwen3.5-9b-instruct \
     -ngl 99 -fa on -c 32768 -ctk turbo3 -ctv turbo3 -t 10 -np 1 --reasoning off \
     --host 127.0.0.1 --port 8080
   ```
3. ML engine, then gateway (the gateway builds the alert book at startup, ~12 s, and then
   pre-generates LLM narratives for CRITICAL/HIGH alerts in the background).
4. Dashboard at http://localhost:5173 (no `.env` needed; `VITE_USE_MOCKS=true` for demo data).

The gateway works without llama-server: explanations fall back to a rule-based narrative and
the status bar shows the LLM node offline. Opening an alert waits up to 10 s for its narrative
(`LLM_INLINE_WAIT_SECONDS`); if it isn't ready, reopen the alert once it finishes.

## Tests

```bash
cd ml-engine && .venv/bin/python -m pytest -q                         # unit + API, needs model.pkl
cd orchestrator-service && .venv/bin/python -m pytest -q test_gateway.py  # needs full stack up
```

## Config

Orchestrator settings live in `orchestrator-service/config.py`; override via env vars or
`orchestrator-service/.env` (see `.env.example`). Scoring weights: `alert_engine.py`.
