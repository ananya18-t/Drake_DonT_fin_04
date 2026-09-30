# Drake&DonT

**Financial Crime & Insider Risk Intelligence Platform**

Drake&DonT links employee access logs, account states and money-flow graphs so that insider misuse and money laundering are caught *together* instead of in separate silos. Graph pattern detection (Neo4j), unsupervised ML anomaly scoring (Isolation Forest) and a **locally hosted LLM** (Qwen3.5-9B on llama.cpp) produce explainable, evidence-cited alerts for compliance reviewers. No data leaves the machine.

> Full technical details (schemas, features, scoring formula, API, results): [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
> Start-up cheat sheet: [RUNNING.md](RUNNING.md)

---

## Why

A compliance analyst overrides an alert on a dormant account, and minutes later that account starts sending $9,500 wires, just under the $10,000 reporting threshold. Access monitoring sees a routine override. Transaction monitoring sees a few small wires. Neither system sees the crime. Drake&DonT puts both in one graph and connects them.

## How it works

```text
 CSV data ──► Neo4j graph ──► Cypher detectors ──┐
 (data-engine)  (Employee, Account,               │   insider action → transfer
                Transaction)                      │   circular money flow
                                                  │   structuring below $10k
                                                  ▼
                                   ML Engine (Isolation Forest, :8001)
                                   scores every transaction involved
                                                  │
                                                  ▼
                                   Gateway (FastAPI, :8000)
                                   composite risk = 50% graph + 50% ML
                                                  │
                           ┌──────────────────────┴──────────────────┐
                           ▼                                         ▼
                 Local LLM (Qwen3.5-9B, :8080)             Dashboard (React, :5173)
                 writes the audit narrative                graph · timeline · evidence
                 with evidence IDs                         assign · freeze · escalate · PDF
```

## Results on the current dataset

10,000 transactions, 3% suspicious, with two fraud scenarios hidden in benign traffic.

| | Result |
|---|---|
| ML recall on held-out data | **100%** of fraud transactions caught |
| ML false positive rate | **1.9%** |
| ML scoring latency | ~1.5 ms features, ~13 ms full score |
| Scenario A: insider override + $9,500 structuring from 72 dormant accounts | **CRITICAL**, score 99.8 |
| Scenario B: KYC phone change + A→B→C→A circular transfers | **CRITICAL**, score 92.5 |
| Background noise (23 alerts) | 1 HIGH, 3 MEDIUM, 19 LOW |
| LLM narrative | ~20 s per new alert, fully local on Apple GPU (Metal) |

## Tech stack

| Layer | Technology |
|---|---|
| Graph database | Neo4j 2026.09 |
| Gateway | Python, FastAPI, httpx, neo4j driver, fpdf2 |
| ML | scikit-learn `IsolationForest`, pandas, NumPy |
| LLM | Qwen3.5-9B-Instruct (GGUF Q4_K_M) on llama.cpp (TurboQuant fork: Metal, flash attention, `turbo3` KV cache) |
| Frontend | React 18, Vite, Tailwind CSS, Cytoscape.js, Lucide |
| Data | Python, Faker, Pydantic v2 (seed 42, reproducible) |

## Repository layout

| Folder | Owner | Contents |
|---|---|---|
| [`data-engine/`](data-engine/) | Ananya | Schemas, scenario generators, generated CSVs in `data/` |
| [`ml-engine/`](ml-engine/) | Sameep | Feature extraction, training, inference API, tests |
| [`orchestrator-service/`](orchestrator-service/) | Dhruv | Neo4j loader and queries, alert engine, LLM explainer, gateway, PDF dossier, tests |
| [`frontend-dashboard/`](frontend-dashboard/) | Gayatri | React investigator dashboard |

---

## Quick start (macOS, Apple Silicon)

### Prerequisites

- Homebrew, Python 3.11+, Node 18+
- A llama.cpp build with the Qwen3.5-9B Q4_K_M GGUF model. This project uses the TurboQuant fork for the `turbo3` cache flags; mainline llama.cpp works if you drop `-ctk/-ctv turbo3`.
- About 8 GB of free RAM for the model

### One-time setup

```bash
# Neo4j
brew install neo4j
neo4j-admin dbms set-initial-password fincrime-dev-2026
neo4j start

# Python environments
cd ml-engine && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
cd orchestrator-service && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..

# Frontend
cd frontend-dashboard && npm install && cd ..

# Train the model and load the graph
cd ml-engine && .venv/bin/python train.py && cd ..
cd orchestrator-service && .venv/bin/python db_loader.py && cd ..
```

Optional: regenerate the dataset with `python data-engine/generator.py`. Then re-run `train.py` and `db_loader.py`.

### Run

Start each service in its own terminal, in this order:

```bash
# 1. Graph database
neo4j start

# 2. Local LLM (from your llama.cpp directory)
./build/bin/llama-server -m models/qwen3.5-9b-q4_k_m.gguf --alias qwen3.5-9b-instruct \
  -ngl 99 -fa on -c 32768 -ctk turbo3 -ctv turbo3 -t 10 -np 1 --reasoning off \
  --host 127.0.0.1 --port 8080

# 3. ML engine
cd ml-engine && .venv/bin/python service.py

# 4. Gateway (builds alerts in ~13 s on startup)
cd orchestrator-service && .venv/bin/python gateway.py

# 5. Dashboard
cd frontend-dashboard && npm run dev
```

Open **http://localhost:5173**.

> Don't copy `frontend-dashboard/.env.example` to `.env` unless you want demo data. It sets `VITE_USE_MOCKS=true`. With no `.env`, the dashboard uses the real gateway at `http://localhost:8000`.

Check that all nodes are online:

```bash
curl -s localhost:8000/api/v1/system/status | python3 -m json.tool
```

The platform still works without the LLM. Explanations fall back to a rule-based narrative, and the dashboard shows the LLM node as offline.

### Ports

| Service | Port |
|---|---|
| Neo4j browser / Bolt | 7474 / 7687 |
| Local LLM (llama-server) | 8080 |
| ML engine | 8001 |
| Gateway | 8000 |
| Dashboard | 5173 |

---

## API

The dashboard talks only to the gateway, under `/api/v1`:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/alerts?status=&risk_level=&limit=&refresh=` | Alert queue, most severe first |
| GET | `/alerts/{id}/investigation` | Graph, timeline, LLM explanation, fraud window |
| POST | `/cases/{id}/assign` | Assign a reviewer |
| POST | `/cases/{id}/freeze` | Freeze linked accounts (written to Neo4j) |
| POST | `/cases/{id}/escalate` | Escalate to the compliance lead |
| GET | `/cases/{id}/dossier?format=pdf\|json` | Download the evidence dossier |
| GET | `/reviewers` | Eligible reviewers (flagged insiders excluded) |
| GET | `/system/status` | Node health and headline metrics |

Interactive docs: http://localhost:8000/docs (gateway) and http://localhost:8001/docs (ML engine).

## Configuration

Gateway settings live in [`orchestrator-service/config.py`](orchestrator-service/config.py). Override them with environment variables or `orchestrator-service/.env` (see `.env.example`). The main settings are `NEO4J_*`, `ML_SERVICE_URL`, `LLM_BASE_URL`, detector windows and thresholds, and `MAX_GRAPH_ACCOUNTS`. Scoring weights are in [`alert_engine.py`](orchestrator-service/alert_engine.py).

## Tests

```bash
cd ml-engine && .venv/bin/python -m pytest -q                              # 8 tests
cd orchestrator-service && .venv/bin/python -m pytest -q test_gateway.py   # 11 tests, needs the stack running
```

## Known limitations

- Case assignments and escalations are kept in memory and reset when the gateway restarts. Frozen account status persists in Neo4j.
- The synthetic data covers all of 2026, so some events are dated in the future.
- There is no authentication yet, and the Neo4j password is a local development default.
- Everything runs on one machine for Submission 1. A two-node Tailscale deployment is planned.

## Team

| Member | Area |
|---|---|
| Ananya | Data Engine |
| Sameep | ML Engine |
| Dhruv | Orchestrator, Graph & Local LLM |
| Gayatri | Frontend Dashboard |
