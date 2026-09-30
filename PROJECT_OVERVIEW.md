# Drake&DonT: Financial Crime & Insider Risk Intelligence Platform

Project overview, current as of 2026-09-29 (Submission 1).
This document describes the project as it exists in the codebase now. All numbers come from the current dataset and the current trained model.

---

## 1. Summary

| Item | Value |
|---|---|
| Product name (shown in UI) | Drake&DonT |
| Domain | Anti-Money Laundering (AML) and insider threat detection |
| Core idea | Link employee access logs, account states and money-flow graphs so that insider misuse and financial crime are detected together instead of in separate silos |
| Detection approach | Graph pattern queries (Neo4j) + unsupervised ML anomaly scoring (Isolation Forest) + a local LLM that writes an explainable audit narrative |
| Deployment for Submission 1 | Everything runs on one Apple Silicon Mac (M4 Pro, 24 GB RAM) on `localhost` |
| Privacy property | No data leaves the machine. The LLM runs locally through llama.cpp on the Mac GPU (Metal) |

### Problem

Banks usually monitor two things separately:

1. Insider privilege misuse, for example an employee overriding a compliance alert or changing a customer's phone number (KYC edit).
2. Financial crime in money movement, for example structuring (many transfers just under a reporting threshold) or circular transfers (money moving A -> B -> C -> A).

When these are monitored in silos, an insider who helps launder money looks harmless in each system on its own.

### Solution

The platform puts employees, accounts and transactions in one graph. It then:

1. Finds suspicious graph patterns with Cypher queries.
2. Scores every transaction involved with an unsupervised ML model.
3. Combines both into one risk score per alert.
4. Uses a local LLM to write a regulator-style explanation that cites evidence IDs.
5. Shows everything in an investigator dashboard: graph, timeline, explanation, and case actions such as assign, freeze, escalate and export PDF.

---

## 2. Team and ownership

Each member contributes 25%.

| Member | Area | Owns |
|---|---|---|
| Ananya | Data Engine | Pydantic schemas and synthetic data generator with injected fraud scenarios (`data-engine/`) |
| Sameep | ML Engine | Isolation Forest training, evaluation (accuracy / FPR), inference API (`ml-engine/`) |
| Dhruv | Orchestrator & Graph | Neo4j ingestion and Cypher queries, local Qwen LLM hosting, FastAPI gateway (`orchestrator-service/`) |
| Gayatri | Frontend Dashboard | Vite/React UI: graph canvas, timeline, evidence panel, case manager (`frontend-dashboard/`) |

---

## 3. Technology stack

| Layer | Technology | Notes |
|---|---|---|
| Graph database | Neo4j 2026.09 (Community, installed with Homebrew) | Stores Employee, Account and Transaction nodes and their relationships |
| Backend gateway | Python 3.14, FastAPI, Uvicorn, httpx, pydantic-settings, neo4j driver, fpdf2 | Orchestrates Neo4j, the ML service and the LLM, and serves the UI API |
| ML engine | scikit-learn 1.9.1 `IsolationForest`, pandas, NumPy, joblib, FastAPI | Unsupervised anomaly scoring of transactions |
| LLM | Qwen3.5-9B-Instruct, GGUF Q4_K_M quantization (5.7 GB file) | Served by `llama-server` from the TurboQuant fork of llama.cpp |
| LLM runtime | llama.cpp fork `TheTom/llama-cpp-turboquant`, Apple Metal GPU, flash attention, `turbo3` KV-cache compression | OpenAI-compatible API on port 8080 |
| Frontend | React 18, Vite 5, Tailwind CSS 3, Cytoscape.js, Lucide icons, Axios | Light theme |
| Data generation | Python, Faker, NumPy, Pydantic v2 | Reproducible (seed 42) |
| Tests | pytest | ML unit and API tests, gateway integration tests |

---

## 4. Architecture

### 4.1 Services and ports

| Service | Port | Process | Start command (run from project root unless noted) |
|---|---|---|---|
| Neo4j browser UI | 7474 | Neo4j | `neo4j start` |
| Neo4j Bolt | 7687 | Neo4j | (same process) |
| Local LLM (`llama-server`) | 8080 | llama.cpp | see section 11 |
| ML Anomaly Engine | 8001 | FastAPI | `cd ml-engine && .venv/bin/python service.py` |
| Orchestrator Gateway | 8000 | FastAPI | `cd orchestrator-service && .venv/bin/python gateway.py` |
| Frontend dashboard | 5173 | Vite dev server | `cd frontend-dashboard && npm run dev` |

The long-term design is a two-node Tailscale mesh VPN. For Submission 1, all services run on one machine on `localhost`.

### 4.2 Data flow

1. The **Data Engine** writes CSV files to `data-engine/data/`.
2. `db_loader.py` loads the CSVs into **Neo4j**.
3. The **ML Engine** trains on the same CSVs and saves `ml-engine/artifacts/model.pkl`.
4. When the **Gateway** starts, it:
   1. runs the three graph detectors in Neo4j,
   2. groups the hits into alerts,
   3. fetches each involved sender's history and context from Neo4j,
   4. sends the transactions to the ML Engine `/score-batch` endpoint,
   5. computes a composite score and risk level for each alert,
   6. queues LLM narratives for CRITICAL and HIGH alerts in the background.
5. The **Frontend** calls the Gateway under `/api/v1`. It never talks to Neo4j, the ML engine or the LLM directly.
6. When a reviewer opens an alert, the Gateway returns the graph, timeline and explanation. The explanation comes from the LLM, or from a rule-based fallback if the LLM text is not ready yet.

### 4.3 Graph model in Neo4j

| Element | Kind | Key / properties |
|---|---|---|
| `Employee` | Node | `emp_id` (unique), `name`, `department`, `role`, `access_tier`, `is_privileged` |
| `Account` | Node | `account_id` (unique), `customer_name`, `balance`, `status`, `risk_category` |
| `Transaction` | Node | `tx_id` (unique), `amount`, `timestamp` (Neo4j datetime), `type` |
| `(Employee)-[:ACCESSED]->(Account)` | Relationship | `log_id`, `action_type`, `timestamp` (datetime) |
| `(Account)-[:SENT]->(Transaction)` | Relationship | sender side of a transfer |
| `(Transaction)-[:TO]->(Account)` | Relationship | receiver side of a transfer |

Indexes: unique constraints on `emp_id`, `account_id` and `tx_id`, plus indexes on `Transaction.timestamp` and `Transaction.amount`.

Loaded counts: 100 Employee nodes, 1,000 Account nodes, 10,000 Transaction nodes, 5,000 ACCESSED edges, 20,000 SENT/TO edges. Loading takes about 3 seconds.

---

## 5. Data Engine (`data-engine/`)

### 5.1 Files

| File | Purpose |
|---|---|
| `schemas.py` | Strict Pydantic v2 models for every record type |
| `scenarios.py` | Generators for the three injected scenarios (A, B, C) |
| `generator.py` | Builds the full dataset: benign background traffic plus scenarios. Validates and writes CSVs. CLI flags `--output-dir` and `--seed` (default 42) |
| `data/*.csv` | Generated dataset |

### 5.2 Schemas

| Record | Fields | Allowed values |
|---|---|---|
| Employee | `emp_id`, `name`, `department`, `role`, `access_tier`, `is_privileged` | `is_privileged`: true or false |
| Account | `account_id`, `customer_name`, `balance`, `status`, `risk_category` | `status`: ACTIVE, DORMANT, FROZEN |
| AccessLog | `log_id`, `emp_id`, `account_id`, `action`, `timestamp` | `action`: VIEW_PROFILE, MODIFY_PHONE, OVERRIDE_ALERT, MANUAL_UNFREEZE |
| Transaction | `tx_id`, `sender_account`, `receiver_account`, `amount`, `timestamp`, `type` | `type`: WIRE, INTERNAL_TRANSFER, ATM |
| GroundTruthLabel | `target_id`, `target_type`, `label`, `scenario_tag` | `target_type`: TRANSACTION, EMPLOYEE, ACCOUNT. `label`: SUSPICIOUS, BENIGN |

### 5.3 Dataset size

| File | Rows |
|---|---|
| `employees.csv` | 100 |
| `accounts.csv` | 1,000 |
| `transactions.csv` | 10,000 |
| `access_logs.csv` | 5,000 |
| `ground_truth_labels.csv` | 11,100 (one row per transaction, account and employee) |

Suspicious transactions are exactly 3% (300 of 10,000). Timestamps span 2026-01-01 to 2026-12-31.

Access-log action counts: VIEW_PROFILE 4,246, MODIFY_PHONE 535, OVERRIDE_ALERT 171, MANUAL_UNFREEZE 48.
Transaction type counts: INTERNAL_TRANSFER 6,805, WIRE 2,440, ATM 755.

### 5.4 Injected scenarios

| Scenario | Label | What happens | Size |
|---|---|---|---|
| A: Smurfing with insider help | SUSPICIOUS | Employee EMP-A-001 (Jordan Lee, Senior Compliance Analyst, privileged) performs OVERRIDE_ALERT on dormant accounts. Each dormant account then sends four wires of exactly $9,500 (just under the $10,000 threshold) to external beneficiaries, within about 2 hours of the override | 72 dormant accounts, 288 transactions |
| B: Circular money flow | SUSPICIOUS | Employee EMP-B-001 (Morgan Patel, KYC Operations Specialist) performs MODIFY_PHONE on an account. Funds then move A -> B -> C -> A in three transfers about 16 hours apart ($180k to $260k each) | 4 cycles, 12 transactions |
| C: Legitimate high volume | BENIGN (hard negative) | A payroll batch of 12 routine payments and 6 large high-net-worth wires. Unusual but legitimate, so it tests false positives | 18 transactions |
| Background | BENIGN | Random normal employees, accounts, access logs and transfers | the remaining rows |

---

## 6. ML Engine (`ml-engine/`)

### 6.1 Files

| File | Purpose |
|---|---|
| `feature_extraction.py` | Builds an 18-column numeric feature matrix per transaction. Accepts CSV paths, DataFrames or dicts, and both column-name styles (`sender_account` or `sender_account_id`, `action` or `action_type`) |
| `train.py` | Trains the Isolation Forest without labels, evaluates on a held-out split, saves the model bundle |
| `service.py` | FastAPI inference service on port 8001 |
| `test_service.py` | 8 pytest tests (features, health, fraud detection, false positives, batch equals single, validation, latency) |
| `artifacts/model.pkl` | Saved bundle: model, scaler, feature names, sorted training-score reference, contamination |
| `requirements.txt` | numpy, pandas, scipy, scikit-learn, joblib, fastapi, uvicorn, pydantic, httpx, pytest |

### 6.2 Features (18)

| # | Feature | Meaning |
|---|---|---|
| 1 | `amount` | Transaction amount |
| 2 | `amount_missing` | 1 if amount is missing or invalid |
| 3 | `sender_missing` | 1 if sender account is missing |
| 4 | `timestamp_missing` | 1 if timestamp is missing |
| 5 | `tx_count_1h` | Number of transactions by the same sender in the last 1 hour (including this one) |
| 6 | `tx_sum_1h` | Sum of those amounts |
| 7 | `tx_count_6h` | Same, 6-hour window |
| 8 | `tx_sum_6h` | Same, 6-hour window |
| 9 | `tx_count_24h` | Same, 24-hour window |
| 10 | `tx_sum_24h` | Same, 24-hour window |
| 11 | `near_reporting_limit` | 1 if amount is within $1,000 of the $10,000 reporting limit |
| 12 | `below_reporting_limit` | 1 if amount is below $10,000 |
| 13 | `seconds_since_access` | Seconds since the most recent employee access to the sender account (0 if none) |
| 14 | `access_missing` | 1 if no prior employee access exists |
| 15 | `last_access_modify_phone` | 1 if the most recent access was MODIFY_PHONE |
| 16 | `last_access_override_alert` | 1 if the most recent access was OVERRIDE_ALERT |
| 17 | `is_dormant_sender` | 1 if the sender account status is DORMANT |
| 18 | `account_missing` | 1 if the sender account is not in the accounts table |

All windows use only data at or before the transaction's own timestamp, so no future information leaks into a score.

### 6.3 Training setup

| Setting | Value |
|---|---|
| Algorithm | `sklearn.ensemble.IsolationForest` |
| Trees (`n_estimators`) | 200 |
| `contamination` | 0.05 (tuned; the original 0.10 caused too many false positives) |
| Scaling | `StandardScaler` fitted on the training split only |
| Labels | Not used for fitting. Used only for the stratified split and for evaluation |
| Train / test split | 75% / 25%, stratified, `random_state=42` |
| Training time | about 1.5 s of CPU time |

### 6.4 Evaluation on held-out data (2,500 transactions, 75 suspicious)

| Metric | Value |
|---|---|
| Recall (fraud caught) | 1.0000 (100%) |
| Precision | 0.6250 |
| False positive rate | 0.0186 (1.9%) |
| PR-AUC (average precision) | 0.8008 |

Contamination sweep that was used to choose 0.05:

| contamination | Precision | Recall | FPR | Scenario A recall | Scenario B recall |
|---|---|---|---|---|---|
| 0.02 | 0.754 | 0.573 | 0.006 | 0.50 | 1.00 |
| 0.03 | 0.713 | 0.760 | 0.009 | 0.75 | 1.00 |
| 0.04 | 0.626 | 0.760 | 0.014 | 0.75 | 1.00 |
| **0.05 (chosen)** | **0.625** | **1.000** | **0.019** | **1.00** | **1.00** |
| 0.10 | 0.339 | 1.000 | 0.060 | 1.00 | 1.00 |

Most of the remaining false positives are Scenario C (legitimate but unusual payroll and large wires), which is expected. Graph evidence and the reviewer handle those cases.

Top anomaly-score correlations (descriptive, not causal): `is_dormant_sender`, `last_access_override_alert`, `tx_count_24h`, `near_reporting_limit`, `tx_count_6h`.

### 6.5 Performance

| Operation | Time |
|---|---|
| Feature extraction for one transaction | about 1.5 ms (target was under 5 ms) |
| Full score including per-feature explanation | about 13 ms |

### 6.6 ML API (port 8001)

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | none | status, `model_loaded`, CPU info |
| POST | `/score-transaction` | `transaction` plus optional `historical_transactions`, `access_logs`, `accounts` | `tx_id`, `anomaly_score` (0 to 1 percentile), `is_anomalous` (bool), `top_contributing_features` (up to 5) |
| POST | `/score-batch` | `transactions` (list) plus shared context lists | `results`: one score per transaction, same order |

- `anomaly_score` is the percentile of the transaction's raw anomaly score among the training scores. It is not a fraud probability.
- `is_anomalous` is the Isolation Forest decision using the 5% contamination threshold.
- `top_contributing_features` come from a counterfactual test: each scaled feature is reset to its training mean, and the features whose reset lowers the anomaly score the most are listed.

---

## 7. Orchestrator Service (`orchestrator-service/`)

### 7.1 Files

| File | Purpose |
|---|---|
| `config.py` | All settings, overridable with environment variables or `.env` |
| `db_loader.py` | Idempotent CSV to Neo4j loader (MERGE on primary keys; timestamps stored as Neo4j datetimes) |
| `graph_queries.py` | `ThreatDetectionEngine`: Cypher detectors, context fetches, reviewer list, freeze accounts |
| `alert_engine.py` | Groups detector hits into alerts, calls the ML engine, computes scores, builds investigation graph, timeline and LLM evidence brief |
| `llm_explainer.py` | `LlamaExplainer`: calls llama-server `/v1/chat/completions`, parses the JSON answer, provides the rule-based fallback |
| `gateway.py` | FastAPI app on port 8000, `/api/v1` routes, background LLM priority queue |
| `dossier_pdf.py` | Renders the case dossier as a PDF with fpdf2 |
| `test_gateway.py` | 11 pytest integration tests against the running stack |
| `.env.example` | Example overrides |

### 7.2 Graph detectors (Cypher)

| Detector | Logic | Default parameters | Result on current data |
|---|---|---|---|
| Insider action -> transfer | Employee performs OVERRIDE_ALERT, MODIFY_PHONE or MANUAL_UNFREEZE on an account, and that account sends money within the window | window 48 hours | 320 links (290 from EMP-A-001, 4 from EMP-B-001, the rest background noise) |
| Circular flow | Directed cycle of 2 to 4 transfers that returns to the starting account, in time order, within 7 days. Uses Neo4j quantified path patterns, and each cycle is counted once | max 4 hops, max 7 days | 7 cycles (all 4 Scenario B cycles plus 3 incidental background cycles). Query takes about 7 s |
| Structuring | Sender has at least 3 transfers between $9,000 and $10,000 inside a sliding 72-hour window | threshold $10,000, margin $1,000, window 72 h, min 3 | 560 transfers in the band; clusters matched mostly to Scenario A |

### 7.3 Alert grouping

| Alert ID format | Kind | Grouping |
|---|---|---|
| `ALT-INS-<emp_id>` | INSIDER | One alert per employee, covering all their linked accounts and transfers |
| `ALT-STR-<account_id>` | STRUCTURING | One per sender, only when the sender is not already inside an insider alert |
| `ALT-CYC-<smallest account_id>` | CYCLE | One per cycle, only when no account in it belongs to an insider alert |

Structuring clusters and cycles that touch an insider's accounts are merged into that insider's alert and add their pattern tags.

### 7.4 Scoring

Graph score = sum of the weights of the patterns present (capped at 1.0). Standalone structuring or cycle alerts get +0.10 base.

| Pattern tag | Weight |
|---|---|
| `INSIDER_OVERRIDE_ALERT` | 0.35 |
| `INSIDER_MANUAL_UNFREEZE` | 0.30 |
| `INSIDER_MODIFY_PHONE` | 0.25 |
| `CIRCULAR_FLOW` | 0.40 |
| `STRUCTURING` | 0.30 |
| `DORMANT_ACCOUNT_ACTIVITY` | 0.25 |
| `MULTI_ACCOUNT_INSIDER` (3 or more accounts) | 0.20 |
| `PRIVILEGED_EMPLOYEE` | 0.05 |

A single insider action followed by a transfer is common in normal work, so it scores low on its own. Corroborating patterns carry most of the weight.

ML score for an alert = `0.5 * (highest percentile among its transactions) + 0.5 * (share of its transactions the model flagged as anomalous)`.

Composite score (0 to 100) = `100 * (0.5 * graph_score + 0.5 * ml_score)`.

| Composite score | Risk level |
|---|---|
| 80 or more | CRITICAL |
| 65 to 79.9 | HIGH |
| 45 to 64.9 | MEDIUM |
| below 45 | LOW |

If the ML engine is down, the composite uses the graph score only and the gateway keeps working.

### 7.5 Current alert results (25 alerts)

| Rank | Alert | Risk | Composite | Graph | ML | Patterns |
|---|---|---|---|---|---|---|
| 1 | ALT-INS-EMP-A-001 (Jordan Lee, Scenario A) | CRITICAL | 99.8 | 1.00 | 1.00 | DORMANT_ACCOUNT_ACTIVITY, INSIDER_MODIFY_PHONE, INSIDER_OVERRIDE_ALERT, MULTI_ACCOUNT_INSIDER, PRIVILEGED_EMPLOYEE, STRUCTURING |
| 2 | ALT-INS-EMP-B-001 (Morgan Patel, Scenario B) | CRITICAL | 92.5 | 0.85 | 1.00 | CIRCULAR_FLOW, INSIDER_MODIFY_PHONE, MULTI_ACCOUNT_INSIDER |
| 3 | ALT-INS-EMP-N-029 (background employee) | HIGH | 67.1 | 0.35 | 0.99 | INSIDER_OVERRIDE_ALERT |
| 4 to 6 | Background employees | MEDIUM | 51.7 to 53.6 | | | |
| 7 to 25 | Background employees and incidental cycles | LOW | 17.6 to 43.2 | | | |

Totals: 2 CRITICAL, 1 HIGH, 3 MEDIUM, 19 LOW. Both planted fraud scenarios are the only CRITICAL alerts. The one HIGH background alert is there because the ML model flagged that transfer as genuinely unusual. Building the whole alert list takes about 12 to 13 seconds.

### 7.6 Investigation payload

When an alert is opened, the gateway builds:

- **Graph**: nodes (Employee, Account, Transaction) and edges (ACCESSED, SENT, TO) with `anomalous` flags and `risk_reasons`. Large cases are capped to the 15 highest-risk sender accounts (Scenario A shows 15 of 73) so the graph stays readable.
- **Timeline**: access events and transfers in time order, with categories ACCESS, TRANSFER or POLICY_BYPASS (OVERRIDE_ALERT and MANUAL_UNFREEZE count as POLICY_BYPASS).
- **Fraud window**: first and last evidence timestamps.
- **Evidence brief** for the LLM: alert, employee, accounts, findings, ML summary, breach tags, recommended action, timeline (at most 40 events are sent to the LLM).

Investigation bundles are cached in memory after the first build.

### 7.7 LLM integration

| Setting | Value |
|---|---|
| Endpoint | `http://localhost:8080/v1/chat/completions` (OpenAI-compatible) |
| Model name reported | `qwen3.5-9b-instruct` |
| Temperature / top_p | 0.1 / 0.9 |
| Max tokens | 1,200 |
| Output format | `response_format: json_object`; thinking disabled with `chat_template_kwargs.enable_thinking = false` |
| Output fields | `summary`, `breach_tags`, `rationale` (steps with title, detail, evidence_refs), `recommended_action`, `confidence` (0 to 1) |
| Request timeout | 180 s |
| Wait inside a UI request | 10 s (`LLM_INLINE_WAIT_SECONDS`); after that the rule-based explanation is returned and generation continues in the background |
| Dossier export wait | up to 75 s |
| Concurrency | One background worker with a priority queue (llama-server has one slot). The alert the reviewer opens jumps the queue; CRITICAL and HIGH alerts are pre-generated at startup |
| Fallback | Deterministic rule-based explanation built from detector findings and ML summary (`model: "rule-based"`) |
| Measured live generation | about 20 s for a new alert |

The system prompt tells the model to use only the given evidence, cite exact IDs, and treat the ML score as a percentile rather than a probability of guilt. Sample verified output for Scenario B: the model correctly named the four MODIFY_PHONE actions, the circular transfers totalling $2,635,000, and a real OVERRIDE_ALERT log (`LOG-N-00233`), with confidence 0.98.

### 7.8 Gateway REST API (port 8000, prefix `/api/v1`)

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/alerts` | `status`, `risk_level`, `limit` (default 100), `refresh` (true rebuilds) | List of alerts, highest composite first |
| GET | `/alerts/{alert_id}/investigation` | none | `alert`, `graph` {nodes, edges}, `explanation`, `timeline`, `associated_account_ids`, `fraud_window` |
| POST | `/cases/{case_id}/assign` | `{ "reviewer_id": "..." }` | `case_id`, `assigned_to`, `status` (ASSIGNED), `assigned_at` |
| POST | `/cases/{case_id}/freeze` | `{ "account_ids": [...] }` (must belong to the case) | `status` FROZEN, `affected_account_ids`, `message`. Also sets the accounts to FROZEN in Neo4j |
| POST | `/cases/{case_id}/escalate` | `{ "note": "..." }` | `status` ESCALATED, `message` |
| GET | `/cases/{case_id}/dossier` | `format=pdf` (default) or `json` | File download with a `Content-Disposition` header |
| GET | `/reviewers` | none | Compliance-department staff, excluding employees flagged in CRITICAL or HIGH alerts |
| GET | `/system/status` | none | Health and latency of the ML, LLM and graph nodes, plus metrics (open alerts, active cases, critical alerts, transactions scanned, flagged employees) |
| GET | `/health` (no prefix) | none | Gateway status and alert build info |

Case state (status, assignee, notes) is kept in memory and survives an alert refresh but not a gateway restart. Case status values are OPEN, ASSIGNED, ESCALATED, FROZEN and CLOSED. Errors use standard codes: 400 for accounts not linked to the case, 404 for an unknown alert, 422 for an invalid body, 503 when Neo4j is unavailable.

### 7.9 Configuration (`config.py`, overridable by environment or `.env`)

| Setting | Default |
|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `fincrime-dev-2026` (local development only) |
| `ML_SERVICE_URL` | `http://localhost:8001` |
| `LLM_BASE_URL` | `http://localhost:8080` |
| `LLM_MODEL` | `qwen3.5-9b-instruct` |
| `LLM_TIMEOUT_SECONDS` | 180 |
| `LLM_INLINE_WAIT_SECONDS` | 10 |
| `INSIDER_WINDOW_HOURS` | 48 |
| `REPORTING_THRESHOLD` / `STRUCTURING_MARGIN` | 10,000 / 1,000 |
| `STRUCTURING_WINDOW_HOURS` / `STRUCTURING_MIN_COUNT` | 72 / 3 |
| `CYCLE_MAX_HOPS` / `CYCLE_MAX_DAYS` | 4 / 7 |
| `MAX_GRAPH_ACCOUNTS` | 15 |
| `API_HOST` / `API_PORT` | `0.0.0.0` / 8000 |
| `CORS_ORIGINS` | `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000` |

---

## 8. Frontend Dashboard (`frontend-dashboard/`)

| Item | Value |
|---|---|
| Name shown | Drake&DonT (header and browser tab title) |
| Theme | Light (light gray page, white panels, dark text, darker accent shades) |
| Framework | React 18 + Vite 5, Tailwind CSS 3, Cytoscape.js, Lucide icons, Axios |
| Backend | Talks only to the Gateway at `VITE_ORCHESTRATOR_URL` (default `http://localhost:8000`) with prefix `/api/v1` |
| Mock mode | `VITE_USE_MOCKS=true` serves built-in demo data without a backend (default is false) |

### 8.1 Layout and components

| File | Role |
|---|---|
| `src/App.jsx` | Shell. Top bar with name, alert queue button and node health (Orchestrator, ML, LLM, Graph DB). Left 60%: graph above timeline. Right 40%: evidence panel. Holds app state and case actions |
| `src/components/GraphCanvas.jsx` | Interactive Cytoscape graph. Employee = blue hexagon, Account = purple circle (dashed when dormant), Transaction = green pill, anomalous = red outline. Override edges are red. Force and hierarchy layouts, zoom controls |
| `src/components/TimelineStream.jsx` | Chronological events with filters (All, Access, Transfers, Policy bypass) and highlighting of the critical fraud window |
| `src/components/EvidencePanel.jsx` | Composite risk bar, two gauges (Graph Pattern Match, ML Anomaly Score), pattern tags, LLM summary, breach tags, step-by-step audit rationale with evidence references, and the buttons Freeze Associated Accounts, Escalate to Compliance Lead, Export Evidence Packet |
| `src/components/CaseManager.jsx` | Alert Queue drawer: search, risk filter (ALL, CRITICAL, HIGH, MEDIUM, LOW), select a case, assign a reviewer |
| `src/api/client.js` | Axios client with retries, typed errors and runtime parsers that validate every response against the API contract |
| `src/api/mockData.js` | Demo data for mock mode |
| `src/utils/format.js` | Formatting helpers |
| `tailwind.config.js` | Light theme. The components were written with dark-theme class names, so the config remaps the slate scale to light values and mirrors accent colors around shade 500 |

---

## 9. Directory structure

```text
TRIAL_DND/
├── PROJECT_OVERVIEW.md          (this file)
├── RUNNING.md                   (start-up guide)
├── data-engine/
│   ├── generator.py
│   ├── scenarios.py
│   ├── schemas.py
│   ├── requirements.txt
│   └── data/
│       ├── employees.csv
│       ├── accounts.csv
│       ├── transactions.csv
│       ├── access_logs.csv
│       └── ground_truth_labels.csv
├── ml-engine/
│   ├── feature_extraction.py
│   ├── train.py
│   ├── service.py
│   ├── test_service.py
│   ├── requirements.txt
│   ├── .venv/
│   └── artifacts/model.pkl
├── orchestrator-service/
│   ├── config.py
│   ├── db_loader.py
│   ├── graph_queries.py
│   ├── alert_engine.py
│   ├── llm_explainer.py
│   ├── gateway.py
│   ├── dossier_pdf.py
│   ├── test_gateway.py
│   ├── requirements.txt
│   ├── .env.example
│   └── .venv/
└── frontend-dashboard/
    ├── index.html
    ├── package.json
    ├── tailwind.config.js
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        ├── api/client.js
        ├── api/mockData.js
        ├── utils/format.js
        └── components/
            ├── GraphCanvas.jsx
            ├── TimelineStream.jsx
            ├── EvidencePanel.jsx
            └── CaseManager.jsx
```

The llama.cpp build and the model live outside the project, at `~/Ddrive/llamacpp_root/llama.cpp/` (model file `models/qwen3.5-9b-q4_k_m.gguf`).

---

## 10. One-time setup

```bash
# Neo4j (already done on this Mac)
brew install neo4j
neo4j-admin dbms set-initial-password fincrime-dev-2026
neo4j start

# Python environments
cd ml-engine && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
cd orchestrator-service && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..

# Frontend dependencies
cd frontend-dashboard && npm install && cd ..

# Train the ML model (writes ml-engine/artifacts/model.pkl)
cd ml-engine && .venv/bin/python train.py && cd ..

# Load the graph into Neo4j (safe to re-run)
cd orchestrator-service && .venv/bin/python db_loader.py && cd ..
```

---

## 11. Running the system

Start in this order, each long-running service in its own terminal.

```bash
# 1. Neo4j (runs in the background)
neo4j start

# 2. Local LLM on port 8080
cd ~/Ddrive/llamacpp_root/llama.cpp && ./build/bin/llama-server -m models/qwen3.5-9b-q4_k_m.gguf --alias qwen3.5-9b-instruct -ngl 99 -fa on -c 32768 -ctk turbo3 -ctv turbo3 -t 10 -np 1 --reasoning off --host 127.0.0.1 --port 8080

# 3. ML engine on port 8001
cd ~/Ddrive/pccoe/TRIAL_DND/ml-engine && .venv/bin/python service.py

# 4. Gateway on port 8000 (start after 2 and 3; builds alerts in about 13 s)
cd ~/Ddrive/pccoe/TRIAL_DND/orchestrator-service && .venv/bin/python gateway.py

# 5. Dashboard on port 5173
cd ~/Ddrive/pccoe/TRIAL_DND/frontend-dashboard && npm run dev
```

llama-server flags explained:

| Flag | Meaning |
|---|---|
| `-ngl 99` | Put all model layers on the GPU (Metal) |
| `-fa on` | Flash attention on |
| `-c 32768` | 32k-token context |
| `-ctk turbo3 -ctv turbo3` | TurboQuant compression of the key/value cache |
| `-t 10` | 10 CPU threads (the M4 Pro has 10 performance cores) |
| `-np 1` | One parallel slot, so one request gets the full context |
| `--reasoning off` | No hidden thinking step, so answers come faster |
| `--alias` | Model name returned by the API |

Health check (all three nodes should print True):

```bash
curl -s localhost:8000/api/v1/system/status | python3 -c "import json,sys; [print(k, v['online']) for k,v in json.load(sys.stdin)['nodes'].items()]"
```

Then open `http://localhost:5173`.

---

## 12. Tests

| Suite | Command | Needs | Current result |
|---|---|---|---|
| ML engine | `cd ml-engine && .venv/bin/python -m pytest -q` | `model.pkl` | 8 passed |
| Gateway integration | `cd orchestrator-service && .venv/bin/python -m pytest -q test_gateway.py` | Neo4j, ML engine and gateway running (LLM optional) | 11 passed, repeatable |

The gateway tests check that every response matches the frontend contract, that both planted scenarios are the only CRITICAL alerts, graph capping, 404 and 422 handling, reviewer exclusion of flagged insiders, the case workflow, PDF and JSON dossier export, and system status.

---

## 13. Known limitations

| Limitation | Detail |
|---|---|
| In-memory case state | Assignments, escalations and notes are lost when the gateway restarts. Frozen account status does persist in Neo4j |
| Future timestamps | The synthetic data covers all of 2026, so some events are after today's date and the UI shows times like "-8d ago". This should be fixed in the data generator |
| Background noise alerts | 23 of the 25 alerts are background activity: 21 employees whose ordinary actions happened to precede a transfer, and 2 incidental money cycles. They are ranked LOW or MEDIUM, except one HIGH where the ML model flagged the transfer |
| LLM latency | A new narrative takes about 20 s. The UI shows the rule-based explanation first, and reopening the alert shows the LLM version |
| No authentication | The gateway has no login. The frontend supports an optional bearer token (`VITE_API_TOKEN`) but the gateway does not check it yet |
| Dev password | The Neo4j password in `config.py` is for local development only |
| Single machine | The Tailscale two-node deployment is planned but not used in Submission 1 |
