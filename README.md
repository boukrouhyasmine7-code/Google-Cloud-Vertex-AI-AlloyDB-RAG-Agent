# Vertex AI Agent Platform + AlloyDB RAG Chat

<p align="center">
  <img src="https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" />
  <img src="https://img.shields.io/badge/AlloyDB-PostgreSQL_Compatible-0F9D58?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_Pro-1.5-DB4437?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/pgvector-VECTOR(768)-F4B400?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Cloud_Run-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
</p>

> **Production-ready Python implementation** of Google Cloud's reference architecture:  
> *"Build an LLM and RAG-based Chat Application with AlloyDB and Agent Platform"*  
> — fully executable locally using a 1:1 PostgreSQL + pgvector replica of AlloyDB.

---

## Table of Contents

- [Architecture: Vertex AI Agent Platform + AlloyDB](#architecture-vertex-ai-agent-platform--alloydb)
- [System Diagram](#system-diagram)
- [GCP Component Mapping](#gcp-component-mapping)
- [AlloyDB: HTAP + Vector Search](#alloydb-htap--vector-search)
- [Gemini Pro: Tool-Calling & MCP Toolbox](#gemini-pro-tool-calling--mcp-toolbox)
- [RAG Pipeline: Vertex AI Embeddings + Cosine Similarity](#rag-pipeline-vertex-ai-embeddings--cosine-similarity)
- [Repository Structure](#repository-structure)
- [Quick Start (Local Replica)](#quick-start-local-replica)
- [API Reference](#api-reference)
- [Deploying to GCP](#deploying-to-gcp)
- [Configuration Reference](#configuration-reference)

---

## Architecture: Vertex AI Agent Platform + AlloyDB

This project implements the full stack described in Google Cloud's reference lab, combining three flagship GCP AI services into a single, coherent agentic system:

| Layer | GCP Service | Role |
|---|---|---|
| **LLM & Reasoning** | Vertex AI — Gemini 1.5 Pro | Multi-turn reasoning, tool selection, response synthesis |
| **Tool Orchestration** | Vertex AI MCP Toolbox | Registers and executes Python functions as Gemini tools |
| **Vector Store + OLTP** | AlloyDB for PostgreSQL | Unified HTAP database: transactional flights + vector amenities |
| **Embedding Model** | Vertex AI text-embedding-004 | 768-dimension semantic embeddings for RAG retrieval |
| **API Layer** | Cloud Run (FastAPI) | Auto-scaled, serverless REST endpoint for the agent |

The architecture implements **Retrieval-Augmented Generation (RAG)** at the database layer: instead of a separate vector database, AlloyDB's integrated pgvector extension stores and retrieves Vertex AI embeddings from the same instance that handles transactional flight queries — a defining capability of AlloyDB's HTAP design.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Google Cloud Platform                                │
│                                                                             │
│  ┌──────────────┐   POST /chat    ┌────────────────────────────────────┐   │
│  │   Client     │ ──────────────► │     Cloud Run (FastAPI)            │   │
│  │  (Browser /  │                 │     main.py                        │   │
│  │   cURL /     │ ◄────────────── │     /chat  /health  /ready         │   │
│  │   App)       │  JSON response  └──────────────┬─────────────────────┘   │
│  └──────────────┘                                │                         │
│                                                  │ invoke                  │
│                                    ┌─────────────▼─────────────────────┐   │
│                                    │   Vertex AI Agent Platform        │   │
│                                    │   agent_platform.py               │   │
│                                    │                                   │   │
│                                    │  ┌──────────────────────────┐    │   │
│                                    │  │   Gemini 1.5 Pro          │    │   │
│                                    │  │   System instruction      │    │   │
│                                    │  │   Session history         │    │   │
│                                    │  │   Tool declarations       │    │   │
│                                    │  └──────────┬───────────────┘    │   │
│                                    │             │ FunctionCall        │   │
│                                    │  ┌──────────▼───────────────┐    │   │
│                                    │  │   MCP Toolbox             │    │   │
│                                    │  │   gcp_toolbox.py          │    │   │
│                                    │  │                           │    │   │
│                                    │  │  search_flights()         │    │   │
│                                    │  │  search_amenities()  ──┐  │    │   │
│                                    │  │  get_airport_info()    │  │    │   │
│                                    │  │  list_routes()         │  │    │   │
│                                    │  └──────────┬─────────────┘  │   │   │
│                                    └─────────────│────────────────│───┘   │
│                                                  │ SQL            │ embed  │
│                                    ┌─────────────▼────────────────▼─────┐  │
│                                    │         AlloyDB Cluster            │  │
│                                    │   (PostgreSQL 16 + pgvector)       │  │
│                                    │                                    │  │
│                                    │  ┌────────────┐ ┌───────────────┐  │  │
│                                    │  │  airports  │ │   flights     │  │  │
│                                    │  │  (OLTP)    │ │   (OLTP)      │  │  │
│                                    │  └────────────┘ └───────────────┘  │  │
│                                    │  ┌─────────────────────────────┐   │  │
│                                    │  │  amenities                  │   │  │
│                                    │  │  embedding VECTOR(768)      │   │  │
│                                    │  │  [HNSW / ScaNN ANN index]   │   │  │
│                                    │  └─────────────────────────────┘   │  │
│                                    └────────────────────────────────────┘  │
│                                                                             │
│                          ┌─────────────────────────┐                       │
│                          │  Vertex AI Embeddings    │                       │
│                          │  text-embedding-004      │                       │
│                          │  output: VECTOR(768)     │                       │
│                          └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## GCP Component Mapping

Every local component in this repository maps 1:1 to a GCP service. The code is structured so that migrating from local to full GCP requires only configuration changes — no code rewrites.

| Local Component | GCP Production Equivalent | Notes |
|---|---|---|
| `pgvector/pgvector:pg16` Docker image | **AlloyDB** Primary Instance | Wire-protocol identical; all SQL runs unchanged |
| `psycopg2` connection pool | **AlloyDB Connector** (Python) | Swap to `google-cloud-alloydb-connector` for IAM auth + mTLS |
| `HNSW` index on `embedding` | AlloyDB **ScaNN ANN Index** | `CREATE INDEX USING scann` in production for billion-scale ANN |
| `google-generativeai` SDK | **Vertex AI Gemini API** | Same SDK; set `vertexai=True` and project for Vertex backend |
| Python `function_declarations` | **MCP Toolbox** tool registry | In production: tools registered in `toolbox-config.yaml` |
| In-memory `_SESSIONS` dict | **Vertex AI Session Service** | Backed by Cloud Spanner; persistent across container restarts |
| FastAPI on `uvicorn` | **Cloud Run** (containerized) | `gcloud run deploy` with Dockerfile |
| `structlog` JSON logs | **Cloud Logging** | GCR auto-ingests structured JSON; no code change needed |
| `.env` file | **Secret Manager** + env vars | Inject secrets via `--set-secrets` on Cloud Run deploy |

---

## AlloyDB: HTAP + Vector Search

**AlloyDB** is Google Cloud's fully-managed, PostgreSQL-compatible database engine, engineered for **HTAP (Hybrid Transactional/Analytical Processing)** — the ability to serve both operational transactions and complex analytical queries from the same cluster, at enterprise scale.

### Why AlloyDB for RAG?

Traditional RAG pipelines separate the operational database (for user/booking data) from a dedicated vector database (for semantic search). AlloyDB eliminates this split. In this architecture:

- **OLTP workload** (transactional): `airports` and `flights` tables store operational data. Flight availability queries are millisecond OLTP lookups, isolated from analytical load by AlloyDB's disaggregated storage engine.

- **Analytical / Vector workload**: `amenities.embedding VECTOR(768)` stores Vertex AI embeddings. Cosine similarity queries (`<=>`) execute as ANN searches using AlloyDB's integrated **ScaNN** (Scalable Nearest Neighbor) index, delivering sub-10ms retrieval at billions of rows.

```sql
-- Hybrid HTAP query: OLTP JOIN + vector search in a single statement
-- This runs identically on local pgvector and AlloyDB production.
SELECT
    a.name,
    ap.city,
    a.category,
    1 - (a.embedding <=> $1::vector)  AS similarity_score
FROM amenities a
JOIN airports ap ON ap.iata_code = a.airport_iata  -- OLTP join
WHERE a.airport_iata = 'SFO'                        -- OLTP filter
ORDER BY a.embedding <=> $1::vector ASC             -- Vector ANN sort
LIMIT 5;
```

### AlloyDB ScaNN Index (Production)

In production AlloyDB, replace the `HNSW` index in `schema.sql` with the native ScaNN index for orders-of-magnitude better performance at scale:

```sql
-- Production AlloyDB ScaNN ANN index (replaces HNSW for pgvector)
CREATE INDEX amenities_embedding_scann
ON amenities USING scann (embedding cosine)
WITH (num_leaves = 500);
```

The ScaNN index is built into the AlloyDB columnar engine, meaning vector queries are accelerated by AlloyDB's separate analytical processing layer without impacting transactional throughput on the primary instance.

---

## Gemini Pro: Tool-Calling & MCP Toolbox

### The Agentic Loop

The **Vertex AI Agent Platform** orchestrates a multi-step reasoning loop where Gemini Pro autonomously selects and executes tools against AlloyDB:

```
User: "Find me a quiet lounge near Gate B at SFO with food"

  ┌── Gemini reasons: this requires amenity search ──┐
  │                                                   │
  │  FunctionCall: search_airport_amenities(          │
  │    query="quiet lounge with food near Gate B",    │
  │    airport_iata="SFO",                            │
  │    category="lounge"                              │
  │  )                                                │
  │                                                   │
  │  [AlloyDB cosine similarity query executes]       │
  │                                                   │
  │  FunctionResponse: {amenities: [...top 3...]}     │
  │                                                   │
  └── Gemini synthesises: "I found the United Club   │
      Lounge in Terminal 3..." ◄─────────────────────┘
```

### MCP Toolbox Pattern

The `gcp_toolbox.py` module implements the **Model Context Protocol (MCP) Toolbox** pattern. Each Python function is annotated with type hints and a structured docstring that Gemini uses to:

1. **Understand** what the tool does and when to call it
2. **Validate** its arguments against the JSON Schema declaration
3. **Parse** the structured result into its reasoning context

```python
# gcp_toolbox.py — Gemini sees this function signature as a tool schema
def search_airport_amenities(
    query: str,           # Natural language; Gemini writes this from the conversation
    airport_iata: str | None = None,   # Extracted from context ("at SFO" → "SFO")
    category: str | None = None,       # Inferred from intent ("lounge" → "lounge")
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Perform semantic vector search over airport amenities using
    Vertex AI embeddings and cosine similarity on AlloyDB pgvector.
    ...
    """
```

In production **Vertex AI MCP Toolbox**, tools are registered via `toolbox-config.yaml` and served by the Toolbox binary, which handles authentication, rate limiting, and schema validation automatically. This code structure mirrors that pattern exactly.

---

## RAG Pipeline: Vertex AI Embeddings + Cosine Similarity

### Ingestion (seed_and_embed.py)

```
Amenity description text
        │
        ▼ text-embedding-004 (RETRIEVAL_DOCUMENT)
  VECTOR(768) embedding
        │
        ▼ INSERT INTO amenities(embedding = ...)
  AlloyDB pgvector column
        │
        ▼ HNSW / ScaNN index auto-updated
  Ready for ANN retrieval
```

### Retrieval (gcp_toolbox.py)

```
User query: "quiet place to work with power outlets"
        │
        ▼ text-embedding-004 (RETRIEVAL_QUERY)
  query_embedding VECTOR(768)
        │
        ▼ SELECT ... ORDER BY embedding <=> query_embedding ASC
  Top-K semantically similar amenities from AlloyDB
        │
        ▼ Cosine similarity score filter (≥ 0.70 threshold)
  Filtered, ranked results → Gemini context → response
```

The asymmetric retrieval design (separate `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types) is a Vertex AI best practice that measurably improves retrieval precision by encoding documents and queries in task-optimised embedding spaces.

---

## Repository Structure

```
alloydb-agent-platform/
│
├── main.py                     # FastAPI app — Cloud Run entrypoint
├── requirements.txt            # GCP AI stack dependencies
├── docker-compose.yml          # AlloyDB local replica (PostgreSQL + pgvector)
├── .env.example                # Environment variable template
│
├── app/
│   ├── __init__.py
│   ├── config.py               # Pydantic settings (maps to Secret Manager in prod)
│   ├── database.py             # AlloyDB connection pool (psycopg2)
│   ├── gcp_toolbox.py          # MCP Toolbox: Gemini tool functions → AlloyDB SQL
│   └── agent_platform.py      # Gemini Pro orchestration + agentic loop
│
├── sql/
│   └── schema.sql              # AlloyDB schema: airports, flights, amenities + HNSW
│
├── scripts/
│   └── seed_and_embed.py      # Data seeding + Vertex AI embedding pipeline
│
└── tests/
    └── test_agent_api.py       # FastAPI endpoint + toolbox unit tests
```

---

## Quick Start (Local Replica)

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- A Google Cloud project with Vertex AI API enabled (for real embeddings/Gemini)  
  *OR* a `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey) for local dev

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/alloydb-agent-platform.git
cd alloydb-agent-platform

cp .env.example .env
# Edit .env: set GOOGLE_CLOUD_PROJECT and GOOGLE_API_KEY (or configure ADC)
```

### 2. Authenticate with Google Cloud

```bash
# Option A: Application Default Credentials (recommended for GCP deployment)
gcloud auth application-default login

# Option B: Google AI Studio API key (fastest for local dev)
# Set GOOGLE_API_KEY=your-key in .env
```

### 3. Start AlloyDB Local Replica

```bash
# Spins up PostgreSQL 16 + pgvector, applies schema.sql automatically
docker compose up -d

# Verify AlloyDB is ready
docker compose ps
# Expected: alloydb-local-replica  Up (healthy)

# Optional: Open AlloyDB Studio (pgAdmin) at http://localhost:5050
```

### 4. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 5. Seed Data + Generate Vertex AI Embeddings

```bash
python scripts/seed_and_embed.py

# Output:
# ✓ Seeded 8 flights and 15 amenities
# Generating Vertex AI embeddings for 15 amenities...
#   Model: text-embedding-004  Dimensions: 768
# [████████████████████████] 15/15  Done
# ✓ AlloyDB local replica seeded and embedded successfully.
```

### 6. Start the Agent Platform API

```bash
python main.py
# Listening on http://0.0.0.0:8080
# Docs: http://localhost:8080/docs
```

### 7. Chat with the Agent

```bash
# Search for flights
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find me flights from San Francisco to New York under $400"}'

# Semantic amenity search (RAG)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a good place to grab breakfast early in the morning at LAX?"}'

# Multi-turn conversation (pass the session_id from the first response)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Are there any lounges nearby?", "session_id": "YOUR_SESSION_ID"}'
```

### 8. Run Tests

```bash
pytest tests/ -v
```

---

## API Reference

### `POST /chat`

Send a message to the Gemini Pro agent. The agent autonomously executes AlloyDB tool calls.

**Request body:**
```json
{
  "message": "Find flights from SFO to JFK this week under $450",
  "session_id": null
}
```

**Response:**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "response": "I found 2 flights from San Francisco (SFO) to New York JFK:\n\n✈ GA201 · Google Air\n  Departure: 09:15 AM UTC · Arrival: 2:45 PM UTC\n  Duration: 5h 30m · Price: $389.00 · 8 seats left\n  Aircraft: Boeing 757-200\n\nWould you like me to check amenities at SFO or JFK?",
  "tool_calls_made": 1,
  "model": "gemini-2.5-flash",
  "latency_ms": 1243.7
}
```

### `GET /health`

Liveness probe for Cloud Run / GKE.

```json
{"status": "ok", "service": "vertex-ai-agent-platform", "version": "1.0.0"}
```

### `GET /ready`

Readiness probe — verifies AlloyDB connectivity before accepting traffic.

```json
{
  "status": "ready",
  "alloydb": "connected",
  "gemini_model": "gemini-2.5-flash",
  "embedding_model": "text-embedding-004"
}
```

### `GET /sessions/{session_id}`

Retrieve session metadata.

### `DELETE /sessions/{session_id}`

Clear a session to start a fresh conversation.

### Interactive Docs

Visit `http://localhost:8080/docs` for the full OpenAPI 3.1 specification with a live try-it-out interface.

---

## Deploying to GCP

### 1. Containerize

```bash
# Build and push to Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/vertex-rag/agent-platform:latest .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/vertex-rag/agent-platform:latest
```

### 2. Deploy to Cloud Run

```bash
gcloud run deploy vertex-rag-agent \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT/vertex-rag/agent-platform:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT \
  --set-env-vars GEMINI_MODEL=gemini-2.5-flash \
  --set-env-vars VERTEX_AI_LOCATION=us-central1 \
  --set-secrets ALLOYDB_PASSWORD=alloydb-password:latest \
  --service-account vertex-rag-sa@YOUR_PROJECT.iam.gserviceaccount.com
```

### 3. Connect to AlloyDB (Production)

Replace the `psycopg2` connection with the [AlloyDB Python Connector](https://cloud.google.com/alloydb/docs/connect-connector) for IAM-based authentication and automatic mTLS:

```python
from google.cloud.alloydb.connector import Connector

connector = Connector()
conn = connector.connect(
    "projects/YOUR_PROJECT/locations/us-central1/clusters/my-cluster/instances/my-instance",
    "pg8000",
    user="alloydb-sa@YOUR_PROJECT.iam",
    enable_iam_auth=True,
    db="vertex_rag_db",
)
```

### 4. Upgrade to AlloyDB ScaNN Index

```sql
-- In production AlloyDB (replace HNSW with native ScaNN for 100x+ throughput)
DROP INDEX IF EXISTS idx_amenities_embedding_hnsw;

CREATE INDEX amenities_embedding_scann
ON amenities USING scann (embedding cosine)
WITH (num_leaves = 500, quantizer = 'sq8');
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | `local-dev-project` | GCP project for Vertex AI API calls |
| `VERTEX_AI_LOCATION` | `us-central1` | Vertex AI / AlloyDB region |
| `GEMINI_MODEL` | `gemini-2.5-flash` | `gemini-2.5-flash` or `gemini-1.5-flash` |
| `EMBEDDING_MODEL` | `text-embedding-004` | Vertex AI embedding model |
| `ALLOYDB_HOST` | `localhost` | AlloyDB instance IP or hostname |
| `ALLOYDB_PORT` | `5432` | PostgreSQL port |
| `ALLOYDB_DATABASE` | `vertex_rag_db` | Target database name |
| `ALLOYDB_USER` | `alloydb_admin` | Database user |
| `ALLOYDB_PASSWORD` | *(required)* | Database password (use Secret Manager in prod) |
| `ALLOYDB_POOL_SIZE` | `10` | Connection pool size |
| `API_PORT` | `8080` | FastAPI listen port |
| `VECTOR_SEARCH_TOP_K` | `5` | Number of ANN results to return |
| `VECTOR_SIMILARITY_THRESHOLD` | `0.70` | Minimum cosine similarity to include in results |

---

## Reference Architecture

This implementation is based on the Google Cloud reference lab:

> **"Build an LLM and RAG-based Chat Application using AlloyDB and Vertex AI Agent Platform"**  
> Google Cloud Skills Boost — Generative AI Learning Path

**Key GCP documentation:**
- [AlloyDB for PostgreSQL](https://cloud.google.com/alloydb/docs)
- [AlloyDB pgvector integration](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings)
- [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/introduction)
- [Vertex AI MCP Toolbox](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/mcp-toolbox)
- [Gemini Function Calling](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)
- [Vertex AI text-embedding-004](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)

---

<p align="center">
  Built on Google Cloud · Vertex AI · AlloyDB · Gemini Pro
</p>
