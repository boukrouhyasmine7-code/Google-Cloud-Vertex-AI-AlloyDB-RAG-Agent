"""
main.py
=======
FastAPI Application — Vertex AI Agent Platform REST Endpoint

Exposes the Gemini Pro + AlloyDB RAG agent as a production-ready
HTTP API compatible with Cloud Run, GKE, or App Engine deployment.

GCP Deployment Mapping
-----------------------
Local                         → GCP
─────────────────────────────────────────────────────────────
uvicorn main:app              → Cloud Run container (auto-scaled)
/chat endpoint                → Vertex AI Agent Builder webhook
session management            → Vertex AI Session Service
Structured JSON logs          → Cloud Logging (automatic on GCR)

Endpoints
---------
POST /chat          → Single-turn or multi-turn agent conversation
GET  /sessions/{id} → Retrieve session metadata and history
DELETE /sessions/{id}→ Clear a session
GET  /health        → Liveness probe (Cloud Run / GKE)
GET  /ready         → Readiness probe (checks AlloyDB connectivity)
GET  /              → API info
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent_platform import GeminiAgent, get_or_create_session
from app.config import Settings, get_settings
from app.database import close_pool, get_db_connection, release_db_connection

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models — request / response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's natural language message to the Vertex AI agent.",
        examples=["Find me flights from SFO to JFK under $400", "What are the best lounges at LAX?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversation continuity. "
                    "Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    session_id: str = Field(description="Session ID for follow-up messages.")
    response: str = Field(description="Gemini Pro's natural language response.")
    tool_calls_made: int = Field(description="Number of AlloyDB tool calls executed in this turn.")
    model: str = Field(description="Gemini model used for this response.")
    latency_ms: float = Field(description="End-to-end response latency in milliseconds.")


class SessionResponse(BaseModel):
    """Response body for GET /sessions/{session_id}."""
    session_id: str
    messages_exchanged: int
    tool_calls_made: int
    history_length: int


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str
    service: str
    version: str = "1.0.0"


class ReadinessResponse(BaseModel):
    """Readiness probe — checks AlloyDB connectivity."""
    status: str
    alloydb: str
    gemini_model: str
    embedding_model: str


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------

_agent: GeminiAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise shared resources on startup, clean up on shutdown.
    In Cloud Run: this runs once per container instance.
    """
    global _agent
    settings = get_settings()

    log.info(
        "Vertex AI Agent Platform starting",
        model=settings.gemini_model,
        alloydb_host=settings.alloydb_host,
        alloydb_database=settings.alloydb_database,
    )

    _agent = GeminiAgent(settings)
    log.info("GeminiAgent initialized and ready")

    yield  # Application is running

    log.info("Shutting down — closing AlloyDB connection pool")
    close_pool()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vertex AI Agent Platform — AlloyDB RAG Chat",
    description=(
        "Production-ready REST API for the Gemini Pro + AlloyDB RAG agent. "
        "Built on Google Cloud's Vertex AI Agent Platform architecture with "
        "AlloyDB for HTAP workloads and pgvector for semantic search."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — update origins for production Cloud Run deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency: settings
# ---------------------------------------------------------------------------

def get_app_settings() -> Settings:
    return get_settings()


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    log.info(
        "HTTP request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(latency, 2),
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=dict, tags=["Info"])
async def root(settings: Settings = Depends(get_app_settings)) -> dict[str, Any]:
    """API information and GCP architecture summary."""
    return {
        "service": "Vertex AI Agent Platform — AlloyDB RAG Chat",
        "version": "1.0.0",
        "architecture": {
            "llm": f"Google Gemini Pro ({settings.gemini_model})",
            "embedding_model": settings.embedding_model,
            "vector_store": "AlloyDB + pgvector (VECTOR(768))",
            "vector_search": "Cosine Similarity / HNSW ANN Index",
            "tool_layer": "MCP Toolbox (function calling)",
            "framework": "FastAPI on Cloud Run",
        },
        "endpoints": {
            "chat": "POST /chat",
            "session": "GET /sessions/{session_id}",
            "health": "GET /health",
            "readiness": "GET /ready",
            "docs": "GET /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health() -> HealthResponse:
    """
    Liveness probe.
    Cloud Run / GKE calls this to determine if the container is alive.
    """
    return HealthResponse(status="ok", service="vertex-ai-agent-platform")


@app.get("/ready", response_model=ReadinessResponse, tags=["Observability"])
async def readiness(settings: Settings = Depends(get_app_settings)) -> ReadinessResponse:
    """
    Readiness probe — verifies AlloyDB connectivity.
    Cloud Run holds traffic until this returns 200.
    """
    try:
        conn = get_db_connection(settings)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            alloydb_status = "connected"
        finally:
            release_db_connection(conn)
    except Exception as exc:
        log.error("AlloyDB readiness check failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"AlloyDB not ready: {exc}")

    return ReadinessResponse(
        status="ready",
        alloydb=alloydb_status,
        gemini_model=settings.gemini_model,
        embedding_model=settings.embedding_model,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_app_settings),
) -> ChatResponse:
    """
    Send a message to the Vertex AI Gemini Pro agent.

    The agent autonomously decides which AlloyDB tools to invoke
    (flight search, semantic amenity search, airport info) and
    returns a synthesised natural language response.

    Supports multi-turn conversations via `session_id`.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    start = time.perf_counter()

    session = get_or_create_session(request.session_id)
    tool_calls_before = session.tool_calls_made

    response_text = _agent.chat(request.message, session)

    latency_ms = (time.perf_counter() - start) * 1000
    tool_calls_this_turn = session.tool_calls_made - tool_calls_before

    log.info(
        "Chat turn completed",
        session_id=session.session_id,
        tool_calls=tool_calls_this_turn,
        latency_ms=round(latency_ms, 2),
    )

    return ChatResponse(
        session_id=session.session_id,
        response=response_text,
        tool_calls_made=tool_calls_this_turn,
        model=settings.gemini_model,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def get_session(session_id: str) -> SessionResponse:
    """
    Retrieve metadata for an existing conversation session.
    In production, this queries the Vertex AI Session Service.
    """
    from app.agent_platform import _SESSIONS
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return SessionResponse(
        session_id=session.session_id,
        messages_exchanged=session.messages_exchanged,
        tool_calls_made=session.tool_calls_made,
        history_length=len(session.history),
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str) -> dict[str, str]:
    """Clear a conversation session (start fresh)."""
    from app.agent_platform import _SESSIONS
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    del _SESSIONS[session_id]
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level,
    )
