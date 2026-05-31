"""
app/gcp_toolbox.py
==================
MCP Toolbox Layer — Gemini Function Calling Definitions

This module implements the Model Context Protocol (MCP) Toolbox pattern
for Google's Agent Platform. Each function in this module is registered
as a Gemini Tool that the Gemini Pro agent can autonomously invoke
during multi-turn conversations.

GCP Architecture Mapping
-------------------------
Local component              → GCP equivalent
─────────────────────────────────────────────────────────────
Python functions below       → MCP Toolbox tool definitions
Gemini function_declarations → Vertex AI Tool schema (JSON)
psycopg2 queries             → AlloyDB SQL over wire protocol
Cosine similarity (<=>)      → AlloyDB ScaNN ANN vector search
Vertex AI Embeddings API     → text-embedding-004 endpoint

Tool Invocation Flow:
  User query
    → Gemini Pro (reasoning)
      → select_tool()
        → gcp_toolbox function
          → AlloyDB SQL query
            → structured result → Gemini → natural language response

Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/agent-platform/mcp-toolbox
"""

from __future__ import annotations

import structlog
from typing import Any
from datetime import datetime

from app.config import get_settings
from app.database import ManagedConnection

log = structlog.get_logger(__name__)


# ============================================================
# Tool 1: search_flights
# ============================================================

def search_flights(
    departure_airport: str,
    arrival_airport: str,
    date: str | None = None,
    max_price: float | None = None,
) -> dict[str, Any]:
    """
    Search for available flights between two airports in the AlloyDB flights table.

    This tool queries the operational flight schedule stored in AlloyDB,
    which represents the OLTP (transactional) side of AlloyDB's HTAP
    (Hybrid Transactional/Analytical Processing) architecture.

    Args:
        departure_airport: IATA code of the departure airport (e.g. 'SFO', 'LAX', 'JFK').
        arrival_airport: IATA code of the destination airport (e.g. 'LAX', 'ORD', 'ATL').
        date: Optional departure date in YYYY-MM-DD format to filter results.
        max_price: Optional maximum ticket price in USD to filter affordable flights.

    Returns:
        A dict containing a list of matching flights with airline, departure/arrival
        times, duration, price, seats available, and aircraft type. Returns an
        empty list if no flights are found for the given criteria.
    """
    log.info(
        "Tool invoked: search_flights",
        departure=departure_airport,
        arrival=arrival_airport,
        date=date,
        max_price=max_price,
    )

    departure_airport = departure_airport.upper().strip()
    arrival_airport = arrival_airport.upper().strip()

    # Build parameterised query (safe against SQL injection)
    conditions = [
        "departure_airport = %(dep)s",
        "arrival_airport = %(arr)s",
        "status != 'cancelled'",
        "departure_time > NOW()",
    ]
    params: dict[str, Any] = {"dep": departure_airport, "arr": arrival_airport}

    if date:
        conditions.append("DATE(departure_time AT TIME ZONE 'UTC') = %(date)s")
        params["date"] = date

    if max_price is not None:
        conditions.append("price_usd <= %(max_price)s")
        params["max_price"] = max_price

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT
            f.flight_number,
            f.airline,
            dep.iata_code   AS departure_airport,
            dep.city        AS departure_city,
            arr.iata_code   AS arrival_airport,
            arr.city        AS arrival_city,
            f.departure_time AT TIME ZONE 'UTC' AS departure_time_utc,
            f.arrival_time   AT TIME ZONE 'UTC' AS arrival_time_utc,
            f.duration_minutes,
            f.price_usd,
            f.seats_available,
            f.aircraft_type,
            f.status
        FROM flights f
        JOIN airports dep ON dep.iata_code = f.departure_airport
        JOIN airports arr ON arr.iata_code = f.arrival_airport
        WHERE {where_clause}
        ORDER BY f.departure_time ASC
        LIMIT 10;
    """

    try:
        with ManagedConnection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        flights = []
        for row in rows:
            flights.append({
                "flight_number": row["flight_number"],
                "airline": row["airline"],
                "departure_airport": row["departure_airport"],
                "departure_city": row["departure_city"],
                "arrival_airport": row["arrival_airport"],
                "arrival_city": row["arrival_city"],
                "departure_time": str(row["departure_time_utc"]),
                "arrival_time": str(row["arrival_time_utc"]),
                "duration_minutes": row["duration_minutes"],
                "price_usd": float(row["price_usd"]),
                "seats_available": row["seats_available"],
                "aircraft_type": row["aircraft_type"],
                "status": row["status"],
            })

        log.info("search_flights result", count=len(flights))
        return {
            "query": {
                "departure_airport": departure_airport,
                "arrival_airport": arrival_airport,
                "date": date,
                "max_price_usd": max_price,
            },
            "flights_found": len(flights),
            "flights": flights,
        }

    except Exception as exc:
        log.error("search_flights failed", error=str(exc))
        return {"error": str(exc), "flights": []}


# ============================================================
# Tool 2: search_airport_amenities
# ============================================================

def search_airport_amenities(
    query: str,
    airport_iata: str | None = None,
    category: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Perform semantic vector search over airport amenities using Vertex AI embeddings
    and cosine similarity on the AlloyDB pgvector VECTOR(768) column.

    This tool implements the RAG (Retrieval-Augmented Generation) retrieval step.
    The query text is embedded using the same Vertex AI text-embedding-004 model
    used during data ingestion, then the embedding is compared against all stored
    amenity embeddings using cosine distance (<=> operator) to find semantically
    similar amenities — even when exact keywords don't match.

    This represents the analytical / vector search side of AlloyDB's HTAP
    architecture, running as a real-time ANN query over the ScaNN index.

    Args:
        query: Natural language query describing what the user is looking for
               (e.g. "quiet place to relax before a long flight",
                     "good coffee and breakfast",
                     "somewhere to charge my laptop and get WiFi").
        airport_iata: Optional IATA airport code to restrict search to one airport
                      (e.g. 'SFO', 'LAX', 'JFK'). Omit to search all airports.
        category: Optional category filter. One of: 'dining', 'lounge', 'retail',
                  'services', 'transportation', 'accessibility', 'entertainment'.
        top_k: Number of most semantically similar amenities to return (default 5).

    Returns:
        A dict containing a ranked list of amenities with name, airport, category,
        description, location, hours, price range, and cosine similarity score.
    """
    log.info(
        "Tool invoked: search_airport_amenities",
        query=query,
        airport=airport_iata,
        category=category,
        top_k=top_k,
    )

    settings = get_settings()

    # Step 1: Embed the user query with Vertex AI
    query_embedding = _embed_query(query, settings)

    # Step 2: Build SQL with cosine similarity (<=>) and optional filters
    conditions = ["a.embedding IS NOT NULL"]
    params: dict[str, Any] = {
        "embedding": query_embedding,
        "top_k": min(top_k, 20),  # Safety cap
    }

    if airport_iata:
        conditions.append("a.airport_iata = %(airport_iata)s")
        params["airport_iata"] = airport_iata.upper().strip()

    if category:
        conditions.append("a.category = %(category)s")
        params["category"] = category.lower().strip()

    where_clause = " AND ".join(conditions)

    # AlloyDB cosine similarity query
    # The <=> operator computes cosine distance (lower = more similar).
    # We convert to similarity: 1 - distance ∈ [0, 1].
    # In production AlloyDB, this query uses the ScaNN ANN index
    # for sub-millisecond retrieval at billions of rows.
    sql = f"""
        SELECT
            a.id,
            a.name,
            a.airport_iata,
            ap.name         AS airport_name,
            ap.city         AS airport_city,
            a.category,
            a.description,
            a.terminal,
            a.location_detail,
            a.hours_of_operation,
            a.price_range,
            1 - (a.embedding <=> %(embedding)s::vector)  AS similarity_score
        FROM amenities a
        JOIN airports ap ON ap.iata_code = a.airport_iata
        WHERE {where_clause}
        ORDER BY a.embedding <=> %(embedding)s::vector ASC
        LIMIT %(top_k)s;
    """

    try:
        with ManagedConnection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        amenities = []
        for row in rows:
            score = float(row["similarity_score"])
            if score >= settings.vector_similarity_threshold:
                amenities.append({
                    "id": row["id"],
                    "name": row["name"],
                    "airport": f"{row['airport_city']} ({row['airport_iata']})",
                    "category": row["category"],
                    "description": row["description"],
                    "terminal": row["terminal"],
                    "location": row["location_detail"],
                    "hours": row["hours_of_operation"],
                    "price_range": row["price_range"],
                    "similarity_score": round(score, 4),
                })

        log.info("search_airport_amenities result", count=len(amenities))
        return {
            "query": query,
            "airport_filter": airport_iata,
            "category_filter": category,
            "amenities_found": len(amenities),
            "amenities": amenities,
            "search_type": "vector_cosine_similarity",
            "embedding_model": settings.embedding_model,
        }

    except Exception as exc:
        log.error("search_airport_amenities failed", error=str(exc))
        return {"error": str(exc), "amenities": []}


# ============================================================
# Tool 3: get_airport_info
# ============================================================

def get_airport_info(iata_code: str) -> dict[str, Any]:
    """
    Retrieve detailed information about an airport from AlloyDB by its IATA code.

    Use this tool when the user asks about a specific airport's location,
    timezone, or general information. This is a simple OLTP lookup against
    the airports master data table in AlloyDB.

    Args:
        iata_code: The 3-letter IATA airport code (e.g. 'SFO', 'JFK', 'LAX', 'ORD').

    Returns:
        A dict containing airport name, city, country, timezone, and coordinates.
        Returns an error message if the airport code is not found in the database.
    """
    log.info("Tool invoked: get_airport_info", iata_code=iata_code)

    sql = """
        SELECT
            iata_code, name, city, country, timezone,
            latitude, longitude
        FROM airports
        WHERE iata_code = %(iata)s;
    """

    try:
        with ManagedConnection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"iata": iata_code.upper().strip()})
                row = cur.fetchone()

        if not row:
            return {"error": f"Airport '{iata_code}' not found in AlloyDB."}

        return {
            "iata_code": row["iata_code"],
            "name": row["name"],
            "city": row["city"],
            "country": row["country"],
            "timezone": row["timezone"],
            "coordinates": {
                "latitude": float(row["latitude"]) if row["latitude"] else None,
                "longitude": float(row["longitude"]) if row["longitude"] else None,
            },
        }

    except Exception as exc:
        log.error("get_airport_info failed", error=str(exc))
        return {"error": str(exc)}


# ============================================================
# Tool 4: list_available_routes
# ============================================================

def list_available_routes() -> dict[str, Any]:
    """
    List all city-pair routes currently served in the AlloyDB flights database.

    Use this tool when a user asks what destinations are available, what routes
    are offered, or wants to know which airports have scheduled flights.
    This is a simple aggregation query against the OLTP flights table.

    Returns:
        A dict containing a list of unique city-pair routes with origin and
        destination city and airport code.
    """
    log.info("Tool invoked: list_available_routes")

    sql = """
        SELECT DISTINCT
            dep.iata_code   AS departure_iata,
            dep.city        AS departure_city,
            arr.iata_code   AS arrival_iata,
            arr.city        AS arrival_city
        FROM flights f
        JOIN airports dep ON dep.iata_code = f.departure_airport
        JOIN airports arr ON arr.iata_code = f.arrival_airport
        WHERE f.status != 'cancelled'
          AND f.departure_time > NOW()
        ORDER BY dep.city, arr.city;
    """

    try:
        with ManagedConnection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        routes = [
            {
                "from": f"{row['departure_city']} ({row['departure_iata']})",
                "to": f"{row['arrival_city']} ({row['arrival_iata']})",
            }
            for row in rows
        ]

        return {"routes_available": len(routes), "routes": routes}

    except Exception as exc:
        log.error("list_available_routes failed", error=str(exc))
        return {"error": str(exc), "routes": []}


# ============================================================
# Internal: Query embedding helper
# ============================================================

def _embed_query(text: str, settings: Any) -> list[float]:
    """
    Generate a 768-dimension Vertex AI embedding for the query text.

    Uses text-embedding-004 with RETRIEVAL_QUERY task type (optimised
    for query-side embeddings in asymmetric retrieval tasks, as opposed
    to RETRIEVAL_DOCUMENT used during indexing).
    """
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

        vertexai.init(
            project=settings.google_cloud_project,
            location=settings.vertex_ai_location,
        )
        model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
        inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_QUERY")]
        embeddings = model.get_embeddings(inputs)
        return embeddings[0].values

    except Exception as exc:
        log.warning(
            "Vertex AI query embedding failed — using mock embedding for local dev.",
            error=str(exc),
        )
        import hashlib
        import random
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)
        vec = [random.gauss(0, 1) for _ in range(768)]
        magnitude = sum(x**2 for x in vec) ** 0.5
        return [x / magnitude for x in vec]


# ============================================================
# Tool Registry — exported for agent_platform.py binding
# ============================================================

#: All callable tools available to the Gemini Agent Platform.
#: Each entry maps a tool name to its Python callable.
TOOL_REGISTRY: dict[str, callable] = {
    "search_flights": search_flights,
    "search_airport_amenities": search_airport_amenities,
    "get_airport_info": get_airport_info,
    "list_available_routes": list_available_routes,
}
