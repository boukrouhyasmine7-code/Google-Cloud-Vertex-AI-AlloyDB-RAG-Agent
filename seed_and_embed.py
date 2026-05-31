"""
seed_and_embed.py
=================
Populates the AlloyDB local replica with mock flight and amenity data,
then generates Vertex AI text embeddings for each amenity record and
persists them to the `embedding` VECTOR(768) column.

GCP Architecture Mapping
-------------------------
Local component          → GCP equivalent
─────────────────────────────────────────────────────────────
PostgreSQL + pgvector    → AlloyDB with integrated pgvector
text-embedding-004 API   → Vertex AI Embeddings endpoint
This script             → Cloud Run Job / Dataflow pipeline
Cosine similarity query  → AlloyDB ScaNN ANN index

Usage
-----
    python scripts/seed_and_embed.py [--reset]

Flags
-----
    --reset   Drop and re-insert all data before embedding.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from repo root or scripts/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.database import get_db_connection

# ---------------------------------------------------------------------------
# Configure structured logger (Cloud Logging compatible JSON format)
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Mock data: Flights
# ---------------------------------------------------------------------------
MOCK_FLIGHTS: list[dict[str, Any]] = [
    # SFO → LAX
    {
        "flight_number": "GA101", "airline": "Google Air",
        "departure_airport": "SFO", "arrival_airport": "LAX",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=2),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=3, minutes=15),
        "duration_minutes": 75, "price_usd": 149.99, "seats_available": 42,
        "aircraft_type": "Boeing 737-800", "status": "scheduled",
    },
    {
        "flight_number": "GA102", "airline": "Google Air",
        "departure_airport": "SFO", "arrival_airport": "LAX",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=6),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=7, minutes=20),
        "duration_minutes": 80, "price_usd": 129.99, "seats_available": 15,
        "aircraft_type": "Airbus A320", "status": "scheduled",
    },
    # SFO → JFK
    {
        "flight_number": "GA201", "airline": "Google Air",
        "departure_airport": "SFO", "arrival_airport": "JFK",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=1),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=6, minutes=30),
        "duration_minutes": 330, "price_usd": 389.00, "seats_available": 8,
        "aircraft_type": "Boeing 757-200", "status": "scheduled",
    },
    # LAX → ORD
    {
        "flight_number": "GA301", "airline": "Google Air",
        "departure_airport": "LAX", "arrival_airport": "ORD",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=3),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=7, minutes=45),
        "duration_minutes": 225, "price_usd": 279.50, "seats_available": 61,
        "aircraft_type": "Boeing 737 MAX 8", "status": "scheduled",
    },
    # JFK → ATL
    {
        "flight_number": "GA401", "airline": "Google Air",
        "departure_airport": "JFK", "arrival_airport": "ATL",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=4),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=6, minutes=20),
        "duration_minutes": 140, "price_usd": 198.00, "seats_available": 0,
        "aircraft_type": "Airbus A319", "status": "scheduled",
    },
    # SEA → DEN
    {
        "flight_number": "GA501", "airline": "Google Air",
        "departure_airport": "SEA", "arrival_airport": "DEN",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=5),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=7, minutes=50),
        "duration_minutes": 170, "price_usd": 234.75, "seats_available": 29,
        "aircraft_type": "Boeing 737-800", "status": "scheduled",
    },
    # ATL → MIA
    {
        "flight_number": "GA601", "airline": "Google Air",
        "departure_airport": "ATL", "arrival_airport": "MIA",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=2, minutes=30),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=4, minutes=30),
        "duration_minutes": 120, "price_usd": 174.99, "seats_available": 38,
        "aircraft_type": "Airbus A320", "status": "scheduled",
    },
    # DEN → SFO
    {
        "flight_number": "GA701", "airline": "Google Air",
        "departure_airport": "DEN", "arrival_airport": "SFO",
        "departure_time": datetime.now(tz=timezone.utc) + timedelta(hours=8),
        "arrival_time": datetime.now(tz=timezone.utc) + timedelta(hours=11, minutes=15),
        "duration_minutes": 195, "price_usd": 315.00, "seats_available": 4,
        "aircraft_type": "Boeing 757-200", "status": "scheduled",
    },
]

# ---------------------------------------------------------------------------
# Mock data: Amenities (text that will be embedded by Vertex AI)
# ---------------------------------------------------------------------------
MOCK_AMENITIES: list[dict[str, Any]] = [
    # SFO amenities
    {
        "airport_iata": "SFO", "name": "Fog City Bistro",
        "description": (
            "A farm-to-table restaurant celebrating Northern California cuisine. "
            "Featuring seasonal small plates, fresh-caught seafood, craft cocktails, "
            "and an extensive local wine list. Perfect for pre-flight dining with "
            "stunning views of the tarmac. Vegetarian and vegan options available."
        ),
        "category": "dining", "terminal": "Terminal 2",
        "location_detail": "Near Gate D40", "hours_of_operation": "5:00 AM - 11:00 PM",
        "price_range": "$$$",
    },
    {
        "airport_iata": "SFO", "name": "United Club Lounge",
        "description": (
            "Premium airport lounge offering complimentary snacks, beverages, and "
            "full bar service. High-speed Wi-Fi, comfortable seating, shower facilities, "
            "and dedicated customer service agents. Business center with printing services. "
            "Accessible to United Polaris, First Class, and 1K members."
        ),
        "category": "lounge", "terminal": "Terminal 3",
        "location_detail": "Above Gate 68", "hours_of_operation": "5:30 AM - 10:00 PM",
        "price_range": "free",
    },
    {
        "airport_iata": "SFO", "name": "Tech Haven Electronics",
        "description": (
            "Full-service electronics and travel accessories store. Carries charging cables, "
            "noise-cancelling headphones, portable power banks, laptop adapters, and "
            "last-minute tech essentials. Apple and Samsung authorized accessories. "
            "Device charging stations available while you browse."
        ),
        "category": "retail", "terminal": "International Terminal G",
        "location_detail": "Gate G92", "hours_of_operation": "6:00 AM - 11:00 PM",
        "price_range": "$$$",
    },
    {
        "airport_iata": "SFO", "name": "Bay Area Wellness Spa",
        "description": (
            "Airport spa offering express massages, manicures, pedicures, and reflexology. "
            "Relax in a serene environment with aromatherapy treatments. "
            "Walk-ins welcome; appointments recommended for longer services. "
            "Shower rooms available for a nominal fee. Perfect for long layovers."
        ),
        "category": "services", "terminal": "Terminal 2",
        "location_detail": "Near Security Checkpoint", "hours_of_operation": "7:00 AM - 9:00 PM",
        "price_range": "$$$",
    },
    # LAX amenities
    {
        "airport_iata": "LAX", "name": "Pacific Rim Food Hall",
        "description": (
            "Multi-vendor food hall celebrating Los Angeles' diverse culinary scene. "
            "Featuring sushi, Korean BBQ, authentic Mexican tacos, and artisan burgers. "
            "Communal seating with phone charging stations at every table. "
            "Late night dining option for red-eye passengers."
        ),
        "category": "dining", "terminal": "Tom Bradley International Terminal",
        "location_detail": "Level 4, Central Atrium", "hours_of_operation": "24 hours",
        "price_range": "$$",
    },
    {
        "airport_iata": "LAX", "name": "Star Alliance Business Lounge",
        "description": (
            "Spacious lounge for Star Alliance Gold and business class passengers. "
            "Open bar, hot buffet meals, private napping pods, and panoramic runway views. "
            "Dedicated conference rooms bookable via the app. "
            "Children's play area available. Shower suites with luxury amenities."
        ),
        "category": "lounge", "terminal": "Tom Bradley International Terminal",
        "location_detail": "Mezzanine Level", "hours_of_operation": "5:00 AM - 1:00 AM",
        "price_range": "free",
    },
    {
        "airport_iata": "LAX", "name": "FlyAway Bus Service",
        "description": (
            "Direct express bus service connecting LAX to downtown Los Angeles Union Station, "
            "Hollywood, Van Nuys, and Westwood/UCLA. Runs every 30 minutes, 24 hours a day. "
            "Luggage storage available. Affordable alternative to rideshare services. "
            "TAP card accepted; tickets purchasable at kiosks."
        ),
        "category": "transportation", "terminal": "Lower Level Arrivals",
        "location_detail": "Island 1, Ground Transportation", "hours_of_operation": "24 hours",
        "price_range": "$",
    },
    # JFK amenities
    {
        "airport_iata": "JFK", "name": "The Centurion Lounge",
        "description": (
            "Iconic American Express Centurion Lounge with locally-inspired food program "
            "curated by NYC celebrity chefs. Full open bar, premium spa treatments, "
            "and a quiet room for rest. Exclusive to Amex Platinum and Centurion cardholders. "
            "Children under 18 welcomed. Stunning Manhattan skyline views on clear days."
        ),
        "category": "lounge", "terminal": "Terminal 4",
        "location_detail": "Concourse B, Near Gate B40", "hours_of_operation": "5:30 AM - 11:00 PM",
        "price_range": "free",
    },
    {
        "airport_iata": "JFK", "name": "Shake Shack Terminal 4",
        "description": (
            "Iconic New York burger institution serving ShackBurgers, crinkle-cut fries, "
            "hand-spun frozen custard shakes, and craft beer. "
            "Vegetarian Shroom Burger and hot dogs available. "
            "Quick-service format ideal for tight connections. Local New York experience."
        ),
        "category": "dining", "terminal": "Terminal 4",
        "location_detail": "Post-Security, Gate B20 area", "hours_of_operation": "6:00 AM - 10:00 PM",
        "price_range": "$$",
    },
    # ORD amenities
    {
        "airport_iata": "ORD", "name": "Chicago Deep Dish by Giordano's",
        "description": (
            "World-famous Chicago deep-dish pizza served in the heart of O'Hare. "
            "Giordano's stuffed pizza requires 45 minutes baking time — order ahead. "
            "Also serving thin-crust options, pasta, and Chicago-style hot dogs. "
            "Craft Illinois beers on tap. Great for long layovers craving an authentic Chicago meal."
        ),
        "category": "dining", "terminal": "Terminal 1",
        "location_detail": "Concourse B, near Gate B16", "hours_of_operation": "6:00 AM - 9:30 PM",
        "price_range": "$$",
    },
    {
        "airport_iata": "ORD", "name": "Aira Accessibility Services Hub",
        "description": (
            "Free accessibility assistance center providing wheelchair services, "
            "visual impairment navigation support via Aira app, and hearing loop systems. "
            "Staffed by trained agents fluent in ASL. Service animal relief areas nearby. "
            "Quiet sensory-friendly space available for passengers with autism or sensory sensitivities."
        ),
        "category": "accessibility", "terminal": "Terminal 3",
        "location_detail": "Near Main Security Checkpoint", "hours_of_operation": "24 hours",
        "price_range": "free",
    },
    # ATL amenities
    {
        "airport_iata": "ATL", "name": "Paschal's Southern Kitchen",
        "description": (
            "Atlanta institution since 1947, now bringing legendary Southern comfort food "
            "to Hartsfield-Jackson. Famous fried chicken, collard greens, cornbread, "
            "and peach cobbler. Full breakfast service starting at 5 AM. "
            "A celebration of Atlanta's rich culinary and cultural heritage."
        ),
        "category": "dining", "terminal": "Concourse B",
        "location_detail": "Near Gate B15", "hours_of_operation": "5:00 AM - 10:00 PM",
        "price_range": "$$",
    },
    # SEA amenities
    {
        "airport_iata": "SEA", "name": "Starbucks Reserve Bar SEA",
        "description": (
            "Exclusive Starbucks Reserve experience at Seattle's home airport. "
            "Rare single-origin coffees, nitro cold brews, and Reserve-only beverages "
            "not available at standard Starbucks locations. Coffee education bar. "
            "Locally sourced pastries and food items. The definitive Seattle coffee experience."
        ),
        "category": "dining", "terminal": "Central Terminal",
        "location_detail": "Main Hall, Pre-Security", "hours_of_operation": "4:30 AM - 11:30 PM",
        "price_range": "$$",
    },
    # DEN amenities
    {
        "airport_iata": "DEN", "name": "Colorado Craft Beer Garden",
        "description": (
            "Celebrating Colorado's legendary craft beer scene with 40 rotating taps "
            "exclusively featuring Colorado breweries: Odell, New Belgium, Breckenridge Brewery, "
            "and Tivoli. Mountain-inspired pub food including bison burgers and green chile nachos. "
            "Live acoustic music on weekends. Must-visit for craft beer enthusiasts."
        ),
        "category": "dining", "terminal": "Jeppesen Terminal",
        "location_detail": "Level 6, East Wing", "hours_of_operation": "10:00 AM - 10:00 PM",
        "price_range": "$$",
    },
    # MIA amenities
    {
        "airport_iata": "MIA", "name": "Miami Duty-Free Luxury Shops",
        "description": (
            "International duty-free shopping featuring Louis Vuitton, Chanel, Dior, "
            "Rolex, and Tiffany & Co. Tax-free pricing on fragrances, cosmetics, spirits, "
            "tobacco, and fine jewelry. Currency exchange services on-site. "
            "Exclusive to departing international passengers with boarding passes."
        ),
        "category": "retail", "terminal": "North Terminal",
        "location_detail": "International Departures, Post-Customs", "hours_of_operation": "6:00 AM - 11:30 PM",
        "price_range": "$$$$",
    },
]


def get_vertex_embedding(text: str, settings: Any) -> list[float]:
    """
    Generate a 768-dimension text embedding via Vertex AI text-embedding-004.

    In production on GCP, this call routes to the Vertex AI Embeddings
    endpoint in your project's region. The embedding dimension (768) is
    fixed by the text-embedding-004 model and matches the VECTOR(768)
    column in the AlloyDB schema.

    Args:
        text: The plain text to embed (amenity description).
        settings: Application settings with GCP project / region config.

    Returns:
        List of 768 float values representing the semantic embedding.
    """
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

        vertexai.init(project=settings.google_cloud_project, location=settings.vertex_ai_location)
        model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
        inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_DOCUMENT")]
        embeddings = model.get_embeddings(inputs)
        return embeddings[0].values

    except ImportError:
        log.warning(
            "vertexai SDK not available — generating mock 768-dim embedding for local dev. "
            "Install google-cloud-aiplatform for real Vertex AI embeddings."
        )
        import random
        random.seed(hash(text) % (2**32))
        vec = [random.gauss(0, 1) for _ in range(768)]
        # Normalize to unit vector (cosine similarity requires this)
        magnitude = sum(x**2 for x in vec) ** 0.5
        return [x / magnitude for x in vec]

    except Exception as exc:
        log.warning(
            "Vertex AI embedding call failed — using deterministic mock embedding.",
            error=str(exc),
        )
        import hashlib
        import struct
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        import random
        random.seed(seed)
        vec = [random.gauss(0, 1) for _ in range(768)]
        magnitude = sum(x**2 for x in vec) ** 0.5
        return [x / magnitude for x in vec]


def seed_flights(cursor: Any, reset: bool) -> int:
    """Insert mock flight records into AlloyDB."""
    if reset:
        cursor.execute("DELETE FROM flights;")
        log.info("Cleared existing flights table")

    insert_sql = """
        INSERT INTO flights (
            flight_number, airline, departure_airport, arrival_airport,
            departure_time, arrival_time, duration_minutes,
            price_usd, seats_available, aircraft_type, status
        ) VALUES (
            %(flight_number)s, %(airline)s, %(departure_airport)s, %(arrival_airport)s,
            %(departure_time)s, %(arrival_time)s, %(duration_minutes)s,
            %(price_usd)s, %(seats_available)s, %(aircraft_type)s, %(status)s
        )
        ON CONFLICT DO NOTHING;
    """
    inserted = 0
    for flight in MOCK_FLIGHTS:
        cursor.execute(insert_sql, flight)
        inserted += cursor.rowcount
    return inserted


def seed_amenities_without_embeddings(cursor: Any, reset: bool) -> int:
    """Insert amenity records (without embeddings first)."""
    if reset:
        cursor.execute("DELETE FROM amenities;")
        log.info("Cleared existing amenities table")

    insert_sql = """
        INSERT INTO amenities (
            airport_iata, name, description, category,
            terminal, location_detail, hours_of_operation, price_range
        ) VALUES (
            %(airport_iata)s, %(name)s, %(description)s, %(category)s,
            %(terminal)s, %(location_detail)s, %(hours_of_operation)s, %(price_range)s
        )
        ON CONFLICT DO NOTHING
        RETURNING id;
    """
    ids = []
    for amenity in MOCK_AMENITIES:
        cursor.execute(insert_sql, amenity)
        row = cursor.fetchone()
        if row:
            ids.append(row[0])
    return len(ids)


def embed_amenities(conn: Any, settings: Any) -> None:
    """
    Fetch all amenities without embeddings, generate Vertex AI embeddings,
    and persist them to the AlloyDB vector column.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, description FROM amenities WHERE embedding IS NULL;")
        rows = cur.fetchall()

    if not rows:
        console.print("[green]✓[/green] All amenities already have embeddings.")
        return

    console.print(f"\n[bold cyan]Generating Vertex AI embeddings for {len(rows)} amenities...[/bold cyan]")
    console.print(f"  Model: [yellow]{settings.embedding_model}[/yellow]  Dimensions: [yellow]768[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding amenities → AlloyDB VECTOR(768)...", total=len(rows))

        for amenity_id, name, description in rows:
            progress.update(task, description=f"Embedding: [italic]{name[:40]}[/italic]")

            embedding = get_vertex_embedding(description, settings)

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE amenities SET embedding = %s WHERE id = %s;",
                    (embedding, amenity_id),
                )
            conn.commit()

            time.sleep(0.05)  # Respect Vertex AI rate limits in production
            progress.advance(task)


def print_summary(conn: Any) -> None:
    """Print a summary table of seeded data."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM airports;")
        airport_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM flights;")
        flight_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM amenities;")
        amenity_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM amenities WHERE embedding IS NOT NULL;")
        embedded_count = cur.fetchone()[0]

    table = Table(title="[bold]AlloyDB Seed Summary[/bold]", show_header=True, header_style="bold magenta")
    table.add_column("Table", style="cyan", no_wrap=True)
    table.add_column("Records", justify="right", style="green")
    table.add_column("Notes", style="dim")

    table.add_row("airports", str(airport_count), "Static master data (OLTP)")
    table.add_row("flights", str(flight_count), "Operational schedule (OLTP)")
    table.add_row("amenities", str(amenity_count), f"{embedded_count}/{amenity_count} embedded (RAG)")

    console.print()
    console.print(table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed AlloyDB local replica with mock data and Vertex AI embeddings."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing data before re-seeding (idempotent re-run).",
    )
    args = parser.parse_args()

    settings = get_settings()

    console.print()
    console.rule("[bold blue]AlloyDB Seed + Vertex AI Embed Pipeline[/bold blue]")
    console.print(
        f"  [dim]Project:[/dim] {settings.google_cloud_project}  "
        f"[dim]Embedding model:[/dim] {settings.embedding_model}  "
        f"[dim]Dimensions:[/dim] 768"
    )
    console.print()

    conn = get_db_connection(settings)
    try:
        with conn.cursor() as cur:
            # Seed flights
            flights_inserted = seed_flights(cur, reset=args.reset)
            conn.commit()
            log.info("Flights seeded", count=flights_inserted)

            # Seed amenities (text only, no embeddings yet)
            amenities_inserted = seed_amenities_without_embeddings(cur, reset=args.reset)
            conn.commit()
            log.info("Amenities seeded (pre-embedding)", count=amenities_inserted)

        # Generate and persist Vertex AI embeddings
        embed_amenities(conn, settings)

        print_summary(conn)

        console.print("[bold green]✓ AlloyDB local replica seeded and embedded successfully.[/bold green]")
        console.print("  Run [cyan]python -m uvicorn main:app --reload[/cyan] to start the Agent Platform API.\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
