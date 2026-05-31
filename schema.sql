-- ============================================================
-- AlloyDB Schema — Vertex AI RAG + Agent Platform Lab
-- ============================================================
-- GCP Architecture Reference:
--   "Build an LLM and RAG-based Chat Application
--    with AlloyDB and Agent Platform"
--
-- AlloyDB HTAP Highlights replicated here:
--   1. pgvector extension  → AlloyDB native vector support
--   2. ivfflat/hnsw index  → Approximate Nearest Neighbor (ANN)
--      (maps to AlloyDB's integrated ScaNN index)
--   3. 768-dim vectors     → Vertex AI text-embedding-004 output
--   4. Hybrid queries      → JOIN transactional + vector columns
-- ============================================================

-- Enable the pgvector extension (pre-installed in AlloyDB)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- TABLE: airports
-- Operational / OLTP data (AlloyDB HTAP — transactional side)
-- ============================================================
CREATE TABLE IF NOT EXISTS airports (
    id              SERIAL PRIMARY KEY,
    iata_code       CHAR(3)         NOT NULL UNIQUE,
    name            VARCHAR(255)    NOT NULL,
    city            VARCHAR(100)    NOT NULL,
    country         VARCHAR(100)    NOT NULL DEFAULT 'United States',
    timezone        VARCHAR(50)     NOT NULL DEFAULT 'America/Chicago',
    latitude        DECIMAL(9, 6),
    longitude       DECIMAL(9, 6),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE airports IS
    'Operational airport master data. OLTP workload in AlloyDB HTAP.';

-- ============================================================
-- TABLE: flights
-- Real-time operational data (AlloyDB HTAP — transactional side)
-- ============================================================
CREATE TABLE IF NOT EXISTS flights (
    id              SERIAL PRIMARY KEY,
    flight_number   VARCHAR(10)     NOT NULL,
    airline         VARCHAR(100)    NOT NULL,
    departure_airport CHAR(3)       NOT NULL REFERENCES airports(iata_code),
    arrival_airport   CHAR(3)       NOT NULL REFERENCES airports(iata_code),
    departure_time  TIMESTAMPTZ     NOT NULL,
    arrival_time    TIMESTAMPTZ     NOT NULL,
    duration_minutes INT            NOT NULL,
    price_usd       DECIMAL(10, 2)  NOT NULL,
    seats_available INT             NOT NULL DEFAULT 0,
    aircraft_type   VARCHAR(50),
    status          VARCHAR(20)     NOT NULL DEFAULT 'scheduled'
                    CHECK (status IN ('scheduled','boarding','departed','arrived','cancelled')),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE flights IS
    'Real-time flight schedule and availability. OLTP workload in AlloyDB HTAP.';

CREATE INDEX IF NOT EXISTS idx_flights_departure_airport
    ON flights(departure_airport);
CREATE INDEX IF NOT EXISTS idx_flights_arrival_airport
    ON flights(arrival_airport);
CREATE INDEX IF NOT EXISTS idx_flights_departure_time
    ON flights(departure_time);
CREATE INDEX IF NOT EXISTS idx_flights_status
    ON flights(status);

-- ============================================================
-- TABLE: amenities
-- RAG knowledge base with Vertex AI embeddings
-- (AlloyDB HTAP — analytical / vector search side)
--
-- Vector column: embedding VECTOR(768)
--   Dimension 768 matches Vertex AI text-embedding-004 output.
--   In AlloyDB production, this uses the integrated ScaNN
--   index for sub-millisecond ANN retrieval at scale.
-- ============================================================
CREATE TABLE IF NOT EXISTS amenities (
    id              SERIAL PRIMARY KEY,
    airport_iata    CHAR(3)         NOT NULL REFERENCES airports(iata_code),
    name            VARCHAR(255)    NOT NULL,
    description     TEXT            NOT NULL,
    category        VARCHAR(50)     NOT NULL
                    CHECK (category IN (
                        'dining', 'lounge', 'retail', 'services',
                        'transportation', 'accessibility', 'entertainment'
                    )),
    terminal        VARCHAR(10),
    location_detail VARCHAR(255),
    hours_of_operation VARCHAR(100),
    price_range     VARCHAR(20)     CHECK (price_range IN ('free','$','$$','$$$','$$$$')),
    -- -------------------------------------------------------
    -- Vertex AI Embedding Column
    -- Populated by: seed_and_embed.py → Vertex AI text-embedding-004
    -- Queried by:   gcp_toolbox.py → cosine similarity (<=>)
    -- -------------------------------------------------------
    embedding       VECTOR(768),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE amenities IS
    'Airport amenities knowledge base with Vertex AI text embeddings. '
    'Analytical / vector search workload in AlloyDB HTAP. '
    'Embedding dimension 768 matches text-embedding-004 model output.';

COMMENT ON COLUMN amenities.embedding IS
    'Vertex AI text-embedding-004 vector (768 dims). '
    'Indexed with HNSW for ANN search (maps to AlloyDB ScaNN in production).';

-- -------------------------------------------------------
-- HNSW Index on embedding column
-- Maps to: AlloyDB integrated ScaNN ANN index
-- Operator class: vector_cosine_ops → cosine distance
-- -------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_amenities_embedding_hnsw
    ON amenities
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX idx_amenities_embedding_hnsw IS
    'HNSW ANN index on Vertex AI embeddings. '
    'Maps to AlloyDB ScaNN index in production for sub-ms retrieval.';

-- ============================================================
-- Seed airports master data
-- ============================================================
INSERT INTO airports (iata_code, name, city, country, timezone, latitude, longitude)
VALUES
    ('SFO', 'San Francisco International Airport',    'San Francisco', 'United States', 'America/Los_Angeles', 37.6213, -122.3790),
    ('LAX', 'Los Angeles International Airport',     'Los Angeles',   'United States', 'America/Los_Angeles', 33.9425, -118.4081),
    ('JFK', 'John F. Kennedy International Airport', 'New York',      'United States', 'America/New_York',    40.6413,  -73.7781),
    ('ORD', 'O''Hare International Airport',         'Chicago',       'United States', 'America/Chicago',     41.9742,  -87.9073),
    ('ATL', 'Hartsfield-Jackson Atlanta Airport',    'Atlanta',       'United States', 'America/New_York',    33.6407,  -84.4277),
    ('SEA', 'Seattle-Tacoma International Airport',  'Seattle',       'United States', 'America/Los_Angeles', 47.4502, -122.3088),
    ('DEN', 'Denver International Airport',          'Denver',        'United States', 'America/Denver',      39.8561, -104.6737),
    ('MIA', 'Miami International Airport',           'Miami',         'United States', 'America/New_York',    25.7959,  -80.2870)
ON CONFLICT (iata_code) DO NOTHING;
