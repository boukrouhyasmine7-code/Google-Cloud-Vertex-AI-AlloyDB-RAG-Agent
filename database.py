"""
app/database.py
===============
AlloyDB connection management using psycopg2.

GCP Architecture Mapping
-------------------------
Local component               → GCP equivalent
─────────────────────────────────────────────────────────────
psycopg2 connection pool      → AlloyDB Connector (Go/Python)
get_db_connection()           → Cloud SQL / AlloyDB Auth Proxy
pgvector extension            → AlloyDB native vector support

AlloyDB uses the standard PostgreSQL wire protocol, so every
query written here executes identically against a live AlloyDB
primary instance in production.
"""

from __future__ import annotations

import structlog
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)

# Module-level connection pool (initialized lazily)
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool(settings: Settings | None = None) -> psycopg2.pool.ThreadedConnectionPool:
    """
    Return (or initialize) the module-level threaded connection pool.

    In GCP production this would use the AlloyDB Python Connector:
        from google.cloud.alloydb.connector import Connector
    The connector handles IAM authentication and mTLS automatically.
    """
    global _pool
    if _pool is None:
        s = settings or get_settings()
        log.info(
            "Initializing AlloyDB connection pool",
            host=s.alloydb_host,
            port=s.alloydb_port,
            database=s.alloydb_database,
            pool_size=s.alloydb_pool_size,
        )
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=s.alloydb_pool_size + s.alloydb_max_overflow,
            host=s.alloydb_host,
            port=s.alloydb_port,
            dbname=s.alloydb_database,
            user=s.alloydb_user,
            password=s.alloydb_password,
            cursor_factory=RealDictCursor,
            options="-c search_path=public",
        )
    return _pool


def get_db_connection(settings: Settings | None = None) -> psycopg2.extensions.connection:
    """
    Checkout a connection from the AlloyDB pool.

    Usage:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(...)
        finally:
            release_db_connection(conn)

    In production: the AlloyDB Connector handles IAM auth + mTLS.
    """
    pool = _get_pool(settings)
    return pool.getconn()


def release_db_connection(conn: psycopg2.extensions.connection) -> None:
    """Return a connection to the pool."""
    pool = _get_pool()
    pool.putconn(conn)


def close_pool() -> None:
    """Gracefully close all pooled connections (call on app shutdown)."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        log.info("AlloyDB connection pool closed")


class ManagedConnection:
    """
    Context manager for AlloyDB connections.

    Usage:
        async with ManagedConnection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._conn: psycopg2.extensions.connection | None = None

    def __enter__(self) -> psycopg2.extensions.connection:
        self._conn = get_db_connection(self._settings)
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn:
            if exc_type:
                self._conn.rollback()
            release_db_connection(self._conn)
            self._conn = None
        return False
