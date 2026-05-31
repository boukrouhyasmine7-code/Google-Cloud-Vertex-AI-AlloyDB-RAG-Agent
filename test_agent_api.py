"""
tests/test_agent_api.py
=======================
Integration tests for the Vertex AI Agent Platform API.

Tests the FastAPI endpoints using httpx TestClient.
Uses pytest-asyncio for async test support.

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short -k "health"
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI test client. Skips DB and Gemini init via mocks."""
    with patch("app.agent_platform.GeminiAgent._init_gemini", return_value=None):
        with patch("app.database._get_pool") as mock_pool:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.__enter__ = lambda s: mock_cursor
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_cursor.fetchone.return_value = {"count": 1}
            mock_conn.cursor.return_value = mock_cursor
            mock_pool.return_value.getconn.return_value = mock_conn
            mock_pool.return_value.putconn.return_value = None

            with TestClient(app) as c:
                yield c


class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "vertex-ai-agent-platform"

    def test_root_returns_architecture_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "AlloyDB" in data["architecture"]["vector_store"]
        assert "Gemini" in data["architecture"]["llm"]


class TestChatEndpoint:
    def test_chat_requires_message(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_returns_session_id(self, client):
        with patch("app.agent_platform.GeminiAgent.chat", return_value="Mock response from Gemini."):
            response = client.post("/chat", json={"message": "Hello"})
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert "response" in data
            assert "latency_ms" in data

    def test_chat_message_too_long(self, client):
        response = client.post("/chat", json={"message": "x" * 5000})
        assert response.status_code == 422

    def test_chat_multi_turn_uses_session(self, client):
        with patch("app.agent_platform.GeminiAgent.chat", return_value="Turn 1 response."):
            r1 = client.post("/chat", json={"message": "First message"})
            session_id = r1.json()["session_id"]

        with patch("app.agent_platform.GeminiAgent.chat", return_value="Turn 2 response."):
            r2 = client.post("/chat", json={"message": "Second message", "session_id": session_id})
            assert r2.json()["session_id"] == session_id


class TestSessionEndpoints:
    def test_get_nonexistent_session_returns_404(self, client):
        response = client.get("/sessions/nonexistent-id-xyz")
        assert response.status_code == 404

    def test_delete_nonexistent_session_returns_404(self, client):
        response = client.delete("/sessions/nonexistent-id-xyz")
        assert response.status_code == 404


class TestToolbox:
    """Unit tests for gcp_toolbox functions (without live AlloyDB)."""

    def test_tool_registry_contains_expected_tools(self):
        from app.gcp_toolbox import TOOL_REGISTRY
        expected = {"search_flights", "search_airport_amenities", "get_airport_info", "list_available_routes"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_search_flights_returns_error_on_db_failure(self):
        from app.gcp_toolbox import search_flights
        with patch("app.gcp_toolbox.ManagedConnection.__enter__", side_effect=Exception("DB down")):
            result = search_flights("SFO", "LAX")
            assert "error" in result
            assert result["flights"] == []

    def test_search_amenities_returns_error_on_db_failure(self):
        from app.gcp_toolbox import search_airport_amenities
        with patch("app.gcp_toolbox.ManagedConnection.__enter__", side_effect=Exception("DB down")):
            result = search_airport_amenities("coffee and wifi")
            assert "error" in result
