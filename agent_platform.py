"""
app/agent_platform.py
=====================
Vertex AI Agent Platform — Gemini Pro Orchestration Layer

This module implements the core Agent Platform loop:
  1. Receive user message
  2. Invoke Gemini Pro with tool declarations (function calling)
  3. If Gemini selects a tool → execute the corresponding gcp_toolbox function
  4. Feed tool result back to Gemini → continue until final text response
  5. Maintain full session chat history for multi-turn conversations

GCP Architecture Mapping
-------------------------
Local component                   → GCP equivalent
─────────────────────────────────────────────────────────────────────
GeminiAgent class                 → Vertex AI Agent Builder / Agent Engine
gemini_model.generate_content()   → Vertex AI Gemini Pro API
function declarations             → MCP Toolbox schema registration
TOOL_REGISTRY dispatch            → MCP Toolbox remote tool execution
session.history                   → Vertex AI Session Service (state)
SystemInstruction                 → Agent Platform system prompt config

Function Calling Flow (Agentic Loop):
  ┌──────────────┐     user_msg     ┌─────────────────┐
  │  FastAPI      │ ──────────────►  │  GeminiAgent     │
  │  /chat        │                  │  .chat()         │
  └──────────────┘                  └────────┬─────────┘
                                             │  generate_content()
                                             ▼
                                    ┌─────────────────┐
                                    │  Gemini Pro      │
                                    │  (Vertex AI)     │
                                    └────────┬─────────┘
                                             │  FunctionCall
                                             ▼
                                    ┌─────────────────┐
                                    │  gcp_toolbox     │
                                    │  TOOL_REGISTRY   │
                                    └────────┬─────────┘
                                             │  tool result
                                             ▼
                                    ┌─────────────────┐
                                    │  Gemini Pro      │  → text response
                                    │  (w/ tool ctx)   │
                                    └─────────────────┘
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.config import Settings, get_settings
from app.gcp_toolbox import TOOL_REGISTRY

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the Gemini Agent
# In Vertex AI Agent Builder, this maps to the Agent's "Instructions" field.
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTION = """
You are a helpful AI travel assistant powered by Google Vertex AI and AlloyDB.
You have access to real-time flight information and airport amenity data stored
in AlloyDB — Google's fully-managed, PostgreSQL-compatible cloud database with
integrated pgvector support for semantic search.

Your capabilities:
- Search for flights between airports using the operational AlloyDB flights table
- Find airport amenities using semantic vector search (RAG) over Vertex AI embeddings
- Look up airport information and available routes

Guidelines:
- Always use tools to retrieve live data from AlloyDB rather than guessing.
- When the user mentions a city, infer the most likely IATA code (e.g., San Francisco → SFO).
- For amenity questions, use semantic search — the user doesn't need to use exact keywords.
- Present flight results in a clear, scannable format with times and prices.
- Be concise, helpful, and proactively suggest related information.
- If no flights or amenities are found, say so clearly and suggest alternatives.
- Format prices in USD, durations as "Xh Ym", and times in a readable format.

You are running in local development mode with a PostgreSQL + pgvector replica of AlloyDB.
In production, this stack connects to Cloud AlloyDB with ScaNN ANN indexing.
""".strip()


# ---------------------------------------------------------------------------
# Session: stores chat history for a single conversation
# ---------------------------------------------------------------------------
@dataclass
class ChatSession:
    """
    Represents an active Agent Platform conversation session.

    In production Vertex AI Agent Engine, sessions are managed by the
    Vertex AI Session Service and can be persisted to Cloud Spanner.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0
    messages_exchanged: int = 0

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "parts": [{"text": content}]})
        self.messages_exchanged += 1

    def add_model_message(self, content: str) -> None:
        self.history.append({"role": "model", "parts": [{"text": content}]})

    def add_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Record a tool call result in the conversation history."""
        self.history.append({
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": tool_name,
                    "response": result,
                }
            }]
        })
        self.tool_calls_made += 1


# ---------------------------------------------------------------------------
# In-memory session store (maps session_id → ChatSession)
# In production: Cloud Firestore / Vertex AI Session Service
# ---------------------------------------------------------------------------
_SESSIONS: dict[str, ChatSession] = {}


def get_or_create_session(session_id: str | None = None) -> ChatSession:
    """Return an existing session or create a new one."""
    if session_id and session_id in _SESSIONS:
        return _SESSIONS[session_id]
    session = ChatSession(session_id=session_id or str(uuid.uuid4()))
    _SESSIONS[session.session_id] = session
    return session


# ---------------------------------------------------------------------------
# Gemini Tool declarations
# ---------------------------------------------------------------------------

def _build_gemini_tools() -> list[dict[str, Any]]:
    """
    Build Gemini function_declarations from the gcp_toolbox registry.

    In the Vertex AI MCP Toolbox, tool schemas are registered via YAML
    and auto-converted to function declarations. Here we define them
    explicitly as JSON Schema objects compatible with the Gemini API.

    Reference:
        https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling
    """
    return [
        {
            "function_declarations": [
                {
                    "name": "search_flights",
                    "description": (
                        "Search for available flights between two airports. "
                        "Use this when the user wants to find flights, check flight availability, "
                        "compare prices, or plan travel between two cities."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "departure_airport": {
                                "type": "string",
                                "description": "IATA 3-letter departure airport code (e.g. 'SFO', 'LAX', 'JFK').",
                            },
                            "arrival_airport": {
                                "type": "string",
                                "description": "IATA 3-letter arrival airport code (e.g. 'LAX', 'ORD', 'ATL').",
                            },
                            "date": {
                                "type": "string",
                                "description": "Optional departure date in YYYY-MM-DD format.",
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Optional maximum ticket price in USD.",
                            },
                        },
                        "required": ["departure_airport", "arrival_airport"],
                    },
                },
                {
                    "name": "search_airport_amenities",
                    "description": (
                        "Search for airport amenities — restaurants, lounges, shops, services — "
                        "using semantic vector search powered by Vertex AI embeddings and AlloyDB pgvector. "
                        "Use this for questions about food, lounges, shopping, WiFi, charging, "
                        "accessibility, transportation, or anything experiential at an airport."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Natural language description of what the user is looking for, "
                                    "e.g. 'quiet lounge with Wi-Fi and food', 'coffee and pastries before 6am'."
                                ),
                            },
                            "airport_iata": {
                                "type": "string",
                                "description": "Optional IATA code to restrict search to one airport.",
                            },
                            "category": {
                                "type": "string",
                                "enum": ["dining", "lounge", "retail", "services", "transportation", "accessibility", "entertainment"],
                                "description": "Optional category filter.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default 5, max 20).",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_airport_info",
                    "description": (
                        "Look up general information about a specific airport by its IATA code. "
                        "Returns name, city, country, timezone, and coordinates. "
                        "Use when the user asks about a specific airport or wants to know where it is."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "iata_code": {
                                "type": "string",
                                "description": "3-letter IATA airport code (e.g. 'SFO', 'JFK', 'LAX').",
                            },
                        },
                        "required": ["iata_code"],
                    },
                },
                {
                    "name": "list_available_routes",
                    "description": (
                        "List all city-pair routes with scheduled flights in the database. "
                        "Use when the user asks what destinations are available, what routes exist, "
                        "or which airports are served."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            ]
        }
    ]


# ---------------------------------------------------------------------------
# GeminiAgent — core orchestration class
# ---------------------------------------------------------------------------

class GeminiAgent:
    """
    Vertex AI Agent Platform — Gemini Pro orchestration engine.

    Manages the agentic loop: send user message → receive tool calls →
    execute tools against AlloyDB → feed results back to Gemini →
    return final natural-language response.
    """

    MAX_TOOL_ITERATIONS = 8  # Prevent infinite agentic loops

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._gemini = self._init_gemini()
        self._tools = _build_gemini_tools()
        log.info(
            "GeminiAgent initialized",
            model=self.settings.gemini_model,
            tools=list(TOOL_REGISTRY.keys()),
        )

    def _init_gemini(self) -> Any:
        """
        Initialize the Gemini client via google-generativeai SDK.

        In production: uses Vertex AI backend with IAM / ADC auth.
        Locally: falls back to Google AI Studio API key if set.
        """
        try:
            import google.generativeai as genai
            import os

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                log.info("Gemini initialized via Google AI Studio (GOOGLE_API_KEY)")
            else:
                # Use Vertex AI ADC (Application Default Credentials)
                # In production: service account / Workload Identity
                import vertexai
                vertexai.init(
                    project=self.settings.google_cloud_project,
                    location=self.settings.vertex_ai_location,
                )
                log.info(
                    "Gemini initialized via Vertex AI ADC",
                    project=self.settings.google_cloud_project,
                    location=self.settings.vertex_ai_location,
                )

            return genai.GenerativeModel(
                model_name=self.settings.gemini_model,
                system_instruction=_SYSTEM_INSTRUCTION,
            )

        except ImportError as exc:
            log.warning(
                "google-generativeai not installed. Running in mock mode.",
                error=str(exc),
            )
            return None

    def chat(self, user_message: str, session: ChatSession) -> str:
        """
        Execute one turn of the agentic conversation loop.

        Sends the user message to Gemini, handles tool calls autonomously,
        and returns the final natural-language response string.

        Args:
            user_message: The raw user input from the /chat API endpoint.
            session: The ChatSession containing conversation history.

        Returns:
            Final model response string after all tool calls are resolved.
        """
        session.add_user_message(user_message)

        if self._gemini is None:
            return self._mock_response(user_message, session)

        try:
            # Build Gemini conversation history (exclude function_response parts
            # from the raw history format into the Gemini Content format)
            gemini_history = self._build_gemini_history(session)

            # Agentic loop
            for iteration in range(self.MAX_TOOL_ITERATIONS):
                log.debug("Gemini agentic loop iteration", iteration=iteration)

                response = self._gemini.generate_content(
                    gemini_history,
                    tools=self._tools,
                    generation_config={
                        "temperature": 0.2,   # Lower temperature for factual tool-augmented responses
                        "top_p": 0.95,
                        "max_output_tokens": 2048,
                    },
                )

                candidate = response.candidates[0]

                # Check for function call
                function_call = self._extract_function_call(candidate)
                if function_call:
                    tool_name = function_call["name"]
                    tool_args = function_call["args"]

                    log.info("Gemini selected tool", tool=tool_name, args=tool_args)

                    # Execute the tool from gcp_toolbox
                    tool_result = self._execute_tool(tool_name, tool_args)

                    # Append function call + result to Gemini history
                    gemini_history.append({
                        "role": "model",
                        "parts": [{"function_call": {"name": tool_name, "args": tool_args}}],
                    })
                    gemini_history.append({
                        "role": "user",
                        "parts": [{"function_response": {"name": tool_name, "response": tool_result}}],
                    })

                    session.add_tool_result(tool_name, tool_result)
                    continue

                # No function call → Gemini has final text response
                final_text = candidate.content.parts[0].text
                session.add_model_message(final_text)
                return final_text

            # Exceeded max iterations
            fallback = "I've gathered the information I need. Please let me know what else you'd like to know."
            session.add_model_message(fallback)
            return fallback

        except Exception as exc:
            log.error("GeminiAgent.chat failed", error=str(exc), exc_info=True)
            error_msg = (
                f"I encountered an error while processing your request: {exc}. "
                "Please check your Vertex AI configuration and AlloyDB connection."
            )
            session.add_model_message(error_msg)
            return error_msg

    def _build_gemini_history(self, session: ChatSession) -> list[dict[str, Any]]:
        """Convert session history to Gemini API Content format."""
        # For the current turn, the last user message is sent fresh
        # Earlier history is included for context
        return [
            msg for msg in session.history
            if "function_response" not in str(msg.get("parts", [{}][0]))
               or msg["role"] == "user"
        ]

    def _extract_function_call(self, candidate: Any) -> dict[str, Any] | None:
        """Extract function call data from a Gemini candidate, if present."""
        try:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    return {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args),
                    }
        except (AttributeError, IndexError):
            pass
        return None

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a Gemini tool call to the corresponding gcp_toolbox function.

        This is the MCP Toolbox execution layer. In production Vertex AI
        Agent Engine, this dispatch is handled by the MCP Toolbox server,
        which routes calls to registered remote tools over gRPC.
        """
        if tool_name not in TOOL_REGISTRY:
            log.warning("Unknown tool called by Gemini", tool=tool_name)
            return {"error": f"Tool '{tool_name}' is not registered in the MCP Toolbox."}

        tool_fn = TOOL_REGISTRY[tool_name]
        try:
            result = tool_fn(**tool_args)
            log.info("Tool executed successfully", tool=tool_name)
            return result
        except Exception as exc:
            log.error("Tool execution failed", tool=tool_name, error=str(exc))
            return {"error": str(exc)}

    def _mock_response(self, user_message: str, session: ChatSession) -> str:
        """
        Fallback mock response when google-generativeai is not installed.
        Demonstrates the tool calling flow without a live Gemini API.
        """
        msg_lower = user_message.lower()

        if any(w in msg_lower for w in ["flight", "fly", "depart", "arrive"]):
            result = TOOL_REGISTRY["search_flights"]("SFO", "LAX")
            response = (
                f"[MOCK MODE — Install google-generativeai for real Gemini responses]\n\n"
                f"I found {result['flights_found']} flights. "
                f"Here's the tool result:\n{json.dumps(result, indent=2, default=str)}"
            )
        elif any(w in msg_lower for w in ["lounge", "restaurant", "food", "eat", "coffee", "amenity", "shop"]):
            result = TOOL_REGISTRY["search_airport_amenities"](user_message)
            response = (
                f"[MOCK MODE — Install google-generativeai for real Gemini responses]\n\n"
                f"Found {result['amenities_found']} amenities. "
                f"Tool result:\n{json.dumps(result, indent=2, default=str)}"
            )
        else:
            response = (
                "[MOCK MODE] I'm the AlloyDB + Vertex AI travel assistant. "
                "Ask me about flights (e.g. 'flights from SFO to LAX') or "
                "airport amenities (e.g. 'good restaurants at JFK'). "
                "Install google-generativeai and set GOOGLE_API_KEY for full Gemini responses."
            )

        session.add_model_message(response)
        return response
