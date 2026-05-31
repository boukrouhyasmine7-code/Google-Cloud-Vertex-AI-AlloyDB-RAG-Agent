import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Cymbal Air - Travel Assistant", page_icon="✈️", layout="centered")

# 2. Inject the Corrected Custom Theme CSS (Fixing Text Contrast)
st.markdown(
    """
    <style>
    /* Force main application background canvas to pure white */
    .stApp {
        background-color: #ffffff !important;
        color: #31333f !important;
    }

    /* The signature Streamlit horizontal pink/coral accent line at the top */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background-color: #ff4b4b !important;
        z-index: 999999;
    }

    /* Target standard headers and basic text markdown blocks */
    h1, h2, h3, h4, .stMarkdown p, p {
        color: #31333f !important;
        font-family: 'Source Sans 3', 'Inter', sans-serif;
    }

    /* Keep page subtitling clear and slightly muted */
    .stCaption {
        color: #555867 !important;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Format standard assistant chat container bubbles */
    [data-testid="stChatMessage"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e6e9ef !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
    }

    /* Target every alternating bubble (the User turn) to apply a soft, warm pink background */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #fff4f4 !important;
        border: 1px solid #fdd2d2 !important;
    }

    /* CRITICAL FIX: Ensure tables, data frames, and markdown data rows inside the chat read dark charcoal black */
    [data-testid="stChatMessage"] table, 
    [data-testid="stChatMessage"] tr, 
    [data-testid="stChatMessage"] td, 
    [data-testid="stChatMessage"] th,
    .stDataFrame, div[data-testid="stMarkdownContainer"] {
        color: #31333f !important;
    }

    /* Style the main bottom text input box wrapper frame */
    div[data-baseweb="input"] {
        background-color: #f8f9fa !important;
        border: 1px solid #ccd0d5 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Give the text input an accent glow when clicked into */
    div[data-baseweb="input"]:focus-within {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 0 1px #ff4b4b !important;
    }

    /* Ensure text inside input reads crisp charcoal black */
    div[data-baseweb="input"] input {
        color: #31333f !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Clean White & Pink Layout Title Headers
st.title("✈️ Cymbal Air")
st.caption("Vertex AI Agent Platform • AlloyDB RAG Grounding Engine")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about flights, routes, or airport amenities..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call your FastAPI backend process
    BACKEND_URL = "http://127.0.0.1:8080/chat"
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = requests.post(BACKEND_URL, json={"message": prompt})
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", data.get("output", str(data)))
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Error: Backend returned status code {response.status_code}"
                message_placeholder.error(error_msg)
        except Exception as e:
            message_placeholder.error(f"Could not connect to FastAPI backend: {e}")