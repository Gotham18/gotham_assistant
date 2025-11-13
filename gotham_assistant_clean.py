import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Generator

import streamlit as st
from openai import OpenAI

# =====================
# App & Config
# =====================
APP_TITLE = "Gotham Assistant"
DATA_DIR = Path(".cache")
DATA_DIR.mkdir(exist_ok=True)

# =====================
# API Setup (Groq)
# =====================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Add it in Streamlit secrets or as an environment variable.")
    st.stop()

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# =====================
# Utilities
# =====================
def load_json(name: str):
    path = DATA_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_json(name: str, data: Dict[str, Any]):
    path = DATA_DIR / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def soft_clip(text: str, limit: int | None) -> str:
    if not isinstance(text, str) or not limit or limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut if cut else text[:limit]) + "…"

# ==============
# Privacy helpers
# ==============
DEFAULT_KNOWN_CLIENTS: List[str] = st.secrets.get("KNOWN_CLIENTS", [])
CLIENT_ALIAS = "Fortune 500 brand"

EMAIL_REGEX = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
PHONE_REGEX = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

def anonymize_known_clients(text: str, on: bool) -> str:
    if not on or not isinstance(text, str) or not DEFAULT_KNOWN_CLIENTS:
        return text
    out = text
    for name in DEFAULT_KNOWN_CLIENTS:
        if not name:
            continue
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        out = pattern.sub(CLIENT_ALIAS, out)
    return out

def redact_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = EMAIL_REGEX.sub("[email hidden]", text)
    text = PHONE_REGEX.sub("[phone hidden]", text)
    return text

# =====================
# Prompting
# =====================
SYSTEM_PROMPT = (
    "You are Gotham Assistant, a helpful analyst. "
    "Be concise, concrete, and cite data if provided. "
    "When anonymization is enabled, never reveal specific client names—replace them with 'Fortune 500 brand'."
)

def build_api_messages(user_text: str, context_block: str, template_msg: str, history: List[Dict[str, str]]):
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_block:
        messages.append({"role": "system", "content": f"Context for the task:\\n{context_block}"})
    if template_msg:
        messages.append({"role": "system", "content": f"Follow this format when applicable:\\n{template_msg}"})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages

def stream_chat_completion(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    try:
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=True,
            temperature=st.session_state.get("temperature", 0.3),
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error("Issue talking to the model. Please try again.")
        st.caption(str(e))
        return

# =====================
# UI
# =====================
st.set_page_config(page_title=APP_TITLE, page_icon="🤖")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Settings")
    st.session_state.setdefault("openai_model", "llama3-8b-8192")
    st.session_state["openai_model"] = st.selectbox(
    "Model",
    options=[
        "llama3-8b-8192",     # Fast and free, best default
        "mixtral-8x7b-32768", # Bigger, more powerful
        "gemma-7b-it"         # Smaller, instruction-tuned
    ],
    index=0,
)
    st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.3))
    anonymize_clients = st.toggle("Anonymize known clients", value=True, help="Redacts configured client names")
    cap_chars = st.number_input("Character cap (0 = off)", min_value=0, max_value=100000, value=0, step=100)

    if st.button("Reset chat", type="secondary"):
        st.session_state["messages"] = []
        st.rerun()

    st.caption("Known clients loaded from secrets: KNOWN_CLIENTS")

# Load/Init state
st.session_state.setdefault("messages", [])

# Optional side context/template
with st.expander("Optional: Add context for the model"):
    context_block = st.text_area("Context", placeholder="Paste notes, data points, or a brief.", height=120)
with st.expander("Optional: Response template"):
    template_msg = st.text_area("Template", placeholder="e.g., Overview → Insights → Next steps", height=100)

MAX_HISTORY_TURNS = 12

# Show chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input area
user_text = st.chat_input("Ask me anything…")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    history_only = [m for m in st.session_state["messages"] if m["role"] in ("user", "assistant")]
    trimmed_history = history_only[-MAX_HISTORY_TURNS:]

    api_messages = build_api_messages(user_text=user_text, context_block=context_block, template_msg=template_msg, history=trimmed_history)

    with st.chat_message("assistant"):
        response_chunks = stream_chat_completion(api_messages)
        response_text = st.write_stream(response_chunks) or ""

        if anonymize_clients:
            response_text = anonymize_known_clients(response_text, on=True)
        response_text = redact_pii(response_text)
        response_text = soft_clip(response_text, cap_chars)
        st.markdown(response_text)

        st.download_button(
            label="Download reply (.md)",
            data=response_text.encode("utf-8"),
            file_name="gotham_assistant_reply.md",
            mime="text/markdown",
        )

    st.session_state["messages"].extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": response_text},
    ])
