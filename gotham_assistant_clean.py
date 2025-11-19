import os
import re
import json
import base64
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

# ---------- Personal profile (updated: no longer a student) ----------
PROFILE = {
    "headline": "Data & AI professional with a foundation in research, analytics, and technology-driven strategy.",
    "roles": [
        "Research Consultant at Reach3 Insights",
        "Co-founder of an e-commerce brand called Nick&Jess, which ranked #10 on Amazon India in 2019"
    ],
    "specialties": [
        "Data Analysis", "AI & Automation", "Market Research",
        "Visualization", "Product Strategy"
    ],
    "skills": [
        "Python", "Streamlit", "SQL", "Power BI",
        "Machine Learning", "Q Research Software"
    ],
    "education": [
        "Applied AI Solutions Development – George Brown College (2025)",
        "Data Science Certificate – Brain Station (2023)",
        "Research Analyst Program – Humber College (2022)",
        "B.Com (Financial Markets) – Mumbai University (2017)"
    ],
   "entrepreneurial_experience": [
    "Co-founded an e-commerce brand that achieved a #10 ranking on Amazon IN, sharpening expertise in market dynamics, pricing strategy, and automation.",
    "Exited the business after COVID-19 due to limited experience scaling operations, transitioning into analytics and research to build deeper expertise."
],
"leadership_experience": [
    "At age 17, served as President of the Rotaract Club of his college, leading a 150+ member team.",
    "Organized and led multiple community service and skill-building initiatives during his tenure.",
    "Received the awards for Best President and Best Rotaract Club in the same year — a rare distinction.",
    "These experiences laid the foundation for his leadership style — collaborative, empathetic, and results-oriented.",
    "Later, as co-founder of Nick&Jess, he applied these skills to guide the brand to a #10 ranking on Amazon India."
],
"achievements": [
    "Co-founded an e-commerce brand that achieved a #10 ranking on Amazon IN.",
    "Recognized for leadership excellence and impact in youth community programs.",
    "Delivered multiple analytics projects driving business-level insights."
],
    "positioning": [
        "Combines business acumen with analytical depth to turn complex data into clear, actionable decisions."
    ],
    "focus": "Building scalable, AI-driven systems that deliver measurable impact.",
    "hobbies_and_interests": [
    "Watching and playing Cricket, Football, and Pickleball.",
    "Practicing kickboxing as a fun workout and stress reliever.",
    "A passionate supporter of Liverpool FC.",
    "Learning and solving Rubik’s Cubes — best time: 2 minutes 12 seconds on a 3x3 cube; planning to master the 4x4 next."
],
    "contact": {
        "linkedin": "https://www.linkedin.com/in/gothamtikyani/",
        "email": "[Reach out on LinkedIn for opportunities or collaborations.]"
    }
}



def _join(arr):
    return ", ".join(arr) if arr else ""

def build_profile_block(p: dict) -> str:
    def j(v):
        if isinstance(v, list):
            return ", ".join(v)
        return v or ""
    contact = p.get("contact", {})
    return (
        f"Headline: {j(p.get('headline'))}\n"
        f"Roles: {j(p.get('roles', []))}\n"
        f"Specialties: {j(p.get('specialties', []))}\n"
        f"Core skills: {j(p.get('skills', []))}\n"
        f"Education: {j(p.get('education', []))}\n"
        f"Entrepreneurial experience: {j(p.get('entrepreneurial_experience', []))}\n"
        f"Leadership experience: {j(p.get('leadership_experience', []))}\n"
        f"Achievements: {j(p.get('achievements', []))}\n"
        f"Positioning: {j(p.get('positioning', []))}\n"
        f"Focus: {j(p.get('focus'))}\n"
        f"Hobbies: {j(p.get('hobbies_and_interests', []))}\n"
        f"LinkedIn: {contact.get('linkedin','')}\n"
        f"Contact note: {contact.get('email','')}"
    )



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
    "You are Gotham Assistant — a personal AI version of Gotham Tikyani. "
    "Your job is to present Gotham’s background, skills, projects, and achievements as an interactive résumé. "
    "Be friendly, confident, and concise (1-2 sentences unless asked for more). "
    "If someone asks how to contact Gotham, share his LinkedIn profile from the provided data. "
    "Never output placeholders like [insert ...] or TODOs; if info is missing, ask one brief clarifying question instead. "
    "When asked about any questions at the start, keep it vague and conversational which will lead to probing"
    "When asked about leadership or teamwork, include examples from his Rotaract presidency and entrepreneurial experience. "
    "When someone says 'Tell me about Gotham', treat it as the person, not the city from comics."
    "Also, when asked about hobbies or interests, reference Gotham’s sports, fitness, and puzzle-solving activities from his profile."
)


# ---------- Output cleaner to remove placeholders + avoid verbatim repeats ----------
PLACEHOLDER_RE = re.compile(r"\[insert[^]]*\]", re.IGNORECASE)
BLANKS_RE = re.compile(r"\n{3,}")

def clean_response(txt: str, last_answer: str | None) -> str:
    if not isinstance(txt, str):
        return txt
    txt = PLACEHOLDER_RE.sub("", txt)
    txt = BLANKS_RE.sub("\n\n", txt).strip()
    if last_answer and txt.strip() == last_answer.strip():
        txt = ("Here’s a quicker angle:\n"
               "- I’m Gotham’s AI résumé.\n"
               "- I showcase skills, projects, and how I solve problems.\n"
               "- Ask about background, projects, or an example deliverable.")
    return txt

def build_api_messages(user_text: str, context_block: str, template_msg: str, history: List[Dict[str, str]]):
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Always include the personal profile so the model has concrete facts
    messages.append({"role": "system", "content": "Personal profile:\n" + build_profile_block(PROFILE)})

    if context_block:
        messages.append({"role": "system", "content": f"Context for the task:\n{context_block}"})
    if template_msg:
        messages.append({"role": "system", "content": f"Follow this format when applicable:\n{template_msg}"})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages

def stream_chat_completion(messages: List[Dict[str, str]]) -> str:
    """Stream from the API but buffer text; return a single final string (no double rendering)."""
    try:
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=True,
            temperature=st.session_state.get("temperature", 0.5),
        )
        chunks = []
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                chunks.append(delta.content)
        return "".join(chunks)
    except Exception as e:
        st.error("Issue talking to the model. Please try again.")
        st.caption(str(e))
        return ""

# =====================
# UI
# =====================
st.set_page_config(page_title=APP_TITLE, page_icon="🤖")
st.title(APP_TITLE)

# Friendly intro on first load
st.session_state.setdefault("messages", [])
if not st.session_state["messages"]:
    st.info("👋 I’m Gotham's Assistant — an interactive assistant. Ask about my background, skills, projects, or how I solve problems.")


with st.sidebar:
    st.subheader("Settings")
    # Groq-supported models (as of late 2025)
    st.session_state.setdefault("openai_model", "llama-3.1-8b-instant")
    st.session_state["openai_model"] = st.selectbox(
        "Model",
        options=[
            "llama-3.1-8b-instant",    # fast default
            "llama-3.3-70b-versatile", # larger, richer
            "gemma2-9b-it"             # smaller instruction-tuned
        ],
        index=0,
    )
    st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.5))
    anonymize_clients = st.toggle("Anonymize known clients", value=True, help="Redacts configured client names")
    cap_chars = st.number_input("Character cap (0 = off)", min_value=0, max_value=100000, value=0, step=100)

    if st.button("Reset chat", type="secondary"):
        st.session_state["messages"] = []
        st.rerun()

    st.caption("Known clients loaded from secrets: KNOWN_CLIENTS")


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

# ... previous code omitted for brevity ...

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    history_only = [m for m in st.session_state["messages"] if m["role"] in ("user", "assistant")]
    trimmed_history = history_only[-MAX_HISTORY_TURNS:]

    api_messages = build_api_messages(
        user_text=user_text,
        context_block=context_block,
        template_msg=template_msg,
        history=trimmed_history
    )

    with st.chat_message("assistant"):
        raw = stream_chat_completion(api_messages) or ""

        if anonymize_clients:
            raw = anonymize_known_clients(raw, on=True)
        raw = redact_pii(raw)
        raw = soft_clip(raw, cap_chars)

        last_answer = next((m["content"] for m in reversed(st.session_state["messages"]) if m["role"] == "assistant"), "")
        response_text = clean_response(raw, last_answer)

        st.markdown(response_text)

        st.download_button(
            label="Download reply (.md)",
            data=response_text.encode("utf-8"),
            file_name="gotham_assistant_reply.md",
            mime="text/markdown",
        )

    # 👇 this should align with "with st.chat_message(...)"
    st.session_state["messages"].extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": response_text},
    ])

# =====================
# Disclaimer Footer
# =====================
st.markdown(
    """
    <hr style="margin-top:2em; margin-bottom:0.5em;">
    <div style="text-align:center; font-size:0.85em; color:gray;">
        ⚠️ <b>Disclaimer:</b> Gotham Assistant is an AI representation of <b>Gotham Tikyani</b>.
        While all responses are generated from verified information about his background, experience, and public profile - there could be some discrepancies.
        For official communication or professional inquiries, please connect via
        <a href="https://www.linkedin.com/in/gothamtikyani/" target="_blank">LinkedIn</a>.
    </div>
    """,
    unsafe_allow_html=True
)

