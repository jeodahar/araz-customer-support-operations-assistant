"""Daraz Support Operations Assistant.

Loads the pre-built FAISS index from ./faiss_index (never re-reads or
re-embeds the PDFs), retrieves relevant policy chunks, and answers with
Groq's openai/gpt-oss-120b. The API key comes from Streamlit secrets.
"""
import json
from pathlib import Path

import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------- config
APP_DIR = Path(__file__).parent
GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TOP_K = 5
MIN_SCORE = 0.20  # cosine similarity below this counts as "not found"

SECTION_LABELS = {
    "returns": "↩️ Returns",
    "delivery": "🚚 Delivery",
    "refunds": "💸 Refunds",
    "sellers": "🏪 Sellers",
    "payments": "💳 Payments",
    "customer_support": "🎧 Customer support",
}
ALL = "all"
AVATARS = {"user": "🧑‍💼", "assistant": "🛍️"}

STARTERS = [
    "How long do customers have to return an item?",
    "When is a refund issued for a cancelled order?",
    "What happens if a delivery attempt fails?",
    "Which payment methods can customers use?",
]

SYSTEM_PROMPT = (
    "You are the Daraz Support Operations Assistant, helping customer-support "
    "agents. Answer only from the policy excerpts provided in the user message. "
    "If the excerpts do not contain the answer, say so plainly and suggest which "
    "section might cover it. Never invent policies, timelines, fees or amounts. "
    "Be concise: lead with the answer, then list conditions or steps briefly. "
    "Cite excerpts by number like [1]. If excerpts conflict, point that out. "
    "Reply in the language the question was asked in."
)

st.set_page_config(
    page_title="Daraz Support Assistant",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');

html, body, .stApp, [class*="st-"] { font-family: 'Roboto', sans-serif; }
.stApp { background: #F5F5F5; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 820px; padding-top: 1.4rem; }

.brand-bar {
    background: #F85606; color: #fff; border-radius: 14px;
    padding: 18px 24px; display: flex; align-items: baseline;
    gap: 16px; flex-wrap: wrap;
}
.brand-bar .word { font-weight: 900; font-size: 2.1rem; letter-spacing: -0.04em; line-height: 1; }
.brand-bar .title { font-weight: 500; font-size: 1.05rem; }
.tagline { color: #6B6B6B; margin: 10px 2px 18px; font-size: 0.95rem; }

[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #EAEAEA; }
.side-title { font-weight: 700; color: #212121; margin: 4px 0 2px; }
.side-note { color: #6B6B6B; font-size: 0.85rem; }

[data-testid="stChatMessage"] {
    background: #FFFFFF; border: 1px solid #EAEAEA;
    border-radius: 14px; padding: 14px 16px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #FFF1EA; border-color: #FFD9C7;
}

.stButton > button {
    border-radius: 999px; border: 1px solid #F85606; color: #F85606;
    background: #FFFFFF; font-weight: 500; text-align: left;
}
.stButton > button:hover { background: #FFF1EA; border-color: #F85606; color: #D94A05; }
.stButton > button:focus-visible { outline: 2px solid #F85606; outline-offset: 2px; }

[data-testid="stExpander"] { background: #FAFAFA; border-radius: 10px; }
</style>
"""


# ---------------------------------------------------------------- loaders
def find_index_dir():
    """Folder containing index.faiss, wherever it sits under the app folder."""
    for f in sorted(APP_DIR.rglob("index.faiss")):
        if (f.parent / "metadata.json").exists() and (f.parent / "config.json").exists():
            return f.parent
    return None


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_kb(index_dir: str):
    """Load the prebuilt index, metadata and the query-embedding model."""
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    records = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    meta = {int(r["id"]): r for r in records}
    cfg = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(cfg["model"])  # same model used at ingest
    return index, meta, model


@st.cache_resource
def get_client():
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return Groq(api_key=key) if key else None


# ---------------------------------------------------------------- helpers
def section_label(key: str) -> str:
    if key == ALL:
        return "📚 All sections"
    return SECTION_LABELS.get(key, key.replace("_", " ").title())


def retrieve(kb, query: str, section: str, top_k: int):
    index, meta, model = kb
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    # Flat index: when filtering by section, rank everything then filter.
    fetch_k = index.ntotal if section != ALL else min(top_k, index.ntotal)
    scores, ids = index.search(q, fetch_k)
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1 or score < MIN_SCORE:
            break  # results are sorted, nothing better follows
        rec = meta[int(idx)]
        if section != ALL and rec["department"] != section:
            continue
        hits.append({**rec, "score": float(score)})
        if len(hits) == top_k:
            break
    return hits


def build_query(prompt: str, history: list) -> str:
    """Short follow-ups ("and for sellers?") borrow the previous question."""
    if len(prompt.split()) < 6:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return f"{prev[-1]} {prompt}"
    return prompt


def stream_answer(client: Groq, question: str, hits: list, history: list):
    context = "\n\n".join(
        f"[{n}] ({h['source_file']}, page {h.get('page', '?')})\n{h['text']}"
        for n, h in enumerate(hits, 1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]
    messages.append(
        {"role": "user", "content": f"Policy excerpts:\n{context}\n\nQuestion: {question}"}
    )
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=2048,
        reasoning_effort="low",
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def render_sources(hits: list):
    with st.expander(f"Sources ({len(hits)})"):
        for n, h in enumerate(hits, 1):
            st.markdown(
                f"**[{n}]** `{h['source_file']}`, page {h.get('page', '?')} "
                f"(relevance {h['score']:.2f})"
            )
            text = h["text"]
            st.caption(text[:320] + ("…" if len(text) > 320 else ""))


# ---------------------------------------------------------------- app
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="brand-bar"><span class="word">daraz</span>'
    '<span class="title">Support Operations Assistant</span></div>'
    '<p class="tagline">Answers come from Daraz policy documents, with the source shown.</p>',
    unsafe_allow_html=True,
)

index_dir = find_index_dir()  # not cached, so a miss is never remembered
kb = load_kb(str(index_dir)) if index_dir else None
if kb is None:
    st.error(
        "Knowledge base not found. The repo needs a folder containing "
        "`index.faiss`, `metadata.json` and `config.json` next to app.py."
    )
    seen = sorted(
        str(p.relative_to(APP_DIR))
        for p in APP_DIR.rglob("*")
        if p.is_file() and ".git" not in p.parts
    )
    with st.expander("Files the app can see"):
        st.code(f"App folder: {APP_DIR}\n\n" + ("\n".join(seen[:60]) or "(nothing)"))
    st.stop()

client = get_client()
if client is None:
    st.error(
        "Groq API key not found. Add `GROQ_API_KEY` to your Streamlit secrets "
        "(Settings → Secrets, or `.streamlit/secrets.toml` locally)."
    )
    st.stop()

_, meta, _ = kb
departments = {r["department"] for r in meta.values()}
ordered = [k for k in SECTION_LABELS if k in departments]
ordered += sorted(departments - set(ordered))

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown('<div class="side-title">Search in</div>', unsafe_allow_html=True)
    section = st.radio(
        "Search in",
        options=[ALL] + ordered,
        format_func=section_label,
        label_visibility="collapsed",
    )
    with st.expander("Search settings"):
        top_k = st.slider("Passages to use", 3, 10, DEFAULT_TOP_K)
    files = {r["source_file"] for r in meta.values()}
    st.markdown(
        f'<div class="side-note">{len(meta)} passages from {len(files)} documents</div>',
        unsafe_allow_html=True,
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

scope = "all sections" if section == ALL else section_label(section).split(" ", 1)[1]
prompt = st.chat_input(f"Ask about {scope}") or st.session_state.pop("pending", None)

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

if not st.session_state.messages and not prompt:
    st.markdown("**Try one of these**")
    cols = st.columns(2)
    for i, q in enumerate(STARTERS):
        if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

if prompt:
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Searching policies…"):
            hits = retrieve(kb, build_query(prompt, history), section, top_k)

        if not hits:
            answer = (
                f"I couldn't find anything about that in {scope}. "
                "Try rephrasing, or switch to All sections."
            )
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(stream_answer(client, prompt, hits, history))
            except Exception as exc:
                answer = "Couldn't get an answer from the AI service. Please try again."
                st.error(f"{answer} ({type(exc).__name__})")
            render_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )
