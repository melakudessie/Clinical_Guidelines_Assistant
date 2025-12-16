import os
import re
import io
import sys
from typing import List, Dict, Tuple, Optional

import streamlit as st
import numpy as np

# --- Dependency Checks for Railway Logs ---
try:
    from openai import OpenAI
except ImportError:
    print("CRITICAL: 'openai' library not found. Add it to requirements.txt")
    sys.exit(1)

try:
    import faiss  # requires 'faiss-cpu' in requirements.txt
except ImportError:
    faiss = None
    print("WARNING: 'faiss' not found. Search will fail.")

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    print("WARNING: 'pypdf' not found. PDF reading will fail.")

# --- Constants ---
APP_TITLE: str = "WHO Antibiotic Guide"
APP_SUBTITLE: str = "AWaRe (Access, Watch, Reserve) Clinical Assistant"
DEFAULT_PDF_PATH: str = "WHOAMR.pdf"

EMBED_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-4o-mini"

STEWARD_FOOTER: str = (
    "Stewardship note: use the narrowest effective antibiotic; reassess at 48 to 72 hours; "
    "follow local guidance and clinical judgment."
)

WHO_SYSTEM_PROMPT: str = """
You are the WHO Antibiotic Guide, an AWaRe (Access, Watch, Reserve) Clinical Assistant.

Purpose: Support rational antibiotic use and antimicrobial stewardship using ONLY the provided WHO AWaRe book context.

Safety rules:
1. Use ONLY the provided WHO context - do not use outside knowledge.
2. If the answer is not explicitly supported by the context, say: "I couldn't find specific information about this in the WHO AWaRe handbook provided."
3. Only recommend avoiding antibiotics if the WHO context explicitly states antibiotics are not needed.

Response format:
**Main Answer:** Direct, concise answer (2-3 sentences). Include AWaRe category.
**Treatment Details:** (If applicable) Dosing, Route, Frequency, Duration.
**When Antibiotics Are NOT Needed:** (If applicable) Clear statement and justification.
**Sources:** Cite page numbers.

Always end with:
Stewardship note: use the narrowest effective antibiotic; reassess at 48 to 72 hours; follow local guidance and clinical judgment.
""".strip()

# --- App Config ---
st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")
st.title(f"💊 {APP_TITLE}")
st.caption(APP_SUBTITLE)

# --- Helper Functions ---

def ensure_footer(text: str) -> str:
    if not text: return STEWARD_FOOTER
    if STEWARD_FOOTER.lower() in text.lower(): return text
    return (text.rstrip() + "\n\n" + STEWARD_FOOTER).strip()

def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _read_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[Dict]:
    bio = io.BytesIO(pdf_bytes)
    reader = PdfReader(bio)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": _clean_text(text)})
    return pages

def _chunk_pages(pages: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    chunks = []
    for p in pages:
        text = p["text"]
        if not text: continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            if chunk: chunks.append({"page": p["page"], "text": chunk})
            if end >= n: break
            start = max(0, end - chunk_overlap)
    return chunks

def _embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    vectors = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(vectors).astype(np.float32)

def _build_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def _search(index, client, query: str, chunks: List[Dict], k: int) -> List[Dict]:
    qvec = _embed_texts(client, [query])
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, k)
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1: continue
        c = chunks[int(idx)]
        hits.append({"score": float(score), "page": c["page"], "text": c["text"]})
    return hits

def _make_context(hits: List[Dict], max_chars: int = 1500) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        excerpt = h["text"]
        if len(excerpt) > max_chars: excerpt = excerpt[:max_chars].rstrip() + " ..."
        blocks.append(f"[Source {i}, Page {h['page']}]:\n{excerpt}")
    return "\n\n".join(blocks)

def _get_openai_key_from_env() -> str:
    # 1. Try Railway Env Var
    key = os.environ.get("OPENAI_API_KEY", "")
    # 2. Try Streamlit Secrets (Local dev)
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            pass
    return key.strip()

def _get_pdf_bytes_from_repo(local_path: str) -> Tuple[str, Optional[bytes], str]:
    if os.path.exists(local_path):
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            return f"repo:{local_path}:{len(data)}", data, f"Using PDF: {local_path}"
        except Exception as e:
            return "repo:error", None, f"Error reading PDF: {e}"
    
    # Debugging for Railway: If file missing, show what IS there
    cwd = os.getcwd()
    files = os.listdir(cwd)
    return "repo:missing", None, f"PDF not found at '{local_path}'. Current Dir '{cwd}' contains: {files}"

@st.cache_resource(show_spinner=True)
def build_retriever(pdf_bytes: bytes, openai_api_key: str):
    client = OpenAI(api_key=openai_api_key)
    pages = _read_pdf_pages_from_bytes(pdf_bytes)
    chunks = _chunk_pages(pages, 1500, 200)
    vectors = _embed_texts(client, [c["text"] for c in chunks])
    index = _build_index(vectors)
    return {"chunks": chunks, "index": index}

# --- Main UI ---
with st.sidebar:
    st.header("⚙️ Config")
    
    # API Key Logic
    api_key = _get_openai_key_from_env()
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.warning("⚠️ API Key missing. Add OPENAI_API_KEY to Railway Variables.")
    else:
        st.success("✅ API Key loaded from Environment")

    # PDF Logic
    pdf_key, pdf_bytes, pdf_status = _get_pdf_bytes_from_repo(DEFAULT_PDF_PATH)
    st.info(pdf_status)  # Shows file status/debug info

    if not pdf_bytes:
        st.error("STOP: PDF missing. Upload 'WHOAMR.pdf' to your GitHub repo root.")
        st.stop()

    if not faiss or not PdfReader:
        st.error("Missing dependencies. Ensure requirements.txt has 'faiss-cpu' and 'pypdf'.")
        st.stop()

    # Build Retriever
    if api_key and pdf_bytes:
        try:
            resources = build_retriever(pdf_bytes, api_key)
            st.success(f"Indexed {len(resources['chunks'])} chunks")
        except Exception as e:
            st.error(f"Index failed: {e}")
            resources = None
    else:
        resources = None

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about antibiotic treatments...")

if question:
    if not resources or not api_key:
        st.error("System not ready. Check Sidebar config.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            client = OpenAI(api_key=api_key)
            hits = _search(resources["index"], client, question, resources["chunks"], k=5)
            
            if not hits:
                resp = "I couldn't find information on that in the WHO AWaRe book."
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                context_str = _make_context(hits)
                # Streaming Response
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": WHO_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{question}"}
                    ],
                    stream=True
                )
                response_text = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
