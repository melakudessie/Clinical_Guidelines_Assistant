import os
import re
import io
import sys
import glob
from typing import List, Dict, Tuple, Optional

import streamlit as st
import numpy as np

# --- Dependency Checks ---
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
APP_TITLE: str = "Clinical Guidelines Assistant"
APP_SUBTITLE: str = "RAG Support for Multiple Medical Documents"
EMBED_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-4o-mini"

SYSTEM_PROMPT: str = """
You are a Clinical Assistant answering questions based ONLY on the provided medical guidelines.

Purpose: Support clinical decision-making using strictly the retrieved context.

Safety rules:
1. Use ONLY the provided context - do not use outside knowledge.
2. If the answer is not in the context, say: "I couldn't find specific information about this in the provided documents."
3. Always cite the specific document and page number when possible.

Response format:
- **Answer:** Direct, concise medical guidance.
- **Source:** Explicitly mention which document the info comes from.
- **Stewardship:** Remind to follow local protocols.
""".strip()

st.set_page_config(page_title=APP_TITLE, page_icon="🏥", layout="wide")
st.title(f"🏥 {APP_TITLE}")
st.caption(APP_SUBTITLE)

# --- Helper Functions ---

def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _read_pdf_file(file_path: str) -> List[Dict]:
    """Reads a single PDF and returns a list of pages with metadata."""
    try:
        reader = PdfReader(file_path)
        pages = []
        filename = os.path.basename(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "page": i + 1, 
                    "text": _clean_text(text), 
                    "source": filename
                })
        return pages
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def _chunk_pages(pages: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            if chunk: 
                chunks.append({
                    "page": p["page"], 
                    "text": chunk, 
                    "source": p["source"]
                })
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
        hits.append({
            "score": float(score), 
            "page": c["page"], 
            "text": c["text"],
            "source": c["source"]
        })
    return hits

def _make_context(hits: List[Dict], max_chars: int = 2000) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        excerpt = h["text"]
        # Include Filename and Page in the context block
        header = f"[Doc: {h['source']}, Page: {h['page']}]"
        if len(excerpt) > 500: excerpt = excerpt[:500] + "..."
        blocks.append(f"{header}\n{excerpt}")
    return "\n\n".join(blocks)

def _get_openai_key() -> str:
    # 1. Railway Env
    key = os.environ.get("OPENAI_API_KEY", "")
    # 2. Streamlit Secrets (for local testing)
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            pass
    return key.strip()

@st.cache_resource(show_spinner=True)
def load_and_index_documents(openai_api_key: str):
    client = OpenAI(api_key=openai_api_key)
    
    # Scan for all PDFs in current directory
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        return None, "No PDF files found in the repository root."

    all_pages = []
    status_msg = f"Found {len(pdf_files)} documents: {', '.join([os.path.basename(f) for f in pdf_files])}"
    
    for pdf in pdf_files:
        pages = _read_pdf_file(pdf)
        all_pages.extend(pages)

    if not all_pages:
        return None, "PDFs found but no text could be extracted (are they scanned images?)."

    chunks = _chunk_pages(all_pages, 1000, 200)
    vectors = _embed_texts(client, [c["text"] for c in chunks])
    index = _build_index(vectors)
    
    return {"chunks": chunks, "index": index, "files": pdf_files}, status_msg

# --- Main UI ---
with st.sidebar:
    st.header("⚙️ Setup")
    
    api_key = _get_openai_key()
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.warning("⚠️ Add OPENAI_API_KEY to Railway Variables.")
    else:
        st.success("✅ API Key Configured")

    st.divider()
    st.markdown("**📂 Documents**")
    
    # Initialization
    if api_key and PdfReader and faiss:
        with st.spinner("Indexing documents..."):
            try:
                resources, msg = load_and_index_documents(api_key)
                if resources:
                    st.success(f"✅ Ready! Indexed {len(resources['chunks'])} segments.")
                    with st.expander("View Loaded Files"):
                        for f in resources["files"]:
                            st.text(f"• {os.path.basename(f)}")
                else:
                    st.error(f"❌ {msg}")
                    # Debug helper for Railway
                    st.info(f"Current Directory: {os.getcwd()}")
                    st.info(f"Files present: {os.listdir('.')}")
            except Exception as e:
                st.error(f"Index Error: {e}")
                resources = None
    else:
        resources = None
        st.info("Waiting for API Key and dependencies...")

# Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a clinical question...")

if question:
    if not resources:
        st.error("⚠️ Documents not indexed. Check Sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            client = OpenAI(api_key=api_key)
            hits = _search(resources["index"], client, question, resources["chunks"], k=5)
            
            if not hits:
                resp = "I couldn't find information about that in the provided documents."
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                context_str = _make_context(hits)
                
                # Stream Response
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{question}"}
                    ],
                    stream=True
                )
                response_text = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Show Sources
                with st.expander("📚 Sources Used"):
                    for h in hits:
                        st.markdown(f"**{h['source']}** (Page {h['page']})")
                        st.caption(h['text'][:200] + "...")
