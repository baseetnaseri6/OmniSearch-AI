from __future__ import annotations

import re
import io
import os
import json
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# ---------- Optional deps with safe fallbacks ----------
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# ---------- Small utilities ----------
def _paper_key(meta: Dict[str, Any]) -> str:
    raw = (meta.get("doi") or "") + "|" + (meta.get("title") or "")
    return "smart_chat_" + hashlib.md5(raw.encode("utf-8")).hexdigest()


def _chunk_text(text: str, max_tokens: int = 800, overlap: int = 120) -> List[str]:
    """
    Simple chunker on paragraph/sentence-ish boundaries.
    Assumes ~4 chars/token roughness; adjust by size.
    """
    if not text:
        return []
    paras = re.split(r"\n{2,}", text)
    chunks = []
    buf = ""
    budget = max_tokens * 4
    ov_chars = overlap * 4
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= budget:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf)
            if chunks and overlap > 0:
                tail = chunks[-1][-ov_chars:]
                buf = (tail + "\n\n" + p).strip()
            else:
                buf = p
            while len(buf) > budget:
                chunks.append(buf[:budget])
                buf = buf[budget - ov_chars:]
    if buf:
        chunks.append(buf)
    return chunks


def _load_embedder(name: str) -> Optional[Any]:
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(name)
    except Exception:
        for n in ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"]:
            try:
                return SentenceTransformer(n)
            except Exception:
                continue
    return None


def _embed_texts(embedder, texts: List[str]) -> Optional[Any]:
    if embedder is None or np is None or not texts:
        return None
    try:
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype="float32")
    except Exception:
        return None


def _cosine_sim(a: Any, b: Any) -> Any:
    if np is None:
        return None
    return (a @ b.T)


def _bm25_rank(query: str, texts: List[str], top_k: int = 6) -> List[Tuple[int, float]]:
    if BM25Okapi is None or not texts:
        return []
    try:
        tokenized_corpus = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked
    except Exception:
        return []


def _select_context(query: str, chunks: List[str], embedder, chunk_vecs, top_k: int = 6) -> List[Tuple[int, float]]:
    """
    Hybrid selection: embeddings + BM25 (if available). Returns list of (idx, score).
    """
    candidates: Dict[int, float] = {}

    # Vector sim path
    if embedder is not None and chunk_vecs is not None and np is not None:
        qv = _embed_texts(embedder, [query])
        if qv is not None:
            sims = _cosine_sim(qv, chunk_vecs).flatten()
            if sims is not None:
                top_idx = np.argsort(-sims)[: max(10, top_k * 2)]
                for i in top_idx:
                    candidates[int(i)] = max(candidates.get(int(i), 0.0), float(sims[i]))

    # BM25 path
    bm25_ranked = _bm25_rank(query, chunks, top_k=max(10, top_k * 2))
    for idx, score in bm25_ranked:
        scaled = 1.0 - math.exp(-max(score, 0.0) / 10.0)
        candidates[int(idx)] = max(candidates.get(int(idx), 0.0), float(scaled))

    if not candidates:
        return [(i, 0.0) for i in range(min(top_k, len(chunks)))]

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


def _simple_answer(query: str, contexts: List[str]) -> str:
    """
    Lightweight answer generator: extractive + tiny synthesis (no external API).
    """
    joined = "\n\n".join(contexts)
    if re.search(r"\b(summar(y|ise|ize)|overview|abstract|tl;dr)\b", query, re.I):
        sents = re.split(r"(?<=[.!?])\s+", joined)
        return " ".join(sents[:6]).strip()

    q_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    picks = []
    for ctx in contexts:
        sents = re.split(r"(?<=[.!?])\s+", ctx.strip())
        scored = []
        for s in sents:
            score = sum(s.lower().count(t) for t in q_terms) + len(s) * 0.0005
            scored.append((score, s))
        scored.sort(reverse=True, key=lambda x: x[0])
        for _, s in scored[:2]:
            picks.append(s)
    if not picks:
        picks = [contexts[0][:600]]
    seen = set()
    final = []
    for s in picks:
        k = s.strip()
        if k and k not in seen:
            seen.add(k)
            final.append(k)
        if len(" ".join(final)) > 1200:
            break
    return " ".join(final).strip()


def _extract_references(paper_meta: Dict[str, Any], paper_text: str) -> List[Dict[str, Any]]:
    refs = []
    meta_refs = paper_meta.get("references") or paper_meta.get("refs") or []
    if isinstance(meta_refs, list) and meta_refs:
        for r in meta_refs:
            if isinstance(r, dict):
                refs.append({
                    "title": r.get("title") or r.get("name") or "",
                    "authors": r.get("authors") or r.get("author") or "",
                    "year": r.get("year") or r.get("date") or "",
                    "venue": r.get("venue") or r.get("journal") or "",
                    "doi": r.get("doi") or "",
                    "url": r.get("url") or ""
                })
            else:
                refs.append({"title": str(r), "authors": "", "year": "", "venue": "", "doi": "", "url": ""})
    if not refs and paper_text:
        m = re.search(r"\bReferences\b(.*)$", paper_text, re.I | re.S)
        if m:
            tail = m.group(1)
            lines = [l.strip() for l in tail.split("\n") if l.strip()]
            buf = []
            for ln in lines:
                if re.match(r"^\s*\[\d+\]\s+|^\s*\d+\.\s+", ln):
                    if buf:
                        refs.append({"title": " ".join(buf), "authors": "", "year": "", "venue": "", "doi": "", "url": ""})
                        buf = []
                    buf.append(ln)
                else:
                    buf.append(ln)
            if buf:
                refs.append({"title": " ".join(buf), "authors": "", "year": "", "venue": "", "doi": "", "url": ""})
    return refs[:30]


def _extract_datasets(paper_text: str) -> List[str]:
    if not paper_text:
        return []
    hits = re.findall(r"(?:dataset|data set|data\s+set|code|github|zenodo|figshare|kaggle)\S*[:\s]*([^\s,;]+)?", paper_text, re.I)
    extras = re.findall(r"(https?://[^\s)]+)", paper_text)
    cands = set()
    for h in hits:
        if h:
            cands.add(h.strip(".,);]"))
    for u in extras:
        if any(k in u.lower() for k in ["github", "zenodo", "figshare", "kaggle", "/doi/"]):
            cands.add(u.strip(".,);]"))
    return sorted(cands)[:20]


def _extract_pdf_images(pdf_bytes: Optional[bytes], max_images: int = 12) -> List[Dict[str, Any]]:
    if not pdf_bytes or fitz is None:
        return []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out = []
        seen_hash = set()
        for page_index in range(len(doc)):
            page = doc[page_index]
            img_list = page.get_images(full=True)
            for img in img_list:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                h = hashlib.md5(img_bytes).hexdigest()
                if h in seen_hash:
                    continue
                seen_hash.add(h)
                filename = f"paper_img_{page_index}_{xref}_{h[:8]}.png"
                save_dir = st.session_state.get("_smart_chat_imgdir", "smart_chat_images")
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, filename)
                with open(path, "wb") as f:
                    f.write(img_bytes)
                out.append({
                    "path": path,
                    "page": page_index + 1,
                    "size": (pix.width, pix.height)
                })
                if len(out) >= max_images:
                    return out
        return out
    except Exception:
        return []


# ---------- UI helpers ----------
def _pill(text: str):
    st.markdown(
        f"""
        <span style="
            display:inline-block;padding:4px 10px;margin:2px;
            border-radius:999px;
            background:rgba(255,255,255,0.1);
            border:1px solid rgba(255,255,255,0.15);
            font-size:12px;">
            {text}
        </span>
        """,
        unsafe_allow_html=True
    )


def _reference_box(meta: Dict[str, Any]):
    title = meta.get("title") or "Untitled"
    authors = ", ".join(meta.get("authors", [])) if isinstance(meta.get("authors"), list) else (meta.get("authors") or "")
    year = meta.get("year") or meta.get("date") or ""
    venue = meta.get("venue") or meta.get("journal") or meta.get("publication") or ""
    doi = meta.get("doi") or ""
    url = meta.get("url") or meta.get("pdf_url") or meta.get("landing_url") or ""
    st.markdown("""
    <div style="
        margin-top:12px;margin-bottom:8px;
        padding:14px;border-radius:16px;
        border:1px solid rgba(255,255,255,0.15);
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    ">
    """, unsafe_allow_html=True)
    st.markdown(f"**{title}**")
    st.markdown(
        f"""
        <div style="opacity:0.9">
        <div><strong>Authors:</strong> {authors or '—'}</div>
        <div><strong>Year:</strong> {year or '—'} &nbsp;&nbsp; <strong>Venue:</strong> {venue or '—'}</div>
        <div><strong>DOI:</strong> {doi or '—'}</div>
        <div><strong>Link:</strong> {url or '—'}</div>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _rag_citations(chunks_idx: List[int], chunks: List[str]):
    if not chunks_idx:
        return
    with st.expander("RAG references used in the answer"):
        for i in chunks_idx:
            snippet = chunks[i].strip()
            if len(snippet) > 420:
                snippet = snippet[:420] + " ..."
            st.markdown(f"- _…{snippet}_")


# ---------- Intent router ----------
_INTENT_PATTERNS = {
    "summarize": re.compile(r"\b(summar(y|ise|ize)|overview|abstract|tl;dr)\b", re.I),
    "authors": re.compile(r"\b(author|authors|who (wrote|is the author))\b", re.I),
    "year": re.compile(r"\b(what(?:'s| is) the year|publish(ed)? year|year of publication)\b", re.I),
    "doi": re.compile(r"\bdoi\b", re.I),
    "references": re.compile(r"\b(refs?|references|citations?|bibliograph|works? cited)\b", re.I),
    "datasets": re.compile(r"\b(dataset|data set|data\s*set|code|github|zenodo|figshare|kaggle)\b", re.I),
    # Broadened to catch "images of this pdf/paper"
    "figures": re.compile(
        r"\b(fig(?:ure)?s?|images?|charts?|plots?|illustrations?)\b|"
        r"\bimages?\s+of\s+(this\s+)?(pdf|paper)\b",
        re.I
    ),
    "about": re.compile(r"\b(tell me (about|something about) (the )?paper)\b", re.I)
}


def _detect_intent(user_q: str) -> str:
    q = (user_q or "").strip().lower()
    for intent, pat in _INTENT_PATTERNS.items():
        if pat.search(q):
            return intent
    if re.search(r"\b(5|five)\s+references\b", q):
        return "references"
    return "qa"


# ---------- Main public API ----------
def render_smart_chat(
    paper_meta: Dict[str, Any],
    paper_text: str,
    library_records: Optional[List[Dict[str, Any]]] = None,
    embed_model_name: str = "all-MiniLM-L6-v2",
    paper_pdf_bytes: Optional[bytes] = None
):
    """
    Renders a chat box that understands the current paper.
    - paper_meta: dict with title/authors/year/doi/refs/links...
    - paper_text: the full plain text of the paper
    - library_records: reserved for "related work" future features
    - embed_model_name: embedding model (auto-fallbacks used if unavailable)
    - paper_pdf_bytes: optional raw PDF bytes for image extraction
    """
    st.markdown("### Chat with this paper")
    st.caption("Ask anything about this paper. Try: *“Summarize the paper”*, *“Who are the authors?”*, *“Show figures”*, *“Give me 5 references.”*")

    # Session state per paper
    key = _paper_key(paper_meta)
    if key not in st.session_state:
        st.session_state[key] = {"history": []}

    # Precompute chunks + embeddings once per paper
    chunks_key = key + "_chunks"
    vecs_key = key + "_vecs"
    embedder_key = key + "_embedder"

    if chunks_key not in st.session_state:
        st.session_state[chunks_key] = _chunk_text(paper_text or "", max_tokens=900, overlap=140)

    if embedder_key not in st.session_state:
        st.session_state[embedder_key] = _load_embedder(embed_model_name)

    if vecs_key not in st.session_state:
        st.session_state[vecs_key] = _embed_texts(st.session_state[embedder_key], st.session_state[chunks_key])

    # Quick facts pills
    with st.container():
        cols = st.columns(4)
        with cols[0]:
            _pill(f"Year: {paper_meta.get('year') or '—'}")
        with cols[1]:
            _pill(f"DOI: {paper_meta.get('doi') or '—'}")
        with cols[2]:
            a_count = len(paper_meta.get('authors', [])) if isinstance(paper_meta.get('authors'), list) else len([x for x in (paper_meta.get('authors') or '').split(",") if x.strip()])
            _pill(f"Authors: {a_count}")
        with cols[3]:
            has_pdf = "yes" if paper_pdf_bytes else "no"
            _pill(f"PDF: {has_pdf}")

    # Figure/Image gallery (lazy extraction)
    img_key = key + "_images"
    if paper_pdf_bytes and img_key not in st.session_state:
        with st.spinner("Scanning PDF for figures…"):
            st.session_state[img_key] = _extract_pdf_images(paper_pdf_bytes, max_images=12)

    # Display image gallery expander (always clean UI; never prints raw lists)
    if img_key in st.session_state and st.session_state[img_key]:
        with st.expander("📷 Paper figures & images"):
            imgs = st.session_state[img_key]
            grid = st.columns(3)
            for i, img in enumerate(imgs):
                with grid[i % 3]:
                    st.image(img["path"], use_column_width=True, caption=f"Page {img['page']} • {img['size'][0]}×{img['size'][1]}")

    # Render history
    for turn in st.session_state[key]["history"]:
        role = turn["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(turn["content"])
                _reference_box(paper_meta)
                if turn.get("citations"):
                    _rag_citations(turn["citations"], st.session_state[chunks_key])

    user_q = st.chat_input("Type your question…")
    if not user_q:
        return

    # Record user turn
    st.session_state[key]["history"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Route intent
    intent = _detect_intent(user_q)

    # Prepare basic facts from meta
    authors_str = ", ".join(paper_meta.get("authors", [])) if isinstance(paper_meta.get("authors"), list) else (paper_meta.get("authors") or "")
    year = paper_meta.get("year") or paper_meta.get("date") or ""
    doi = paper_meta.get("doi") or ""
    url = paper_meta.get("url") or paper_meta.get("pdf_url") or paper_meta.get("landing_url") or ""
    title = paper_meta.get("title") or "this paper"

    answer = ""
    citations_idx: List[int] = []

    # ----- Intent handlers -----
    if intent in ("summarize", "about"):
        chunks = st.session_state[chunks_key]
        top_idx = list(range(min(6, len(chunks))))
        contexts = [chunks[i] for i in top_idx]
        answer = _simple_answer("summary " + (user_q or ""), contexts)
        citations_idx = top_idx

    elif intent == "authors":
        answer = f"**Authors:** {authors_str or '—'}\n\n**Title:** {title}\n**Year:** {year or '—'}\n**DOI:** {doi or '—'}\n**Link:** {url or '—'}"

    elif intent == "year":
        answer = f"The paper was published in **{year or '—'}**."

    elif intent == "doi":
        answer = f"The DOI is **{doi or '—'}**."

    elif intent == "references":
        refs = _extract_references(paper_meta, paper_text)
        n_req = 5 if re.search(r"\b(5|five)\b", user_q.lower()) else min(10, len(refs))
        refs = refs[:max(5, n_req)]
        if not refs:
            answer = "I couldn’t find structured references in this paper."
        else:
            lines = []
            for i, r in enumerate(refs, 1):
                line = f"**[{i}]** {r.get('title') or 'Untitled'}"
                auth = r.get("authors") or ""
                yr = r.get("year") or ""
                ven = r.get("venue") or ""
                doi_r = r.get("doi") or ""
                url_r = r.get("url") or ""
                meta_line = " — ".join([x for x in [auth, yr, ven] if x])
                if meta_line:
                    line += f" — {meta_line}"
                if doi_r:
                    line += f" — DOI: {doi_r}"
                if url_r:
                    line += f" — {url_r}"
                lines.append(line)
            answer = "\n\n".join(lines)

    elif intent == "datasets":
        ds = _extract_datasets(paper_text)
        if ds:
            answer = "**Dataset/Code links mentioned:**\n\n" + "\n".join(f"- {u}" for u in ds)
        else:
            answer = "I didn’t detect explicit dataset/code links in the text. Try checking the references or the paper’s footnotes."

    elif intent == "figures":
        imgs = st.session_state.get(img_key, [])
        # Build assistant response first
        if imgs:
            answer = f"I found **{len(imgs)}** figure(s). Showing thumbnails below. A full gallery also lives in **📷 Paper figures & images** above."
        else:
            if not paper_pdf_bytes:
                answer = (
                    "I can’t show images because no PDF bytes were provided to the chat. "
                    "Pass `paper_pdf_bytes=...` when calling `render_smart_chat(...)`, "
                    "and I’ll extract any embedded figures."
                )
            else:
                answer = (
                    "I couldn’t extract any embedded images from this PDF. "
                    "Some PDFs flatten figures or use vector-only elements that don’t export as bitmaps."
                )

        with st.chat_message("assistant"):
            st.markdown(answer)
            _reference_box(paper_meta)

            # Inline thumbnails (clean UI; never print raw lists)
            if imgs:
                grid = st.columns(3)
                for i, img in enumerate(imgs):
                    with grid[i % 3]:
                        st.image(
                            img["path"],
                            use_column_width=True,
                            caption=f"Page {img['page']} • {img['size'][0]}×{img['size'][1]}"
                        )

        st.session_state[key]["history"].append({
            "role": "assistant",
            "content": answer,
            "citations": []  # images don’t use text snippets
        })
        return

    else:
        # Generic Q&A via RAG
        chunks = st.session_state[chunks_key]
        embedder = st.session_state[embedder_key]
        vecs = st.session_state[vecs_key]
        selection = _select_context(user_q, chunks, embedder, vecs, top_k=6)
        if selection:
            citations_idx = [i for (i, _) in selection]
            contexts = [chunks[i] for i in citations_idx]
        else:
            contexts = chunks[:6]
            citations_idx = list(range(min(6, len(chunks))))
        answer = _simple_answer(user_q, contexts)

    # ----- Render assistant turn + reference box + rag citations -----
    with st.chat_message("assistant"):
        st.markdown(answer)
        _reference_box(paper_meta)
        if citations_idx:
            _rag_citations(citations_idx, st.session_state[chunks_key])

    st.session_state[key]["history"].append({
        "role": "assistant",
        "content": answer,
        "citations": citations_idx
    })
