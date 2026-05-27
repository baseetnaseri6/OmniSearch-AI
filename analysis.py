import re, os, io, math, difflib, json, time
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer, CrossEncoder = None, None
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None
try:
    import spacy
except Exception:
    spacy = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None
try:
    import networkx as nx
except Exception:
    nx = None
try:
    from pyvis.network import Network
except Exception:
    Network = None
try:
    import requests
except Exception:
    requests = None


# -------------------- Model hub --------------------
class ModelHub:
    """Central place to load embedding & reranker models with graceful fallbacks."""
    _EMBED_IDS = {
        "specter2": "allenai/specter2",
        "bge-m3": "BAAI/bge-m3",
        "miniLM": "all-MiniLM-L6-v2",
    }
    _RERANK_IDS = ["BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L-6-v2"]

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_embedding_model(choice: str = "specter2"):
        if SentenceTransformer is None:
            return None
        tried = []
        ids = []
        if choice in ModelHub._EMBED_IDS:
            ids.append(ModelHub._EMBED_IDS[choice])
        # robust fallbacks
        ids += [ModelHub._EMBED_IDS["miniLM"], ModelHub._EMBED_IDS["bge-m3"], ModelHub._EMBED_IDS["specter2"]]
        seen = set()
        ids = [i for i in ids if not (i in seen or seen.add(i))]
        for mid in ids:
            try:
                return SentenceTransformer(mid)
            except Exception as e:
                tried.append(f"{mid} ({e.__class__.__name__})")
                continue
        st.warning("No embedding model could be loaded. Tried: " + " → ".join(tried))
        return None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_reranker():
        if CrossEncoder is None:
            return None
        for rid in ModelHub._RERANK_IDS:
            try:
                return CrossEncoder(rid)
            except Exception:
                continue
        return None


# -------------------- Embeddings & BM25 --------------------
@st.cache_data(show_spinner=False)
def embed_texts(model, texts: list[str]):
    if model is None or cosine_similarity is None or not texts:
        return None
    try:
        return model.encode(texts, show_progress_bar=False)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def build_bm25_titles(titles_tuple):
    if BM25Okapi is None:
        return None
    tokenized = [re.findall(r"\w+", (t or "").lower()) for t in titles_tuple]
    if not tokenized:
        return None
    return BM25Okapi(tokenized)


# -------------------- Heuristics & analysis helpers --------------------
DATASET_HINT_PATTERNS = [
    r"\bdata\s+available\b", r"\bdataset\b", r"\bdata set\b",
    r"zenodo\.org", r"figshare\.com", r"kaggle\.com", r"archive\.org",
    r"github\.com", r"gitlab\.com", r"bitbucket\.org", r"osf\.io", r"dataverse", r"doi\.org/10\.5281/zenodo",
    r"supplementary\s+(?:material|data)", r"code\s+available", r"data\s+repository", r"released\s+dataset"
]
def find_dataset_hints(text: str, url: str) -> list[str]:
    hits, blob = [], f"{text or ''}\n{url or ''}".lower()
    for pat in DATASET_HINT_PATTERNS:
        if re.search(pat, blob): hits.append(re.sub(r"\\b","",pat))
    return sorted(list(set(hits)))

MONTHS = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
DATE_REGEX = re.compile(rf"(?:\b(?:{MONTHS})\s+\d{{4}}\b)|\b(?:19|20)\d{{2}}\b", re.I)
def detect_dates_in_text(text: str) -> list[str]:
    if not text: return []
    return list(dict.fromkeys(m.group(0) for m in re.finditer(DATE_REGEX, text)))

STOPWORDS = set(("the of and to in a for on with as by we from this that an be is are were was at it its their our your his her into about using use via within among between can may might could should would have has had not no yes over under").split())

def _extract_keywords_simple(text, top_k=6):
    words = re.findall(r'\w{4,}', (text or "").lower())
    freq = {}
    for w in words: freq[w] = freq.get(w, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

@st.cache_resource(show_spinner=False)
def _get_spacy():
    if not spacy: return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

def _canon_title(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 :._-]", "", s)
    return s.strip()

def _norm_doi(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x)
    x = re.sub(r"^doi:\s*", "", x)
    x = re.sub(r"[>\.\s,;]+$", "", x)
    return x

def get_references_column(df: pd.DataFrame) -> str:
    candidates = ["References", "Reference", "refs", "Refs", "Bibliography", "bibliography"]
    for c in candidates:
        if c in df.columns:
            return c
    return "References"


# -------------------- Analyzer (search + analysis + visuals) --------------------
class Analyzer:
    """Holds corpus, embeddings, and search/analysis methods."""
    def __init__(self, df: pd.DataFrame, emb, model, bm25=None):
        self.df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        self.emb = emb
        self.model = model
        self.bm25 = bm25 if bm25 is not None else build_bm25_titles(tuple(self.df.get("Title", pd.Series()).astype(str).tolist()))
        self._nlp = _get_spacy()

    # ------- Retrieval / suggestions
    def suggest_titles(self, q: str, limit: int = 10):
        if not q: return []
        if self.bm25 is not None and not self.df.empty:
            toks = re.findall(r"\w+", q.lower().strip())
            scores = self.bm25.get_scores(toks)
            order = np.argsort(scores)[::-1]
            out = []
            for i in order[:limit*2]:
                title = self.df.iloc[i]["Title"]
                if q.lower() in str(title).lower() or scores[i] > 0:
                    out.append((int(i), title, float(scores[i])))
                if len(out) >= limit: break
            return out
        # naive fallback
        titles = self.df.get("Title", pd.Series()).astype(str).tolist()
        scored = []
        ql = q.lower().strip()
        for i, t in enumerate(titles):
            tl = t.lower(); score = 0.0
            if ql in tl: score += 2.0
            overlap = len(set(ql.split()) & set(tl.split()))
            score += 0.2 * overlap
            if score > 0: scored.append((i, t, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:limit]

    @st.cache_data(show_spinner=False)
    def _encode_query(_self, text: str):
        if not _self.model: return None
        try: return _self.model.encode([text], show_progress_bar=False)
        except Exception: return None

    def suggest_titles_semantic(self, q: str, limit: int = 20):
        if not q: return []
        if self.emb is None or self.model is None or self.df.empty or cosine_similarity is None:
            return self.suggest_titles(q, limit=min(10, limit))
        q_vec = self._encode_query(q)
        if q_vec is None: return self.suggest_titles(q, limit=min(10, limit))
        try:
            sims = cosine_similarity(q_vec, self.emb)[0]
            order = np.argsort(sims)[::-1]
            out = []
            for i in order[:limit * 2]:
                title = self.df.iloc[int(i)]["Title"]
                out.append((int(i), title, float(sims[int(i)])))
                if len(out) >= limit:
                    break
            return out
        except Exception:
            return self.suggest_titles(q, limit=min(10, limit))

    def suggest_titles_hybrid(self, q: str, limit: int = 20, alpha: float = 0.65):
        kw = self.suggest_titles(q, limit=limit*4)
        kw_scores = {i: s for (i, _, s) in kw}
        sem = self.suggest_titles_semantic(q, limit=limit*4)
        sem_scores = {i: s for (i, _, s) in sem}
        all_idx = set(kw_scores) | set(sem_scores)
        if not all_idx: return []
        def _norm(d):
            if not d: return {}
            vals = list(d.values()); mn, mx = min(vals), max(vals)
            denom = (mx - mn) if (mx - mn) != 0 else 1.0
            return {k: (v - mn)/denom for k, v in d.items()}
        kw_norm = _norm(kw_scores); sem_norm = _norm(sem_scores)
        combined = []
        for i in all_idx:
            score = alpha * kw_norm.get(i, 0.0) + (1.0 - alpha) * sem_norm.get(i, 0.0)
            combined.append((int(i), self.df.iloc[int(i)]["Title"], float(score)))
        combined.sort(key=lambda x: x[2], reverse=True)
        return combined[:limit]

    # ------- Reranking
    def rerank_results(self, query: str, results: list[dict], text_keys=("Title","Abstract")) -> list[dict]:
        ce = ModelHub.get_reranker()
        if ce is None or not results:
            return results
        pairs = []
        for r in results:
            blob = " ".join([str(r.get(k, "")) for k in text_keys])[:4096]
            pairs.append((query, blob))
        try:
            scores = ce.predict(pairs)
            order = np.argsort(scores)[::-1]
            ranked = []
            for i in order:
                rr = results[int(i)].copy()
                rr["rerank_score"] = float(scores[int(i)])
                ranked.append(rr)
            return ranked
        except Exception:
            return results

    # ------- Corpus similarity
    def related_count_by_abstract(self, abstract: str, threshold: float = 0.35, active_emb=None) -> int:
        emb = active_emb if active_emb is not None else self.emb
        if emb is None or not abstract or self.model is None or cosine_similarity is None:
            return 0
        try:
            vec = self.model.encode([abstract])
            sims = cosine_similarity(vec, emb)[0]
            return int((sims > threshold).sum())
        except Exception:
            return 0

    def similarity_stats_vs_corpus(self, abstract: str | None, idx: int | None, active_emb=None):
        emb = active_emb if active_emb is not None else self.emb
        if emb is None or self.model is None or cosine_similarity is None:
            return (0.0, 0.0)
        sims = None
        try:
            if abstract:
                vec = self.model.encode([abstract])
                sims = cosine_similarity(vec, emb)[0]
            elif idx is not None and 0 <= int(idx) < len(emb):
                vec = emb[int(idx):int(idx)+1]
                sims = cosine_similarity(vec, emb)[0]
        except Exception:
            pass
        if sims is None or len(sims)==0: return (0.0, 0.0)
        max_sim = float(np.max(sims))
        top5 = np.sort(sims)[-5:]
        avg_top5 = float(np.mean(top5)) if len(top5) else 0.0
        return (max_sim, avg_top5)

    # ------- Paper analysis
    def analyze_paper_dict(self, paper: dict, idx: int | None, active_df=None, active_emb=None):
        title = paper.get("Title",""); abstract = paper.get("Abstract","") or ""
        text = f"{title}. {abstract}".strip()
        toks = re.findall(r"\w+", text.lower()); length = len(toks)
        uniq = len(set([t for t in toks if t not in STOPWORDS])); richness = (uniq/length) if length else 0.0

        keywords, kw_freq = [], {}
        if TfidfVectorizer is not None and text:
            try:
                vec = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words="english")
                X = vec.fit_transform([text]); feats = vec.get_feature_names_out(); weights = X.toarray()[0]
                top_idx = np.argsort(weights)[-8:][::-1]
                keywords = [feats[i] for i in top_idx if feats[i] not in STOPWORDS][:8]
                for k in keywords:
                    kw_freq[k] = max(1, int(round(100*weights[list(feats).tolist().index(k)])))
            except Exception:
                pass
        if not keywords:
            words = re.findall(r'\w{4,}', (abstract or title).lower())
            for w in words: kw_freq[w] = kw_freq.get(w,0)+1
            keywords = [w for w,_ in sorted(kw_freq.items(), key=lambda x:x[1], reverse=True)[:8]]
            kw_freq = {k: (abstract.lower().count(k.lower()) + title.lower().count(k.lower())) for k in keywords}

        ents = {}
        try:
            nlp = self._nlp
            if nlp:
                doc = nlp(text)
                for e in doc.ents:
                    if e.label_ in ("ORG","GPE","NORP","PERSON","LOC","DATE","TIME","CARDINAL","QUANTITY","EVENT"):
                        ents[e.label_] = ents.get(e.label_, 0) + 1
        except Exception:
            pass

        dates = detect_dates_in_text(abstract)
        y = paper.get("Year")
        try: has_year = (not pd.isna(y))
        except Exception: has_year = bool(str(y).strip())
        hints = find_dataset_hints(abstract + "\n" + title, paper.get("URL",""))
        signals = {"has_year": bool(has_year), "has_dates": bool(dates), "has_datasets": bool(hints)}

        df_c = active_df if isinstance(active_df, pd.DataFrame) else self.df
        emb_c = active_emb if active_emb is not None else self.emb
        max_sim, avg_top5 = self.similarity_stats_vs_corpus(abstract if idx is None else None, idx, emb_c)
        novelty = max(0.0, 1.0 - max_sim)

        return {
            "length": length, "uniq": uniq, "richness": richness,
            "keywords": keywords, "kw_freq": kw_freq, "entities": ents,
            "dates_list": dates, "signals": signals, "dataset_hints": hints,
            "max_sim": max_sim, "avg_top5": avg_top5, "novelty": novelty,
        }

    # ------- Mind map (Plotly fallback, PyVis if available)
    def show_mind_map_generic(self, title: str, abstract: str, authors_str: str, active_df=None, active_emb=None, colors=None):
        if go is None:
            st.info("Plotly unavailable."); return

        df_c = active_df if isinstance(active_df, pd.DataFrame) else self.df
        emb_c = active_emb if active_emb is not None else self.emb

        # keywords
        kws = []
        try:
            nlp = self._nlp
            if nlp:
                doc = nlp(abstract or title)
                kws = [c.text for c in doc.noun_chunks if len(c.text) > 3][:8]
            else:
                kws = _extract_keywords_simple(abstract or title, top_k=8)
        except Exception:
            kws = _extract_keywords_simple(abstract or title, top_k=8)

        # related
        related = []
        if emb_c is not None and abstract and self.model is not None and cosine_similarity is not None:
            try:
                vec = self.model.encode([abstract])
                sims = cosine_similarity(vec, emb_c)[0]
                order = sims.argsort()[::-1]
                for j in order[:8]:
                    if 0 <= int(j) < len(df_c):
                        t = df_c.iloc[int(j)].get("Title",""); u = df_c.iloc[int(j)].get("URL","#")
                        if t: related.append((t,u))
            except Exception:
                pass

        # PyVis (interactive) if available
        if nx is not None and Network is not None:
            try:
                bg = (colors or {}).get('bg_color', '#000019')
                fg = (colors or {}).get('text_color', '#f2f2f7')
                accent = (colors or {}).get('link_color', '#58D8FA')
                net = Network(height="540px", width="100%", bgcolor=bg, font_color=fg)
                net.toggle_physics(True)
                center = title[:80]+"…" if len(title)>80 else title
                net.add_node("center", label=center, shape="ellipse", color=accent)
                for a in [a.strip() for a in (authors_str or "").split(",") if a.strip()][:6]:
                    node = f"a_{a}"; net.add_node(node, label=a, shape="dot"); net.add_edge("center", node)
                for k in kws:
                    node = f"k_{k}"; net.add_node(node, label=k, shape="dot"); net.add_edge("center", node)
                for (t,u) in related[:6]:
                    node = f"r_{hash(t)%99999}"
                    label = t[:48]+"…" if len(t)>48 else t
                    net.add_node(node, label=label, shape="box", href=(u or "#"), target="_blank")
                    net.add_edge("center", node)
                tmp = "mindmap_auto.html"; net.show(tmp)
                with open(tmp,"r",encoding="utf-8") as f: html = f.read()
                st.components.v1.html(html, height=560, scrolling=True); return
            except Exception:
                pass

        # Plotly fallback (static)
        def ring_coords(n, r, phase=0.0):
            if n <= 0: return []
            return [(r*np.cos(2*np.pi*i/float(n)+phase), r*np.sin(2*np.pi*i/float(n)+phase)) for i in range(n)]

        nodes = [{"id": "center", "label": title, "x": 0.0, "y": 0.0, "group": "center"}]
        for lab, r, phase, group, seq in [
            ("authors", 1.0, 0.0, "author", [a.strip() for a in (authors_str or "").split(",") if a.strip()][:8]),
            ("keywords", 1.4, 0.5, "keyword", kws[:8]),
            ("related", 1.8, 0.2, "related", [t for t in related[:8]]),
        ]:
            for (item, (x,y)) in zip(seq, ring_coords(len(seq), r, phase)):
                label = item if isinstance(item, str) else (item[0] if isinstance(item, (list, tuple)) else str(item))
                nodes.append({"id": f"{group}_{hash(label)%999999}", "label": label, "x": x, "y": y, "group": group})

        edges = [("center", n["id"]) for n in nodes[1:]]

        edge_x, edge_y = [], []
        id2pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
        for s, t in edges:
            if s in id2pos and t in id2pos:
                x0, y0 = id2pos[s]; x1, y1 = id2pos[t]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        fig = go.Figure()
        if edge_x: fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', hoverinfo='none'))
        for grp, size in [("center", 20), ("author", 10), ("keyword", 10), ("related", 12)]:
            xs = [n["x"] for n in nodes if n["group"] == grp]
            ys = [n["y"] for n in nodes if n["group"] == grp]
            labels = [n["label"] for n in nodes if n["group"] == grp]
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text', text=labels, textposition='top center',
                                         marker=dict(size=size), name=grp.capitalize()))
        fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                          margin=dict(l=100, r=20, t=60, b=40), height=520)
        st.plotly_chart(fig, use_container_width=True)


# -------------------- Utilities you already call in main --------------------
def summarize_text(text: str, max_sentences: int = 3) -> str:
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    sents = [s for s in sents if s]
    return " ".join(sents[:max_sentences]) if sents else ""

def build_analyzer(df: pd.DataFrame, emb, model):
    bm25 = build_bm25_titles(tuple(df.get("Title", pd.Series()).astype(str).tolist()))
    return Analyzer(df, emb, model, bm25=bm25)

def load_reranker():
    return ModelHub.get_reranker()

def rerank_results(query: str, results: list[dict], text_keys=("Title","Abstract")) -> list[dict]:
    ce = load_reranker()
    if ce is None or not results:
        return results
    pairs = []
    for r in results:
        blob = " ".join([str(r.get(k, "")) for k in text_keys])[:4096]
        pairs.append((query, blob))
    try:
        scores = ce.predict(pairs)
        order = np.argsort(scores)[::-1]
        ranked = []
        for i in order:
            rr = results[int(i)].copy()
            rr["rerank_score"] = float(scores[int(i)])
            ranked.append(rr)
        return ranked
    except Exception:
        return results


# ==================== Debug + Model Evaluation Toolkit ====================
def debug_reference_coverage(df: pd.DataFrame):
    """Quick diagnostics for References/DOI coverage in the library CSV."""
    refs_col_name = get_references_column(df)
    # ✅ use df.get(col, fallback) — avoid truthiness on Series
    refs_col = df.get(refs_col_name, pd.Series(dtype=str)).astype(str)

    nonempty = refs_col[
        ~refs_col.isna()
        & (refs_col.str.strip() != "")
        & (refs_col.str.lower() != "nan")
    ]
    doi_col = df.get("DOI")
    has_any_doi = bool(doi_col is not None and (doi_col.astype(str).str.strip() != "").any())

    st.caption(f"Rows with non-empty '{refs_col_name}': **{len(nonempty)} / {len(df)}**")
    st.caption(f"Any DOI populated in dataset: **{has_any_doi}**")

    sample = nonempty.head(3).tolist()
    if sample:
        st.markdown("**Sample reference cells (first 3):**")
        for s in sample:
            st.code(s[:600] + ("..." if len(s) > 600 else ""))


# ------- External fetchers for enrichment (Crossref / OpenAlex)
def fetch_crossref_refs_for_doi(doi: str, max_refs: int = 80) -> list[str]:
    if requests is None or not doi:
        return []
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=15)
        if not r.ok:
            return []
        msg = r.json().get("message", {})
        refs = []
        for ref in msg.get("reference", [])[:max_refs]:
            if isinstance(ref, dict):
                if ref.get("unstructured"):
                    refs.append(ref["unstructured"])
                elif ref.get("DOI"):
                    refs.append(f"DOI: {ref['DOI']}")
        return refs
    except Exception:
        return []

def fetch_openalex_refs_by_title(title: str, max_refs: int = 60) -> list[str]:
    if requests is None or not title:
        return []
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": title, "per_page": 1},
            timeout=15
        )
        if not r.ok:
            return []
        items = r.json().get("results", [])
        if not items:
            return []
        ids = items[0].get("referenced_works", [])[:max_refs]
        if not ids:
            return []
        r2 = requests.get(
            "https://api.openalex.org/works",
            params={"filter": "ids.openalex:" + ("|".join(ids)), "per_page": len(ids)},
            timeout=15
        )
        refs = []
        if r2.ok:
            for it in r2.json().get("results", []):
                t = it.get("display_name", "") or ""
                y = it.get("publication_year", "") or ""
                doi = (it.get("doi", "") or "")
                refs.append(f"{t} ({y}) {doi}".strip())
        return refs
    except Exception:
        return []


@st.cache_data(show_spinner=True)
def batch_enrich_references(df: pd.DataFrame, strategy: str = "auto", max_rows: int = 200):
    """
    Fill empty References using Crossref (by DOI) and OpenAlex (by Title).
    strategy: 'auto' (DOI first then Title), 'doi_first', or 'title_first'
    Returns: (new_df, updated_count)
    """
    refs_col = get_references_column(df)
    work = df.copy()
    updated = 0

    # ensure References column exists
    if refs_col not in work.columns:
        work[refs_col] = ""

    mask_empty = work.get(refs_col, pd.Series(dtype=str)).astype(str).str.strip().isin(
        ["", "nan", "none", "NaN", "None"]
    )
    idxs = work[mask_empty].index.tolist()[:max_rows]

    for i in idxs:
        doi = _norm_doi(str(work.at[i, "DOI"]) if "DOI" in work.columns else "")
        title = str(work.at[i, "Title"]) if "Title" in work.columns else ""
        refs = []

        if strategy in ("auto", "doi_first"):
            if doi:
                refs = fetch_crossref_refs_for_doi(doi)
            if not refs and title:
                refs = fetch_openalex_refs_by_title(title)
        else:  # title_first
            if title:
                refs = fetch_openalex_refs_by_title(title)
            if not refs and doi:
                refs = fetch_crossref_refs_for_doi(doi)

        if refs:
            work.at[i, refs_col] = "\n".join(refs)
            updated += 1
            time.sleep(0.2)  # be gentle to APIs

    return work, updated


@st.cache_data(show_spinner=False)
def build_qrels_from_references(df: pd.DataFrame, title_col="Title", refs_col=None,
                                max_refs_per_query: int = 60, cutoff: float = 0.66):
    """
    Build pseudo ground-truth by mapping each paper's references to indices in df.
    Uses DOI exact matches first, then fuzzy title matches.
    """
    if refs_col is None:
        refs_col = get_references_column(df)
    if title_col not in df.columns:
        return {}, {"queries": 0, "matches": 0}

    # title + DOI maps
    title_map = {_canon_title(t): i for i, t in enumerate(df[title_col].astype(str).tolist())}
    doi_map = {}
    if "DOI" in df.columns:
        for i, d in enumerate(df["DOI"].astype(str).tolist()):
            nd = _norm_doi(d)
            if nd:
                doi_map[nd] = i

    QUOTE_PATTS = [r'“([^”]+)”', r'"([^"]+)"', r'‘([^’]+)’', r"'([^']+)'"]
    YEAR = r"(?:19|20)\d{2}"

    def _extract_title_from_ref_line(line: str) -> str:
        if not line:
            return ""
        x = re.sub(r"https?://\S+", " ", line)
        x = re.sub(r"\bdoi:\s*\S+", " ", x, flags=re.I)
        x = re.sub(r"\s+", " ", x).strip()
        # Try quoted title first
        for qp in QUOTE_PATTS:
            m = re.search(qp, x)
            if m and len(m.group(1)) >= 6:
                return re.sub(r"\s+", " ", m.group(1)).strip()
        # Try longest alpha-ish chunk after year
        m2 = re.search(rf"{YEAR}[^A-Za-z]*([A-Za-z0-9 :._-]{{10,}})", x)
        if m2:
            cand = m2.group(1)
        else:
            parts = [p.strip(" .,:;-—") for p in re.split(r"\.\s|;\s|\s-\s|—", x) if p.strip()]
            if not parts:
                parts = [x]
            parts.sort(key=lambda t: sum(c.isalpha() for c in t), reverse=True)
            cand = parts[0]
        return re.sub(r"\s+", " ", cand).strip()

    def _best_match_index(title: str) -> int | None:
        if not title:
            return None
        ct = _canon_title(title)
        if not ct:
            return None
        if ct in title_map:
            return title_map[ct]
        keys = list(title_map.keys())
        match = difflib.get_close_matches(ct, keys, n=1, cutoff=cutoff)
        if match:
            return title_map[match[0]]
        return None

    qrels = {}
    total_matches = 0

    # ✅ use df.get(col, fallback) — avoid truthiness on Series
    series = df.get(refs_col, pd.Series(dtype=str)).astype(str)
    for i, cell in enumerate(series.tolist()):
        cell_clean = (cell or "").strip().lower()
        if not cell_clean or cell_clean in ("", "nan", "none"):
            continue

        # allow JSON arrays like '["ref1","ref2",...]'
        lines = []
        if cell.strip().startswith("[") and cell.strip().endswith("]"):
            try:
                arr = json.loads(cell)
                lines = [str(x) for x in arr if str(x).strip()]
            except Exception:
                pass
        if not lines:
            lines = [p.strip(" ;") for p in re.split(r"[\n;]+", cell) if p.strip(" ;")]

        found = set()
        for ln in lines[:max_refs_per_query]:
            # DOI exact
            doi_match = re.search(r"\b10\.\d{4,9}/\S+\b", ln, flags=re.I)
            if doi_match:
                nd = _norm_doi(doi_match.group(0))
                if nd in doi_map and doi_map[nd] != i:
                    found.add(int(doi_map[nd]))
                    continue
            # Title fuzzy
            tt = _extract_title_from_ref_line(ln)
            j = _best_match_index(tt)
            if j is not None and j != i:
                found.add(int(j))

        if found:
            qrels[i] = found
            total_matches += len(found)

    return qrels, {"queries": len(qrels), "matches": total_matches}


def _dcg_at_k(ranked_hits: list[int], relevant_set: set[int], k: int) -> float:
    dcg = 0.0
    for rank, idx in enumerate(ranked_hits[:k], start=1):
        if idx in relevant_set:
            dcg += 1.0 / math.log2(rank + 1.0)
    return dcg

def _ndcg_at_k(ranked_hits: list[int], relevant_set: set[int], k: int) -> float:
    dcg = _dcg_at_k(ranked_hits, relevant_set, k)
    ideal_rel = min(len(relevant_set), k)
    idcg = sum([1.0 / math.log2(r + 1.0) for r in range(1, ideal_rel + 1)])
    return (dcg / idcg) if idcg > 0 else 0.0

@st.cache_data(show_spinner=False)
def _embed_corpus_for_model(model_id: str, texts: list[str]):
    mdl = ModelHub.get_embedding_model(model_id)
    emb = embed_texts(mdl, texts)
    return emb

@st.cache_data(show_spinner=False)
def evaluate_model_on_qrels(df: pd.DataFrame, model_id: str, qrels: dict[int,set[int]],
                            text_col="Abstract", k_primary: int = 5, k_secondary: int = 10,
                            limit_docs: int | None = 2500):
    """
    Evaluate retrieval: for each query index i with qrels[i] != empty,
    rank the whole corpus by cosine over embeddings(text_col), compute P@k, R@k, MRR@10, nDCG@10.
    """
    if cosine_similarity is None:
        return {"error": "cosine_similarity unavailable"}, None

    # optional subset to keep fast
    df_eval = df.copy()
    if isinstance(limit_docs, int) and limit_docs > 0 and len(df_eval) > limit_docs:
        df_eval = df_eval.iloc[:limit_docs].reset_index(drop=True)
        # reindex qrels to this slice
        keep = set(range(len(df_eval)))
        qrels = {i: {j for j in js if j in keep} for i, js in qrels.items() if i in keep}
        qrels = {i: s for i, s in qrels.items() if s}

    texts = df_eval.get(text_col, pd.Series()).astype(str).fillna("").tolist()
    emb = _embed_corpus_for_model(model_id, texts)
    if emb is None:
        return {"error": f"failed to embed with {model_id}"}, None

    p5_list, r5_list, mrr10_list, ndcg10_list = [], [], [], []
    first_hit_ranks = []

    n = len(df_eval)
    for i, relset in qrels.items():
        if i >= n:
            continue
        qv = emb[i:i+1]
        sims = cosine_similarity(qv, emb)[0]
        sims[i] = -1e9  # exclude self
        order = np.argsort(sims)[::-1]
        ranked = order.tolist()

        # metrics
        topk = ranked[:k_primary]
        hits_k = [r for r in topk if r in relset]
        p5 = len(hits_k) / float(k_primary)
        r5 = len(hits_k) / float(len(relset)) if relset else 0.0

        # MRR@K
        topk2 = ranked[:k_secondary]
        rr = 0.0
        for rank, idx in enumerate(topk2, start=1):
            if idx in relset:
                rr = 1.0 / float(rank); break

        # nDCG@K
        nd = _ndcg_at_k(ranked, relset, k_secondary)

        p5_list.append(p5); r5_list.append(r5); mrr10_list.append(rr); ndcg10_list.append(nd)

        # first hit rank (for histogram)
        fh = None
        for rank, idx in enumerate(ranked, start=1):
            if idx in relset: fh = rank; break
        first_hit_ranks.append(fh if fh is not None else 0)

    agg = {
        "model": model_id,
        "queries": len(p5_list),
        "P@5": float(np.mean(p5_list)) if p5_list else 0.0,
        "R@5": float(np.mean(r5_list)) if r5_list else 0.0,
        "MRR@10": float(np.mean(mrr10_list)) if mrr10_list else 0.0,
        "nDCG@10": float(np.mean(ndcg10_list)) if ndcg10_list else 0.0,
        "first_hit_ranks": first_hit_ranks,
    }
    details = {
        "p5_list": p5_list, "r5_list": r5_list, "mrr10_list": mrr10_list, "ndcg10_list": ndcg10_list
    }
    return agg, details


def _heatmap_metrics(models_summary: list[dict]):
    if go is None: return
    rows = [m["model"] for m in models_summary]
    cols = ["P@5", "R@5", "MRR@10", "nDCG@10"]
    z = [[m.get(c, 0.0) for c in cols] for m in models_summary]
    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=rows, coloraxis="coloraxis"))
    fig.update_layout(
        title="Model comparison (higher is better)",
        coloraxis=dict(colorscale="Blues"),
        xaxis=dict(side="top")
    )
    return fig

def _bars_metrics(models_summary: list[dict]):
    if go is None: return
    cols = ["P@5", "R@5", "MRR@10", "nDCG@10"]
    x = [m["model"] for m in models_summary]
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Bar(name=c, x=x, y=[m.get(c, 0.0) for m in models_summary]))
    fig.update_layout(barmode='group', title="Metrics by model")
    return fig

def _hist_first_hit(models_summary: list[dict]):
    if go is None: return
    fig = go.Figure()
    for m in models_summary:
        ranks = [r for r in (m.get("first_hit_ranks") or []) if r and r > 0 and r < 200]
        if not ranks: continue
        fig.add_trace(go.Histogram(x=ranks, name=m["model"], opacity=0.65))
    fig.update_layout(barmode="overlay", title="First relevant hit rank (lower is better)",
                      xaxis_title="Rank", yaxis_title="Count")
    return fig


def render_model_eval_dashboard(df: pd.DataFrame):
    """
    Streamlit UI: build qrels from References (DOI + fuzzy), optional enrichment, compare models, visualize.
    """
    st.markdown("### 🔬 Model Evaluation (beta)")

    if df is None or df.empty or "Title" not in df.columns:
        st.info("Load your library (CSV) with Title/Abstract/References to evaluate.")
        return

    # Quick coverage diagnostics
    debug_reference_coverage(df)

    # Optional: try to fill empty References via APIs
    with st.expander("🛠️ Fix missing references (Crossref/OpenAlex)"):
        st.write("Populate empty reference lists using DOI (Crossref) and Title (OpenAlex).")
        strat = st.selectbox("Strategy", ["auto", "doi_first", "title_first"], index=0)
        max_rows = st.slider("Rows to try", 50, min(2000, len(df)), min(200, len(df)), 50)
        if requests is None:
            st.warning("Python package 'requests' not available — install it to use enrichment.")
            st.session_state.pop("__enriched_df__", None)
        else:
            if st.button("Enrich missing references now", key="btn_enrich_refs"):
                with st.spinner("Fetching references from APIs…"):
                    enriched_df, updated = batch_enrich_references(df, strategy=strat, max_rows=max_rows)
                st.success(f"Filled references for {updated} rows.")
                st.session_state["__enriched_df__"] = enriched_df

    # choose dataset for evaluation (original vs enriched)
    df_eval = st.session_state.get("__enriched_df__", df)

    with st.expander("Ground-truth setup"):
        st.write("We map each paper’s **References** to other rows in your library (prefer DOI, then fuzzy title).")
        cutoff = st.slider("Title match cutoff (fuzzy)", 0.5, 0.95, 0.66, 0.01, key="eval_cutoff")
        max_refs = st.slider("Max refs per query", 5, 100, 60, 5, key="eval_max_refs")

    with st.expander("Evaluation controls", expanded=True):
        candidate_models = st.multiselect(
            "Models to compare", ["specter2","bge-m3","miniLM"], default=["specter2","bge-m3","miniLM"]
        )
        limit_docs = st.slider("Max corpus size (to keep it fast)", 500, 8000, 2500, 500)
        k_primary = st.slider("K for Precision/Recall@K", 5, 20, 5, 1)
        k_secondary = st.slider("K for MRR/nDCG@K", 5, 30, 10, 1)
        text_col = st.selectbox("Text column to embed", ["Abstract","Title","Title+Abstract"], index=0)

    # Compose Title+Abstract if selected
    if text_col == "Title+Abstract":
        work = df_eval.copy()
        work["__TPLUS__"] = (work.get("Title","").astype(str) + ". " + work.get("Abstract","").astype(str)).str.strip()
        text_col_use = "__TPLUS__"
        df_for_qrels = work
    else:
        text_col_use = text_col
        df_for_qrels = df_eval

    with st.spinner("Building pseudo ground-truth from references…"):
        qrels, stats = build_qrels_from_references(df_for_qrels, cutoff=cutoff, max_refs_per_query=max_refs)

    if stats.get("queries", 0) < 8:
        st.warning(f"Few usable queries from references ({stats.get('queries',0)}). "
                   f"Add more rows with populated References/DOIs, run enrichment above, or lower the cutoff.")
    st.caption(f"Query papers with mapped references: **{stats.get('queries',0)}** | Matched links: **{stats.get('matches',0)}**")

    if not qrels:
        return

    # Evaluate models
    summaries = []
    progress = st.progress(0.0)
    for i, mid in enumerate(candidate_models):
        progress.progress((i)/max(1,len(candidate_models)))
        with st.spinner(f"Evaluating {mid}…"):
            agg, details = evaluate_model_on_qrels(
                df_for_qrels, mid, qrels, text_col=text_col_use,
                k_primary=k_primary, k_secondary=k_secondary, limit_docs=limit_docs
            )
            if "error" in agg:
                st.error(f"{mid}: {agg['error']}")
            else:
                summaries.append(agg)
    progress.progress(1.0)

    if not summaries:
        st.info("No results to show.")
        return

    # Visuals
    c1, c2 = st.columns(2)
    with c1:
        fig_hm = _heatmap_metrics(summaries)
        if fig_hm: st.plotly_chart(fig_hm, use_container_width=True)
    with c2:
        fig_bars = _bars_metrics(summaries)
        if fig_bars: st.plotly_chart(fig_bars, use_container_width=True)

    fig_hist = _hist_first_hit(summaries)
    if fig_hist: st.plotly_chart(fig_hist, use_container_width=True)

    # Table summary
    st.markdown("#### Summary table")
    st.dataframe(pd.DataFrame([{
        "Model": s["model"],
        "Queries": s["queries"],
        f"P@{k_primary}": round(s["P@5"],4) if k_primary==5 else round(s["P@5"],4),
        f"R@{k_primary}": round(s["R@5"],4) if k_primary==5 else round(s["R@5"],4),
        f"MRR@{k_secondary}": round(s["MRR@10"],4) if k_secondary==10 else round(s["MRR@10"],4),
        f"nDCG@{k_secondary}": round(s["nDCG@10"],4) if k_secondary==10 else round(s["nDCG@10"],4),
    } for s in summaries]).set_index("Model"))

    # Winner
    best = max(summaries, key=lambda s: (s["nDCG@10"], s["MRR@10"], s["P@5"]))
    st.success(
        f"🏆 Best by nDCG@{k_secondary}: **{best['model']}** "
        f"(P@{k_primary}={best['P@5']:.3f}, R@{k_primary}={best['R@5']:.3f}, "
        f"MRR@{k_secondary}={best['MRR@10']:.3f}, nDCG@{k_secondary}={best['nDCG@10']:.3f})"
    )
