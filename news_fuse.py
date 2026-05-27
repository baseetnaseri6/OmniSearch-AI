from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from html import unescape
import re
import json
from io import BytesIO
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as st_html  # HTML renderer (isolated)

# Optional deps
try:
    import requests
except Exception:
    requests = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
except Exception:
    SentenceTransformer = None  # type: ignore

# Optional TF-IDF fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cos
except Exception:
    TfidfVectorizer = None  # type: ignore
    _sk_cos = None  # type: ignore

# (LVK/skymap libs only needed if you reintroduce a monitor in the future)
try:
    from ligo.gracedb.rest import GraceDb  # pip install ligo-gracedb
except Exception:
    GraceDb = None  # type: ignore

try:
    from ligo.skymap.io import read_sky_map  # pip install ligo-skymap
    import healpy as hp  # pip install healpy
except Exception:
    read_sky_map = None  # type: ignore
    hp = None  # type: ignore


@dataclass
class Event:
    id: str
    when: datetime
    ra: float | None
    dec: float | None
    instruments: list[str]
    topics: list[str]
    summary: str
    src: str  # "GCN", "TNS", "LVK", "ATel"


@dataclass
class Paper:
    id: str
    when: datetime
    title: str
    abstract: str
    url: str
    kws: list[str]


# ===== UTILS =====
def _parse_iso_utc(ts: Optional[str]) -> Optional[datetime]:
    """Accepts ISO8601 and 'YYYY-MM-DD HH:MM:SS UTC' strings."""
    if not ts:
        return None
    ts = ts.strip()
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts.replace("UTC", "").strip(), fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _format_far_human(far_hz: Optional[float]) -> str:
    """Return a friendly FAR string like '≈ 1 per 46.0 days (2.50e-07 Hz)'."""
    if not far_hz or far_hz <= 0:
        return "—"
    seconds = 1.0 / far_hz
    if seconds < 120:
        val, unit = seconds, "s"
    elif seconds < 7200:
        val, unit = seconds / 60.0, "min"
    elif seconds < 172800:
        val, unit = seconds / 3600.0, "h"
    elif seconds < 31557600:
        val, unit = seconds / 86400.0, "days"
    else:
        val, unit = seconds / 31557600.0, "years"
    return f"≈ 1 per {val:.1f} {unit} ({far_hz:.2e} Hz)"


def _reason_msg(code: Optional[str]) -> str:
    """Map internal reason codes to friendly messages."""
    mapping = {
        None: "",
        "no_preferred_event": "No preferred event yet",
        "detail_failed": "Couldn't fetch event details",
        "missing_field": "Not provided by alert",
        "not_computed_yet": "Not yet published",
        "no_public_skymap": "No public sky map",
        "skymap_libs_missing": "Install ligo-skymap + healpy",
        "skymap_read_error": "Sky map couldn't be read",
        "http_cannot_fetch_skymap": "HTTP fallback can't read sky map",
        "files_list_failed": "Couldn't list files",
    }
    return mapping.get(code, str(code))


# ===== (REMOVED) Real-time Black Hole Monitor =====
# fetch_blackhole_data() and render_blackhole_monitor() were removed by request.


# ------------------ FETCHERS (papers/events for matching) ------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_events(enabled_sources: Optional[Dict[str, bool]] = None) -> List[Event]:
    """
    Stub for now (keeps an LVK-looking event so UI is alive).
    Swap with real adapters later (GraceDB/TNS/GCN/ATel).
    """
    srcs = enabled_sources or {"LVK": True}
    demo: List[Event] = []
    if srcs.get("LVK", True):
        demo.append(Event(
            id="demo-lvk-001",
            when=datetime.utcnow(),
            ra=None, dec=None,
            instruments=["LVK"],
            topics=["binary black hole", "gravitational waves", "GW"],
            summary="Public alert: candidate BBH merger in O4 window.",
            src="LVK",
        ))
    return demo


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fresh_papers(max_results: int = 25) -> List[Paper]:
    """
    Pulls *real* latest arXiv entries from astro-ph (HE/IM/CO).
    """
    if not requests:
        return []

    query = "astro-ph.HE+OR+astro-ph.IM+OR+astro-ph.CO"
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=cat:{query}"
        "&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )

    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "GraviSearch/0.1"})
        r.raise_for_status()
        text = r.text
    except Exception:
        return []

    # Find all <entry>...</entry> blocks
    entries = re.findall(r"<entry>(.*?)</entry>", text, flags=re.S | re.I)
    papers: List[Paper] = []

    for chunk in entries:
        # title
        m_title = re.search(r"<title>(.*?)</title>", chunk, flags=re.S | re.I)
        title = unescape(m_title.group(1).strip()) if m_title else ""
        # summary
        m_sum = re.search(r"<summary>(.*?)</summary>", chunk, flags=re.S | re.I)
        summary = unescape(m_sum.group(1).strip()) if m_sum else ""
        # id url (abs link)
        m_id = re.search(r"<id>(.*?)</id>", chunk, flags=re.S | re.I)
        id_url = (m_id.group(1).strip() if m_id else "")

        # arXiv id extraction
        arx_id: Optional[str] = None
        m_new = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", id_url)
        if m_new:
            arx_id = m_new.group(1)
        else:
            tail = id_url.rsplit("/", 1)[-1] if id_url else ""
            if tail:
                arx_id = tail

        if not arx_id:
            continue

        abs_url = f"https://arxiv.org/abs/{arx_id}"

        # updated time
        when = datetime.utcnow()
        m_upd = re.search(r"<updated>(.*?)</updated>", chunk, flags=re.S | re.I) or \
                re.search(r"<published>(.*?)</published>", chunk, flags=re.S | re.I)
        if m_upd:
            iso = m_upd.group(1).strip().replace("Z", "+00:00")
            try:
                when = datetime.fromisoformat(iso)
            except Exception:
                pass

        # categories → keywords
        kws = [m.group(1)
               for m in re.finditer(r'<category[^>]*term="([^"]+)"', chunk, flags=re.I)]

        papers.append(Paper(
            id=arx_id, when=when, title=title, abstract=summary, url=abs_url, kws=kws
        ))

        if len(papers) >= max_results:
            break

    return papers


# ------------------ MATCHING ------------------

TOP_K_MATCHES = 3  # show only the top three matches

def _event_text(e: Event) -> str:
    return " ; ".join([
        " ".join(e.topics or []),
        " ".join(e.instruments or []),
        e.summary or ""
    ]).strip()


def _paper_text(p: Paper) -> str:
    return f"{p.title}. {p.abstract}".strip()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T  # in [-1, 1]


def _encode_texts(texts: List[str], encoder: Optional[Any]) -> Optional[np.ndarray]:
    if encoder is not None and hasattr(encoder, "encode"):
        try:
            return np.array(encoder.encode(texts, normalize_embeddings=True))
        except Exception:
            return None
    return None


def _tfidf_sim(ev_texts: List[str], pp_texts: List[str]) -> Optional[np.ndarray]:
    if TfidfVectorizer is None or _sk_cos is None:
        return None
    try:
        vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        X = vec.fit_transform(ev_texts + pp_texts)
        E = X[:len(ev_texts)]
        P = X[len(ev_texts):]
        return _sk_cos(E, P)  # in [0,1]
    except Exception:
        return None


def fuse_stories(events: List[Event], papers: List[Paper], encoder: Optional[Any]) -> List[Dict[str, Any]]:
    if not events or not papers:
        return []

    ev_texts = [_event_text(e) for e in events]
    pp_texts = [_paper_text(p) for p in papers]

    sim: Optional[np.ndarray] = None
    E = _encode_texts(ev_texts, encoder)
    P = _encode_texts(pp_texts, encoder) if E is not None else None
    if E is not None and P is not None:
        sim = _cosine_sim(E, P)
        sim = np.maximum(sim, 0.0)
    if sim is None:
        sim = _tfidf_sim(ev_texts, pp_texts)
    if sim is None:
        sim = np.zeros((len(events), len(papers)))
        def toks(x: str) -> set[str]:
            return set(re.findall(r"[a-zA-Z0-9\-\+]{2,}", x.lower()))
        ev_k = [toks(t) for t in ev_texts]
        pp_k = [toks(t) for t in pp_texts]
        for i, ek in enumerate(ev_k):
            for j, pk in enumerate(pp_k):
                inter = len(ek & pk)
                union = len(ek | pk) or 1
                sim[i, j] = inter / union

    stories: List[Dict[str, Any]] = []
    for i, e in enumerate(events):
        order = np.argsort(-sim[i])[:TOP_K_MATCHES]
        matches = []
        for j in order:
            p = papers[j]
            raw = float(sim[i, j])
            score_pct = max(0.0, min(1.0, raw)) * 100.0
            matches.append({
                "paper": p.id,
                "title": p.title,
                "url": p.url,
                "score_pct": round(score_pct, 1)
            })
        stories.append({
            "event": e,
            "matches": matches,
        })
    return stories


# ------------------ REAL EVIDENCE ESTIMATOR ------------------

def _normalize_text(t: str) -> str:
    """Unify dashes/spacing and lowercase text."""
    t = t.lower()
    t = re.sub(r"[\u2010-\u2015]", "-", t)  # all dash types -> "-"
    t = re.sub(r"\s+", " ", t)
    return t

_OBS_TERMS = {
    "observation", "observations", "observational", "survey", "data", "light curve", "lightcurve",
    "spectrum", "spectra", "spectroscopic", "spectroscopy", "imaging", "image", "flux", "photometry",
    "detected", "detection", "measure", "measured", "measurement", "catalog", "archival",
    "alma", "jwst", "hst", "chandra", "xmm", "nustar", "vla", "meerkat", "lofar", "keck", "vlt",
    "subaru", "gaia", "ztf", "pan-starrs", "lsst", "ligo", "virgo", "kagra", "icecube", "fermi",
    "swift", "nicer", "integral", "hess", "magic", "veritas", "cta", "erosita", "sdss", "des", "desi",
    "askap", "gmrt", "noema", "parkes", "atca"
}

# Expanded simulation terms (broad coverage for GW/astro papers)
_SIM_TERMS = {
    # generic cores
    "simulation", "simulations", "simulate", "simulated", "simulating", "simulation-based", "simulation based",
    "numerical", "numerically", "computational",
    "forward model", "forward modeling", "forward modelling",
    # methods / phrases
    "n-body", "nbody", "hydrodynamic", "hydrodynamics", "hydrodynamical", "hydro",
    "mhd", "grmhd", "pic", "particle-in-cell", "particle in cell",
    "monte carlo", "monte-carlo",
    "radiative transfer", "ray-tracing", "ray tracing",
    "mock", "synthetic",
    # common codes / families
    "sph", "arepo", "athena++", "gadget", "enzo", "flash", "zeus", "bhac", "harm", "pluto",
    # GW-specific sim/model tooling often used with NR
    "numerical relativity", "nr", "nr-based", "nr sur", "nrsur", "surrogate", "surrogates",
    "eob", "seob", "seobnr", "imr", "imrphenom", "phenom", "waveform model", "waveforms", "waveform",
    # numerics & grids
    "finite-volume", "finite difference", "finite-difference",
    "finite element", "adaptive mesh", "amr", "mesh refinement", "shock-capturing",
    "grid-based", "lattice", "solver", "solvers", "code", "codes", "pipeline"
}

_TH_TERMS = {
    "theory", "theoretical", "analytic", "analytical", "semi-analytic", "semi analytic", "framework",
    "model", "models", "we derive", "derivation", "prove", "constraint", "constraints",
    "parameterization", "parametrization", "phenomenological", "phenomenology", "approximation",
    "post-newtonian", "effective field theory", "eft"
}

_CAT_HINTS = {
    "astro-ph.im": ("obs", 2.0),
    "astro-ph.ga": ("obs", 1.5),
    "astro-ph.sr": ("obs", 1.2),
    "astro-ph.co": ("sim", 1.2),   # cosmology often sim-heavy
    "astro-ph.he": ("sim", 0.6),   # high-energy often uses simulations
    "gr-qc":       ("th", 1.6),
    "hep-th":      ("th", 1.8),
    "math-ph":     ("th", 1.5),
    "physics.comp-ph": ("sim", 1.6),
}

def _count_hits(text: str, terms: set[str]) -> int:
    t = _normalize_text(text)
    hits = 0
    for term in terms:
        q = term.lower()
        # escape, then allow any hyphen variant where a hyphen appears in the term
        pattern = re.escape(q).replace(r"\-", r"[-\u2010-\u2015]")
        # word-ish boundaries that still allow hyphenated compounds
        rx = re.compile(rf"(?<![\w]){pattern}(?![\w])")
        hits += len(rx.findall(t))
    return hits

# extra hint detector to avoid exact zeros when clear sim cues exist
_SIM_HINTS = {"waveform", "seob", "seobnr", "imr", "imrphenom", "phenom", "surrogate", "numerical relativity", "nr"}

def _has_sim_hint(p: Paper) -> bool:
    txt = _normalize_text(f"{p.title} {p.abstract}")
    for h in _SIM_HINTS:
        if re.search(rf"(?<!\w){re.escape(h)}(?!\w)", txt):
            return True
    # category-based hint
    for cat in (p.kws or []):
        if "astro-ph.co" in cat.lower() or "physics.comp-ph" in cat.lower():
            return True
    return False

def _evidence_from_paper(p: Paper, w: float) -> Dict[str, float]:
    txt = f"{p.title} {p.abstract}"
    obs = float(_count_hits(txt, _OBS_TERMS))
    sim = float(_count_hits(txt, _SIM_TERMS))
    th  = float(_count_hits(txt, _TH_TERMS))

    for cat in (p.kws or []):
        c = (cat or "").lower()
        for key, (bucket, gain) in _CAT_HINTS.items():
            if key in c:
                if bucket == "obs": obs += gain
                elif bucket == "sim": sim += gain
                elif bucket == "th":  th  += gain

    obs *= w; sim *= w; th *= w
    return {"obs": obs, "sim": sim, "th": th}

def compute_evidence_for_story(event: Event, matches: List[Dict[str, Any]], pmap: Dict[str, Paper]) -> Dict[str, float]:
    obs = sim = th = 0.0
    sim_hint_present = False

    for m in matches:
        pid = m.get("paper")
        p = pmap.get(pid)
        if not p:
            continue
        w = max(0.0, float(m.get("score_pct", 0.0))) / 100.0
        scores = _evidence_from_paper(p, w)
        obs += scores["obs"]; sim += scores["sim"]; th += scores["th"]
        # track hints even if sim terms didn't directly match
        sim_hint_present = sim_hint_present or _has_sim_hint(p)

    # smart epsilon smoothing: avoid exact zero if strong sim hints exist
    if sim == 0.0 and sim_hint_present:
        sim += 0.8  # tiny pseudo-count

    total = obs + sim + th
    if total <= 0:
        return {"observational": 0.0, "simulation": 0.0, "theory": 0.0}

    return {
        "observational": obs / total,
        "simulation":    sim / total,
        "theory":        th  / total,
    }


# ------------------ EVIDENCE BLOCK (HTML + renderer) ------------------

def _evidence_block_html(obs: float = 0.0, sim: float = 0.0, th: float = 0.0) -> str:
    """
    Animated bars with smooth grow + subtle moving stripes; white labels.
    Uses 1-decimal text and a minimum visible width (~1.2%) for any nonzero value
    so tiny shares don’t look like 0.
    """
    def pctf(x: float) -> float:
        try:
            x = float(x or 0.0)
        except Exception:
            x = 0.0
        x = max(0.0, min(1.0, x))
        return x * 100.0

    po, ps, pt = pctf(obs), pctf(sim), pctf(th)
    # Ensure tiny but nonzero values are visible with a min-width
    wo = (1.2 if po > 0.0 and po < 1.2 else po)
    ws = (1.2 if ps > 0.0 and ps < 1.2 else ps)
    wt = (1.2 if pt > 0.0 and pt < 1.2 else pt)

    return f"""
    <div style="padding:6px 0; font-size:14px; width:100%; max-width:100%; color:#fff;">
      <style>
        .e-row {{ margin:12px 0; width:100%; }}
        .e-label {{ display:flex; align-items:center; margin-bottom:6px; width:100%; color:#fff; }}
        .e-title {{ font-weight:700; color:#fff; }}
        .e-desc {{ font-size:12px; margin-left:8px; color:#fff; }}
        .e-track {{ position:relative; height:14px; width:100%; background:rgba(78,119,217,0.25); border-radius:7px; overflow:hidden; }}
        .e-fill {{
          position:absolute; top:0; left:0; bottom:0; width:0; border-radius:7px;
          animation: grow 900ms ease-out forwards;
        }}
        .e-fill::after {{
          content:""; position:absolute; inset:0;
          background-image: linear-gradient(45deg, rgba(255,255,255,0.20) 25%, transparent 25%, transparent 50%,
                                            rgba(255,255,255,0.20) 50%, rgba(255,255,255,0.20) 75%, transparent 75%, transparent);
          background-size:24px 24px; opacity:0.6; animation: move 1.2s linear infinite;
          mix-blend-mode: overlay;
        }}
        .e-text {{ position:absolute; top:50%; right:10px; transform:translateY(-50%); font-size:12px; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,0.6); pointer-events:none; }}

        @keyframes grow {{ from {{ width:0; }} to {{ width: var(--w); }} }}
        @keyframes move {{ from {{ background-position:0 0; }} to {{ background-position:48px 0; }} }}
      </style>

      <div class="e-row">
        <div class="e-label">
          <div><span class="e-title">Observational</span>&nbsp;<span class="e-desc">direct instrument data (telescopes, GW, spectra, images)</span></div>
        </div>
        <div class="e-track">
          <div class="e-fill" style="--w:{wo:.1f}%; background:linear-gradient(90deg,#4e77d9,#6ea0ff);"></div>
          <div class="e-text">{po:.1f}%</div>
        </div>
      </div>

      <div class="e-row">
        <div class="e-label">
          <div><span class="e-title">Simulation</span>&nbsp;<span class="e-desc">numerical modelling (N-body, GRMHD, radiative transfer)</span></div>
        </div>
        <div class="e-track">
          <div class="e-fill" style="--w:{ws:.1f}%; background:linear-gradient(90deg,#9d4edd,#c07cff);"></div>
          <div class="e-text">{ps:.1f}%</div>
        </div>
      </div>

      <div class="e-row">
        <div class="e-label">
          <div><span class="e-title">Theory</span>&nbsp;<span class="e-desc">analytic frameworks &amp; predictions (no direct data)</span></div>
        </div>
        <div class="e-track">
          <div class="e-fill" style="--w:{wt:.1f}%; background:linear-gradient(90deg,#64b5f6,#9ed1ff);"></div>
          <div class="e-text">{pt:.1f}%</div>
        </div>
      </div>
    </div>
    """

def _render_evidence_block(obs: float = 0.0, sim: float = 0.0, th: float = 0.0) -> None:
    st_html(_evidence_block_html(obs, sim, th), height=260, width=1300, scrolling=False)


# ------------------ ENHANCED UI ------------------

def render_news_section(encoder: Optional[Any] = None, enabled_sources: Optional[Dict[str, bool]] = None) -> None:
    st.markdown("""
    <style>
    .event-desk {
        background: rgba(30, 30, 50, 0.88);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 24px;
        border-left: 4px solid #9d4edd;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        color: #fff;
    }
    .event-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .event-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
        background: rgba(157, 78, 221, 0.15);
        color: #fff;
    }
    .match-card {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #4e77d9;
        transition: transform 0.2s;
        color: #fff;
    }
    .match-card:hover { transform: translateX(4px); }
    .score-badge {
        display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
        background: linear-gradient(90deg, #4e77d9, #9d4edd); color: white;
    }
    .live-pill {
        display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border-radius:999px;
        background:rgba(255, 59, 59, 0.12); border:1px solid rgba(255, 59, 59, 0.35);
        color:#ff4d4f; font-weight:800; letter-spacing:0.6px; text-transform:uppercase; font-size:12px;
    }
    .live-dot {
        width:12px; height:12px; border-radius:50%; background:#ff2d2f;
        box-shadow: 0 0 0 0 rgba(255,45,47,0.7); animation: pulse 1.5s infinite, glow 1.5s infinite;
    }
    .live-dot.small { width:9px; height:9px; }
    @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,45,47,0.7);} 70%{box-shadow:0 0 0 12px rgba(255,45,47,0);} 100%{box-shadow:0 0 0 0 rgba(255,45,47,0);} }
    @keyframes glow  { 0%,100%{filter:drop-shadow(0 0 4px rgba(255,77,79,0.9));} 50%{filter:drop-shadow(0 0 10px rgba(255,77,79,1));} }
    .live-title { display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-start; color:#fff; }
    .live-sub { opacity:0.9; margin-top:4px; color:#fff; }

    /* Animated bars next to the Evidence Analysis title */
    .analysis-anim { display:inline-flex; align-items:center; gap:10px; }
    .an-bars { display:inline-flex; align-items:flex-end; gap:3px; width:26px; height:16px; margin-right:2px; }
    .an-bars span { display:block; width:4px; height:10px; background:#64b5f6; animation: ab 1s infinite ease-in-out; transform-origin: bottom; }
    .an-bars span:nth-child(2){ animation-delay: .1s; background:#9d4edd; }
    .an-bars span:nth-child(3){ animation-delay: .2s; background:#4e77d9; }
    .an-bars span:nth-child(4){ animation-delay: .3s; background:#64b5f6; }
    @keyframes ab { 0%,100%{ transform: scaleY(0.6);} 50%{ transform: scaleY(1.25);} }
    </style>
    """, unsafe_allow_html=True)

    events = fetch_events(enabled_sources)
    papers = fetch_fresh_papers()
    pmap = {p.id: p for p in papers}

    with st.container():
        st.markdown("""
        <div class="live-header">
          <div class="live-title">
            <span class="live-pill"><span class="live-dot"></span>LIVE</span>
            <span style="font-size:1.5rem; font-weight:700;"> Live Event Paper Desk</span>
          </div>
          <div class="live-sub">Real-time research matching for astronomical events</div>
        </div>
        """, unsafe_allow_html=True)

        if not papers:
            st.info("Couldn't fetch arXiv updates. Check connection or try later.")
            return

        stories = fuse_stories(events, papers, encoder)
        for story in stories:
            story["evidence"] = compute_evidence_for_story(story["event"], story["matches"], pmap)

        scores = [m["score_pct"] for s in stories for m in s["matches"]] if stories else []
        top_score = max(scores) if scores else None
        avg_score = (sum(scores) / len(scores)) if scores else None

        cols = st.columns([1, 1, 1, 1])
        with cols[0]:
            st.metric("Active Events", len(events), "Live")
        with cols[1]:
            st.metric("New Papers", len(papers), "Today")
        with cols[2]:
            st.metric("Top Match", f"{top_score:.1f}%" if top_score is not None else "—")
        with cols[3]:
            st.metric("Avg Score", f"{avg_score:.1f}%" if avg_score is not None else "—")

        if not stories:
            st.info("No significant matches found yet.")
            return

        for story in stories:
            event = story["event"]

            st.markdown(f"""
            <div class="event-desk">
                <div class="event-header">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="live-dot small" title="Live"></span>
                        <span class="event-tag">{event.src}</span>
                        <span class="event-tag" style="background:rgba(78, 119, 217, 0.15);">
                            {event.instruments[0] if event.instruments else 'Event'}
                        </span>
                    </div>
                    <small>{event.when.strftime('%Y-%m-%d %H:%M UTC')}</small>
                </div>
                <h4 style="margin:12px 0; color:#fff;">{event.summary}</h4>
            """, unsafe_allow_html=True)

            for match in story["matches"]:
                st.markdown(f"""
                <div class="match-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <a href="{match['url']}" target="_blank" style="color:#64b5f6; text-decoration:none;">
                            <b>{match['title']}</b>
                        </a>
                        <span class="score-badge">{match['score_pct']}% match</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Title row with animated bars to the left
            st.markdown("""
            <div class="analysis-anim" style="margin-top:8px; color:#fff;">
              <span class="an-bars"><span></span><span></span><span></span><span></span></span>
              <span style="font-weight:700;">Evidence Analysis</span>
            </div>
            """, unsafe_allow_html=True)

            # Collapsible section using Streamlit's expander
            ev = story.get("evidence", {"observational": 0.0, "simulation": 0.0, "theory": 0.0})
            with st.expander("Show analysis", expanded=False):
                _render_evidence_block(
                    obs=ev.get("observational", 0.0),
                    sim=ev.get("simulation", 0.0),
                    th=ev.get("theory", 0.0),
                )

                # --------- OPTIONAL DEBUG: Raw contributions per paper (added) ----------
                show_dbg = st.checkbox("Show evidence details for this event", key=f"dbg_{event.id}")
                if show_dbg:
                    for m in story["matches"]:
                        p = pmap.get(m["paper"])
                        if not p:
                            continue
                        w = (m.get("score_pct", 0.0) or 0.0) / 100.0
                        txt = f"{p.title} {p.abstract}"
                        obs_hits = _count_hits(txt, _OBS_TERMS)
                        sim_hits = _count_hits(txt, _SIM_TERMS)
                        th_hits  = _count_hits(txt, _TH_TERMS)

                        # category gains per bucket
                        obs_gain = sim_gain = th_gain = 0.0
                        cats_shown = []
                        for cat in (p.kws or []):
                            cats_shown.append(cat)
                            lc = (cat or "").lower()
                            for key, (bucket, gain) in _CAT_HINTS.items():
                                if key in lc:
                                    if bucket == "obs": obs_gain += gain
                                    elif bucket == "sim": sim_gain += gain
                                    elif bucket == "th":  th_gain  += gain

                        obs_contrib = w * (obs_hits + obs_gain)
                        sim_contrib = w * (sim_hits + sim_gain)
                        th_contrib  = w * (th_hits  + th_gain)

                        st.markdown(f"""
**{p.title}**

- weight (match): `{w:.3f}`  
- **Observational**: hits `{obs_hits}`, cat+ `{obs_gain:.2f}` → weighted **`{obs_contrib:.2f}`**  
- **Simulation**: hits `{sim_hits}`, cat+ `{sim_gain:.2f}` → weighted **`{sim_contrib:.2f}`**  
- **Theory**: hits `{th_hits}`, cat+ `{th_gain:.2f}` → weighted **`{th_contrib:.2f}`**  
- cats: `{", ".join(cats_shown[:8]) if cats_shown else "—"}`
""")

            st.markdown("</div>", unsafe_allow_html=True)  # close .event-desk
