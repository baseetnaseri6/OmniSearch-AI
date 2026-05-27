# about.py — Sidebar "About" button + centered modal popup with top-right (X)
import streamlit as st
from streamlit.components.v1 import html as components_html

ABOUT_MD = """
### About this platform
**AI Review Copilot** helps you ask research questions, mine claims from papers, and score evidence with per-paper quotes.  
**Live Event Paper Desk** matches fresh arXiv papers to live GW/astro events and summarizes the *observational vs simulation vs theory* balance.  
**Extras**: soundscapes for focus, quick social links, exports (CSV/JSON), and a sleek heatmap for claim support/refutation.

**Why it’s useful**
- Save hours on literature triage with claim matrices and quotes.
- See conflicts at a glance (support vs refute).
- Keep your review reproducible with dataset/code hints.

**Tips**
- Load abstracts or PDFs for best results (≥ 500 chars each).
- Try *Power Mode* for chunked retrieval + reranker + NLI.
- Use the **Online** tab or uploads to widen the corpus.

Built for fast, explainable research workflows ✨
"""

def render_about_sidebar_button(label: str = "About"):
    # One-time init
    if "about_open" not in st.session_state:
        st.session_state["about_open"] = False
    # Sidebar button
    if st.button(label, key="btn_about_sidebar", use_container_width=True):
        st.session_state["about_open"] = True

def render_about_modal():
    """Show a centered popup. Uses native st.modal when available; else a JS-closable overlay."""
    if not st.session_state.get("about_open"):
        return

    # 1) Preferred: Streamlit native modal (centered). Add an 'X' button inside.
    try:
        with st.modal("About", key="about_modal"):
            # right-aligned X inside the modal box
            c1, c2 = st.columns([1, 0.12])
            with c2:
                if st.button("✕", key="about_close_x", help="Close"):
                    st.session_state["about_open"] = False
                    st.rerun()

            # your original markdown body
            st.markdown(ABOUT_MD)
        return
    except Exception:
        pass

    # 2) Fallback: centered overlay with a real inside top-right X (uses JS)
    # Convert markdown → HTML for nicer formatting (if python-markdown is installed)
    try:
        from markdown import markdown as _md
        content_html = _md(ABOUT_MD)
    except Exception:
        esc = (ABOUT_MD.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
        content_html = f"<pre style='white-space:pre-wrap; font-family:inherit; margin:0;'>{esc}</pre>"

    html_code = """
    <div id="about-overlay" style="
      position:fixed; inset:0; display:flex; align-items:center; justify-content:center;
      background:rgba(0,0,0,.55); z-index:99999;">
      <div style="
        position:relative; max-width:640px; width:92%;
        background:rgba(30,30,50,.98);
        border:1px solid rgba(255,255,255,.12); border-radius:14px;
        box-shadow:0 12px 40px rgba(0,0,0,.45); padding:20px; color:#fff;
        font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
        <!-- inside-box X -->
        <button id="about-close-btn" aria-label="Close" title="Close" style="
          position:absolute; top:10px; right:10px;
          width:32px; height:32px; border-radius:50%;
          background:rgba(255,255,255,.08);
          border:1px solid rgba(255,255,255,.24); color:#fff;
          cursor:pointer; font-size:18px; line-height:0;">&times;</button>

        <h3 style="margin:0 0 8px 0;">About</h3>
        <div style="font-size:.95rem; line-height:1.55;">{content}</div>
      </div>
    </div>
    <script>
      (function(){{
        const overlay = document.getElementById('about-overlay');
        const btn = document.getElementById('about-close-btn');
        if (btn) btn.addEventListener('click', ()=> overlay && overlay.remove());
        document.addEventListener('keydown', (e)=>{{ if(e.key==='Escape') overlay && overlay.remove(); }});
      }})();
    </script>
    """.format(content=content_html)  # doubled {{ }} keep JS braces intact

    components_html(html_code, height=520)

    # Ensure it won't pop back on the next rerun
    st.session_state["about_open"] = False
