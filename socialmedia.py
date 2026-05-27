# socialmedia.py — sidebar icons-only (round, white, side-by-side, shows disabled if missing)
import streamlit as st

# Your links
SOCIAL_LINKS = {
    "MBN": {
        "LinkedIn":  "https://www.linkedin.com/in/baseetnaseri6/",
        "Github":  "https://github.com/baseetnaseri6",  
        "Website":   "https://baseet.mbnitsolutions.com",
    },
    "Nikhil Shetty": {
        "LinkedIn":  "https://www.linkedin.com/in/nikhil-shetty-027402236/",
        "Github": "",  # add later
        "Website":   "",
    },
}

# Inline SVG (inherits currentColor → we set to white via CSS)
SVG_LINKEDIN = """
<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
<path d="M4.98 3.5C4.98 4.88 3.86 6 2.5 6S0 4.88 0 3.5 1.12 1 2.5 1s2.48 1.12 2.48 2.5zM0 8h5v16H0V8zm7.5 0h4.8v2.2h.07c.67-1.2 2.3-2.46 4.73-2.46 5.06 0 6 3.33 6 7.66V24h-5v-6.9c0-1.64-.03-3.75-2.29-3.75-2.29 0-2.64 1.79-2.64 3.64V24h-5V8z"/>
</svg>
"""
# ✅ GitHub icon (replaces Instagram)
SVG_GITHUB = """
<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" role="img" aria-label="GitHub">
  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.39.6.11.82-.26.82-.58
           0-.29-.01-1.06-.02-2.08-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76
           -1.09-.74.08-.73.08-.73 1.2.08 1.83 1.23 1.83 1.23 1.07 1.83 2.8 1.3 3.48.99
           .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.34-5.47-5.95 0-1.31.47-2.38 1.24-3.22
           -.12-.3-.54-1.52.12-3.16 0 0 1.01-.32 3.3 1.23.96-.27 1.98-.4 3-.41
           1.02 0 2.04.14 3 .41 2.28-1.55 3.29-1.23 3.29-1.23.66 1.64.24 2.86.12 3.16
           .77.84 1.24 1.91 1.24 3.22 0 4.62-2.8 5.65-5.48 5.95.43.37.81 1.1.81 2.22
           0 1.6-.01 2.88-.01 3.27 0 .32.22.69.83.57C20.57 21.8 24 17.3 24 12
           24 5.37 18.63 0 12 0z"/>
</svg>
"""
SVG_GLOBE = """
<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
<path d="M12 2a10 10 0 100 20 10 10 0 000-20zm7.94 9h-3.17a16.9 16.9 0 00-1.05-4.33A8.04 8.04 0 0119.94 11zM12 4c.98 0 2.67 2.27 3.36 6H8.64C9.33 6.27 11.02 4 12 4zM6.28 6.67A16.9 16.9 0 005.23 11H2.06a8.04 8.04 0 014.22-4.33zM4.06 13h3.17c.2 1.51.57 2.97 1.05 4.33A8.04 8.04 0 014.06 13zM12 20c-.98 0-2.67-2.27-3.36-6h6.72C14.67 17.73 12.98 20 12 20zm5.72-2.67c.48-1.36.85-2.82 1.05-4.33h3.17a8.04 8.04 0 01-4.22 4.33z"/>
</svg>
"""

# CSS — smaller, round, white; always side-by-side; disabled style if missing link
_CSS = """
<style>
.sm-wrap{margin:.25rem 0 .5rem 0;}
.sm-row{
  display:flex;
  gap:1.1rem; /* increased gap for more space */
  align-items:center;
  flex-wrap:nowrap;
}
.sm-icon{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:36px;
  height:36px;
  border-radius:50%;
  background:rgba(255,255,255,0.10);
  border:1.5px solid rgba(255,255,255,0.22);
  color:#fff; /* pure white */
  text-decoration:none;
  transition:
    transform .18s cubic-bezier(.4,2,.6,1),
    background .18s,
    box-shadow .18s,
    opacity .12s;
  box-shadow:0 2px 8px 0 rgba(0,0,0,0.08);
  will-change:transform,box-shadow;
}
.sm-icon:hover, .sm-icon:focus {
  background:rgba(255,255,255,0.22);
  transform:translateY(-3px) scale(1.08) rotate(-2deg);
  box-shadow:0 6px 18px 0 rgba(0,0,0,0.13);
  outline:none;
}
.sm-icon svg{
  width:22px;
  height:22px;
  display:block;
  color:#fff; /* ensure SVG is white */
  fill:#fff;
  transition:filter .18s;
  filter:drop-shadow(0 0 2px #fff8);
}
.sm-icon.disabled{
  opacity:.38;
  filter:grayscale(70%);
  cursor:not-allowed;
  pointer-events:none;
  background:rgba(255,255,255,0.08);
  border-style:dashed;
}
</style>
"""

def _icon(href: str, svg: str, label: str):
    if href:
        return f'<a class="sm-icon" href="{href}" target="_blank" rel="noopener" aria-label="{label}" title="{label}">{svg}</a>'
    # show disabled placeholder so icons don’t disappear if link missing
    return f'<span class="sm-icon disabled" aria-label="{label}" title="{label} (add link to enable)">{svg}</span>'

def render_social_sidebar():
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("#### 🌐 Social Media")
    person = st.selectbox("Select a person", list(SOCIAL_LINKS.keys()), key="sm_person_icons")
    links = SOCIAL_LINKS.get(person, {})

    html = '<div class="sm-wrap"><div class="sm-row">'
    html += _icon(links.get("LinkedIn",""),  SVG_LINKEDIN,  f"{person} — LinkedIn")
    html += _icon(links.get("Github",""), SVG_GITHUB, f"{person} — Github")
    html += _icon(links.get("Website",""),   SVG_GLOBE,     f"{person} — Website")
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)
