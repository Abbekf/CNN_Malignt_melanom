# pages/2_utvärdering.py
# -----------------------------------------------
# Evaluation page: "insights image" + "expand to chart"
# Uses files already in ./figures:
#   - *_insights.png  (intro images)
#   - *_curve.png / Umap.png / ROC_curve.png / Classification_report.png (plots)
# -----------------------------------------------

from pathlib import Path
import streamlit as st
from streamlit.components.v1 import html


# Måste ligga tidigt
st.set_page_config(page_title="Malignt melanom", layout="wide", initial_sidebar_state="collapsed")
html('<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">', height=0)

# --- MOBIL ULTRA-COMPACT OVERRIDES ---
st.markdown("""
<style>
/* Mindre grundtyp på mobil (gör allt kompaktare) */
@media (max-width: 820px){
  html { font-size: 14px !important; }                      /* sänker basen */
  .block-container { padding: 10px !important; }            /* mindre sidopadding */
  section.main > div { max-width: 100% !important; }        /* låt innehåll använda hela bredden */
}

/* Rubriker (tvinga ned storleken ordentligt) */
h1, .stMarkdown h1, [data-testid="stMarkdownContainer"] h1 {
  font-size: clamp(1.2rem, 4.2vw, 1.7rem) !important;
  line-height: 1.15 !important;
  margin: 0.35rem 0 0.5rem 0 !important;
}
h2, .stMarkdown h2, [data-testid="stMarkdownContainer"] h2 {
  font-size: clamp(1.05rem, 3.6vw, 1.35rem) !important;
  line-height: 1.2 !important;
  margin: 0.4rem 0 0.45rem 0 !important;
}
h3, .stMarkdown h3 {
  font-size: clamp(0.95rem, 3.2vw, 1.2rem) !important;
}

/* Brödtext & listor */
p, .stMarkdown p, [data-testid="stMarkdownContainer"] p, li {
  font-size: clamp(0.95rem, 3.3vw, 1.02rem) !important;
  line-height: 1.45 !important;
  margin: 0.35rem 0 !important;
}

/* Kort/containers med mycket luft: pressa ned padding */
[data-testid="stVerticalBlock"] > div { padding: 0.6rem !important; }

/* Knappar / inputs i full bredd på mobil */
@media (max-width: 820px){
  .stButton>button, .stDownloadButton>button,
  .stTextInput input, .stNumberInput input, .stSelectbox, .stFileUploader {
    width: 100% !important;
  }
}

/* Bilder & diagram: fyll bredd men inte mer än skärmen höjdmässigt */
.stImage img, .stPyplot, .stPlotlyChart, .stAltairChart {
  width: 100% !important; height: auto !important; max-height: 80vh !important;
}

/* Tabs: gör listan scroll-bar istället för att trycka ihop titlar */
[data-baseweb="tab-list"] {
  overflow-x: auto !important;
  flex-wrap: nowrap !important;
  scrollbar-width: none;
}
[data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
[data-baseweb="tab"] { min-width: max-content !important; }

/* Tab-panel: lite tajtare padding */
[data-baseweb="tab-panel"] { padding: 0.5rem 0 0 0 !important; }

/* Tabeller – horisontell scroll vid behov */
[data-testid="stDataFrame"] { overflow: auto hidden; }

/* Dölj Streamlit-menyer på mobil för ren vy (frivilligt) */
@media (max-width: 820px){ #MainMenu, footer { display: none !important; } }
</style>
""", unsafe_allow_html=True)



st.set_page_config(page_title="Utvärdering", layout="wide")

# ---------- Load global CSS (theme) ----------
def inject_css(path: str = "style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Hittade inte {path}. Kontrollera sökvägen i projektroten.")

inject_css("style.css")  # <-- se till att style.css ligger i repo-roten

# ---------- Small style tweaks (page-local) ----------
st.markdown("""
<style>
.streamlit-expanderHeader { font-weight: 600; font-size: 1.02rem; }
.block-container img { margin-bottom: .25rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Helper ----------
def show_section(title: str, insight_path: str, chart_path: str = None):
    """
    Renders one evaluation section:
      1) the insights image (smaller, centered)
      2) an expander that reveals the real chart underneath
    """
    st.subheader(title)

    # Center the insights image in a middle column
    ip = Path(insight_path)
    if ip.exists():
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            st.image(str(ip), width=700)
    else:
        st.warning(f"Hittade inte insights-bild: {insight_path}")

    # Reveal the underlying chart/plot on click
    with st.expander(f"Visa {title}-grafen"):
        if chart_path is None:
            st.info("Ingen graf angiven ännu.")
            return
        cp = Path(chart_path)
        if cp.exists():
            st.image(str(cp), use_container_width=True)  # new API
        else:
            st.warning(f"Hittade inte graf: {chart_path}")

# ---------- Tabs ----------
tab_loss, tab_acc, tab_cr, tab_cm, tab_roc, tab_umap = st.tabs(
    ["Loss", "Accuracy", "Classification report", "Confusion Matrix", "ROC-kurva", "UMAP"]
)

# ---- LOSS ----
with tab_loss:
    show_section("Loss", "figures/loss_insights.png", "figures/loss_curve.png")

# ---- ACCURACY ----
with tab_acc:
    show_section("Accuracy", "figures/accuracy_insights.png", "figures/accuracy_curve.png")

# ---- CLASSIFICATION REPORT ----
with tab_cr:
    show_section("Classification report", "figures/classification_report_insights.png", "figures/Classification_report.png")

# ---- CONFUSION MATRIX ----
with tab_cm:
    show_section("Confusion Matrix", "figures/confusion_matrix_insights.png", "figures/Confusion_matrix.png")

# ---- ROC ----
with tab_roc:
    show_section("ROC-kurva", "figures/roc_curve_insights.png", "figures/ROC_curve.png")

# ---- UMAP ----
with tab_umap:
    show_section("UMAP", "figures/umap_insights.png", "figures/Umap.png")
