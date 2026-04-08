from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Utvärdering", layout="wide")

def inject_css(path: str = "style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Hittade inte {path}. Kontrollera sökvägen i projektroten.")

inject_css("style.css")
st.markdown("""
<style>
.streamlit-expanderHeader { font-weight: 600; font-size: 1.02rem; }
.block-container img { margin-bottom: .25rem; }
</style>
""", unsafe_allow_html=True)

def show_section(title: str, insight_path: str, chart_path: str = None):
    """
    Renders one evaluation section:
      1) the insights image (smaller, centered)
      2) an expander that reveals the real chart underneath
    """
    st.subheader(title)

    ip = Path(insight_path)
    if ip.exists():
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            st.image(str(ip), width=700)
    else:
        st.warning(f"Hittade inte insights-bild: {insight_path}")

    with st.expander(f"Visa {title}-grafen"):
        if chart_path is None:
            st.info("Ingen graf angiven ännu.")
            return
        cp = Path(chart_path)
        if cp.exists():
            st.image(str(cp), use_container_width=True)
        else:
            st.warning(f"Hittade inte graf: {chart_path}")

tab_loss, tab_acc, tab_cr, tab_cm, tab_roc, tab_umap = st.tabs(
    ["Loss", "Accuracy", "Classification report", "Confusion Matrix", "ROC-kurva", "UMAP"]
)

with tab_loss:
    show_section("Loss", "figures/loss_insights.png", "figures/loss_curve.png")

with tab_acc:
    show_section("Accuracy", "figures/accuracy_insights.png", "figures/accuracy_curve.png")

with tab_cr:
    show_section("Classification report", "figures/classification_report_insights.png", "figures/Classification_report.png")

with tab_cm:
    show_section("Confusion Matrix", "figures/confusion_matrix_insights.png", "figures/Confusion_matrix.png")

with tab_roc:
    show_section("ROC-kurva", "figures/roc_curve_insights.png", "figures/ROC_curve.png")

with tab_umap:
    show_section("UMAP", "figures/umap_insights.png", "figures/Umap.png")
