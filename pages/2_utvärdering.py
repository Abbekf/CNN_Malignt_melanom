# pages/3_utvardering.py
# -----------------------------------------------
# Evaluation page: "insights image" + "expand to chart"
# Uses files already in ./figures:
#   - *_insights.png  (intro images)
#   - *_curve.png / Umap.png / ROC_curve.png / Classification_report.png (plots)
# -----------------------------------------------

from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Utvärdering", layout="wide")

# ---------- Small style tweaks ----------
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
            st.image(str(cp), use_column_width=True)
        else:
            st.warning(f"Hittade inte graf: {chart_path}")


# ---------- Tabs ----------
tab_loss, tab_acc, tab_cr, tab_cm, tab_roc, tab_umap = st.tabs(
    ["Loss", "Accuracy", "Classification report", "Confusion Matrix", "ROC-kurva", "UMAP"]
)

# ---- LOSS ----
with tab_loss:
    show_section(
        title="Loss",
        insight_path="figures/loss_insights.png",
        chart_path="figures/loss_curve.png"
    )

# ---- ACCURACY ----
with tab_acc:
    show_section(
        title="Accuracy",
        insight_path="figures/accuracy_insights.png",
        chart_path="figures/accuracy_curve.png"
    )

# ---- CLASSIFICATION REPORT ----
with tab_cr:
    show_section(
        title="Classification report",
        insight_path="figures/classification_report_insights.png",
        chart_path="figures/Classification_report.png"
    )

# ---- CONFUSION MATRIX ----
with tab_cm:
    show_section(
        title="Confusion Matrix",
        insight_path="figures/confusion_matrix_insights.png",
        chart_path="figures/Confusion_matrix.png"
    )

# ---- ROC ----
with tab_roc:
    show_section(
        title="ROC-kurva",
        insight_path="figures/roc_curve_insights.png",
        chart_path="figures/ROC_curve.png"
    )

# ---- UMAP ----
with tab_umap:
    show_section(
        title="UMAP",
        insight_path="figures/umap_insights.png",
        chart_path="figures/Umap.png"
    )
