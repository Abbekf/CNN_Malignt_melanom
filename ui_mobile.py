# ui_mobile.py
import streamlit as st

def mobile_only_css():
    """Inject mobile-only responsive CSS without altering desktop layout."""
    try:
        st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass

    st.markdown("""
    <style>
    /* --- DESKTOP GUARD: nothing changes on ≥ 768px --- */
    @media (min-width: 768px) {
      .block-container { padding-left: 1.5rem; padding-right: 1.5rem; }
      [data-testid="column"] { flex: 1 1 0% !important; } /* default column behavior */
      .stButton > button { width: auto !important; }       /* default button width */
      [data-testid="stSidebar"] { width: 20rem !important; min-width: 20rem !important; }
      html { zoom: 1 !important; } /* in case zoom was tried earlier */
    }

    /* --- MOBILE: apply tweaks only on < 768px --- */
    @media (max-width: 767.98px) {
      /* Compact paddings */
      .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }

      /* Stack columns on mobile */
      [data-testid="column"] { flex-basis: 100% !important; width: 100% !important; }
      [data-testid="column"] > div { padding-right: 0 !important; }

      /* Full-width buttons on mobile */
      .stButton > button { width: 100% !important; }

      /* Images and charts should never overflow */
      [data-testid="stImage"] img { max-width: 100% !important; height: auto !important; }
      .element-container:has(canvas), .stAltairChart { max-width: 100% !important; }

      /* Tabs: allow horizontal scroll if many */
      [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; }
      [data-baseweb="tab"] { white-space: nowrap; }

      /* Inputs/selects fill available width */
      [data-baseweb="input"], [data-baseweb="select"] { width: 100% !important; }

      /* File uploader: compact on phones */
      [data-testid="stFileUploaderDropzone"] { min-height: 64px !important; padding: .5rem !important; }
      [role="button"][tabindex="0"] { font-size: .95rem !important; }

      /* Tables: horizontal scroll instead of squishing text */
      .stDataFrame, .stTable { overflow-x: auto; }

      /* Sidebar stays tucked away on mobile (user can still open it) */
      [data-testid="stSidebar"] { width: 0 !important; min-width: 0 !important; }

      /* Slight font adjustments for very small screens */
      html, body, [data-testid="stMarkdownContainer"] { font-size: 15px; }
      h1 { font-size: 1.4rem; } h2 { font-size: 1.2rem; } h3 { font-size: 1.05rem; }
    }
    </style>
    """, unsafe_allow_html=True)
