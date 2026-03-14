import os, tempfile, time
import streamlit as st
import pandas as pd
import plotly.express as px

from model import load_model, classify_card
from pdf_utils import get_total_pages, iter_page_crops

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Voter Roll Card Classifier",
    page_icon="🗳️",
    layout="wide"
)

# ─────────────────────────────────────────────
#  AUTO-DETECT MODEL FROM REPO
# ─────────────────────────────────────────────
def find_model_in_repo():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(app_dir):
        if f.endswith(".pth"):
            return os.path.join(app_dir, f)
    return None

REPO_MODEL_PATH = find_model_in_repo()

# ─────────────────────────────────────────────
#  CACHE MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def get_model(model_path):
    return load_model(model_path)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# ── Model ──
st.sidebar.subheader("🤖 Model")
if REPO_MODEL_PATH:
    model_name   = os.path.basename(REPO_MODEL_PATH)
    st.sidebar.success(f"✅ Auto-loaded: **{model_name}**")
    model_file   = None
    model_ready  = True
    model_source = REPO_MODEL_PATH
else:
    st.sidebar.warning("⚠️ No .pth found in repo — upload manually")
    model_file   = st.sidebar.file_uploader(
        "Upload trained model (.pth)", type=["pth"]
    )
    model_ready  = model_file is not None
    model_source = None

# ── Page Range only ──
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Page Range")
skip_first = st.sidebar.number_input("Skip first N pages", value=2, min_value=0)
skip_last  = st.sidebar.number_input("Skip last N pages",  value=1, min_value=0)

# ── Fixed grid constants (not exposed to user) ──
COLS      = 3
ROWS      = 10
HEADER_PX = 120
FOOTER_PX = 110
MARGIN_L  = 45
MARGIN_R  = 45

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.title("🗳️ Voter Roll Card Classifier")
st.markdown(
    "Upload a voter roll PDF — the app extracts every card and classifies it as "
    "**Active**, **Deleted**, **Adjudication**, or **Empty**."
)

CLASS_COLORS = {
    "active"      : "#4CAF50",
    "deleted"     : "#F44336",
    "adjudication": "#FF9800",
    "empty"       : "#9E9E9E",
}

# ── File uploader ──
pdf_file = st.file_uploader("📄 Upload Voter Roll PDF", type=["pdf"])

# ── Store PDF bytes in session state immediately on upload ──
if pdf_file is not None:
    st.session_state["pdf_bytes"] = pdf_file.read()
    st.session_state["pdf_name"]  = pdf_file.name

pdf_ready = "pdf_bytes" in st.session_state

# ── Show loaded file + clear button ──
if pdf_ready:
    col_info, col_clear = st.columns([5, 1])
    col_info.success(f"📄 Ready: **{st.session_state['pdf_name']}**")
    if col_clear.button("🗑️ Clear"):
        del st.session_state["pdf_bytes"]
        del st.session_state["pdf_name"]
        if "df" in st.session_state:
            del st.session_state["df"]
        st.rerun()

# ─────────────────────────────────────────────
#  RUN INFERENCE
# ─────────────────────────────────────────────
if st.button("🚀 Run Classification",
             disabled=(not pdf_ready or not model_ready)):

    # ── Save PDF from session state to temp file ──
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(st.session_state["pdf_bytes"])
        pdf_path = f.name

    # ── Resolve model path ──
    if model_source:
        final_model_path = model_source
        cleanup_model    = False
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp.write(model_file.read())
        tmp.close()
        final_model_path = tmp.name
        cleanup_model    = True

    # ── Load model (cached) ──
    with st.spinner("Loading model..."):
        model, class_names, idx_to_class = get_model(final_model_path)
    st.success(f"✅ Model loaded  |  Classes: {class_names}")

    # ── Page range ──
    total_pages = get_total_pages(pdf_path)
    first_page  = skip_first + 1
    last_page   = total_pages - skip_last
    total_cards = (last_page - first_page + 1) * COLS * ROWS

    st.info(
        f"📄 PDF has **{total_pages}** pages  |  "
        f"Processing pages **{first_page} → {last_page}**  |  "
        f"Expected cards: **{total_cards}**"
    )

    # ── Progress UI ──
    progress_bar = st.progress(0)
    status_text  = st.empty()
    records      = []
    cards_done   = 0
    start_time   = time.time()

    for page_num, c_idx, crop_pil in iter_page_crops(
        pdf_path, first_page, last_page,
        COLS, ROWS, HEADER_PX, FOOTER_PX, MARGIN_L, MARGIN_R
    ):
        pred, conf, probs, raw, fallback = classify_card(
            crop_pil, model, idx_to_class
        )
        records.append({
            "page"              : page_num,
            "card_idx"          : c_idx,
            "row"               : c_idx // COLS + 1,
            "col"               : c_idx % COLS + 1,
            "prediction"        : pred,
            "raw_prediction"    : raw,
            "confidence"        : conf,
            "fallback_triggered": fallback,
            **{f"prob_{k}": v for k, v in probs.items()}
        })
        cards_done += 1
        progress_bar.progress(cards_done / total_cards)
        if cards_done % 10 == 0 or cards_done == total_cards:
            elapsed = time.time() - start_time
            eta     = (elapsed / cards_done) * (total_cards - cards_done)
            status_text.markdown(
                f"Processing card **{cards_done}/{total_cards}** "
                f"| Page {page_num} "
                f"| ⏱️ {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
            )

    df = pd.DataFrame(records)
    status_text.success(
        f"🎉 Done! Classified {len(df)} cards in {time.time()-start_time:.1f}s"
    )

    st.session_state["df"]          = df
    st.session_state["class_names"] = class_names

    # ── Cleanup ──
    os.unlink(pdf_path)
    if cleanup_model:
        os.unlink(final_model_path)

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if "df" in st.session_state:
    df          = st.session_state["df"]
    class_names = st.session_state["class_names"]
    counts      = df["prediction"].value_counts()

    st.markdown("---")
    st.header("📊 Results")

    # ── KPI row ──
    cols_ui = st.columns(len(class_names) + 1)
    cols_ui[0].metric("📦 Total Cards", len(df))
    for i, cls in enumerate(class_names):
        cnt = counts.get(cls, 0)
        pct = cnt / len(df) * 100
        cols_ui[i + 1].metric(cls.capitalize(), cnt, f"{pct:.1f}%")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🥧 Class Distribution")
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map=CLASS_COLORS, hole=0.4
        )
        fig_pie.update_traces(textinfo="label+percent+value")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("📈 Confidence Distribution")
        fig_hist = px.histogram(
            df, x="confidence", color="prediction",
            color_discrete_map=CLASS_COLORS,
            nbins=30, barmode="overlay", opacity=0.75
        )
        fig_hist.add_vline(x=0.60, line_dash="dash",
                           line_color="red", annotation_text="0.60")
        fig_hist.add_vline(x=0.85, line_dash="dash",
                           line_color="blue", annotation_text="0.85")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col3:
        st.subheader("📄 Cards per Page")
        pivot = df.groupby(["page", "prediction"]).size().reset_index(name="count")
        fig_bar = px.bar(
            pivot, x="page", y="count", color="prediction",
            color_discrete_map=CLASS_COLORS, barmode="stack"
        )
        fig_bar.update_layout(xaxis_title="Page", yaxis_title="Cards")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Heatmap ──
    st.subheader("🔥 Deleted Cards Heatmap (Page × Row)")
    deleted_df = df[df["prediction"] == "deleted"]
    if len(deleted_df) > 0:
        heat_data = deleted_df.groupby(["page", "row"]).size().reset_index(name="count")
        fig_heat  = px.density_heatmap(
            heat_data, x="page", y="row", z="count",
            color_continuous_scale="Reds"
        )
        fig_heat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No deleted cards found.")

    # ── Fallback warnings ──
    fallbacks = df[df["fallback_triggered"]]
    if len(fallbacks) > 0:
        st.warning(
            f"⚠️ {len(fallbacks)} card(s) reassigned to active due to low confidence"
        )
        st.dataframe(
            fallbacks[["page", "card_idx", "row", "col",
                        "raw_prediction", "confidence",
                        *[f"prob_{c}" for c in class_names]]],
            use_container_width=True
        )

    # ── Full table ──
    st.subheader("📋 Full Results Table")
    filter_cls = st.multiselect(
        "Filter by class", options=class_names, default=class_names
    )
    st.dataframe(
        df[df["prediction"].isin(filter_cls)],
        use_container_width=True, height=400
    )

    # ── Download ──
    st.markdown("---")
    st.download_button(
        "⬇️ Download Full Results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="voter_roll_classification.csv",
        mime="text/csv"
    )
