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
#  CUSTOM CSS — bigger, catchier KPI cards
# ─────────────────────────────────────────────
st.markdown("""
<style>
.kpi-box {
    border-radius: 14px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    margin: 6px 0;
}
.kpi-total   { background: linear-gradient(135deg, #1a1a2e, #16213e); color: #ffffff; }
.kpi-active  { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); color: #1b5e20; }
.kpi-deleted { background: linear-gradient(135deg, #ffebee, #ffcdd2); color: #b71c1c; }
.kpi-adj     { background: linear-gradient(135deg, #fff3e0, #ffe0b2); color: #e65100; }
.kpi-empty   { background: linear-gradient(135deg, #f5f5f5, #eeeeee); color: #424242; }
.kpi-voters  { background: linear-gradient(135deg, #e3f2fd, #bbdefb); color: #0d47a1; }

.kpi-label {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 4px 0;
}
.kpi-total .kpi-value { font-size: 4rem; }
.kpi-pct {
    font-size: 1.0rem;
    font-weight: 600;
    opacity: 0.75;
    margin-top: 4px;
}
.section-divider {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def get_model(model_path):
    return load_model(model_path)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("🤖 Model")
if REPO_MODEL_PATH:
    model_name   = os.path.basename(REPO_MODEL_PATH)
    st.sidebar.success(f"✅ Auto-loaded: **{model_name}**")
    model_file   = None
    model_ready  = True
    model_source = REPO_MODEL_PATH
else:
    st.sidebar.warning("⚠️ No .pth found in repo — upload manually")
    model_file   = st.sidebar.file_uploader("Upload trained model (.pth)", type=["pth"])
    model_ready  = model_file is not None
    model_source = None

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Page Range")
skip_first = st.sidebar.number_input("Skip first N pages", value=2, min_value=0)
skip_last  = st.sidebar.number_input("Skip last N pages",  value=1, min_value=0)

# Fixed grid constants
COLS      = 3
ROWS      = 10
HEADER_PX = 120
FOOTER_PX = 110
MARGIN_L  = 45
MARGIN_R  = 45

CLASS_COLORS = {
    "active"      : "#4CAF50",
    "deleted"     : "#F44336",
    "adjudication": "#FF9800",
    "empty"       : "#9E9E9E",
}

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.title("🗳️ Voter Roll Card Classifier")
st.markdown(
    "Upload a voter roll PDF — the app extracts every card and classifies it as "
    "**Active**, **Deleted**, **Under Adjudication**, or **Empty**."
)

pdf_file = st.file_uploader("📄 Upload Voter Roll PDF", type=["pdf"])

if pdf_file is not None:
    st.session_state["pdf_bytes"] = pdf_file.read()
    st.session_state["pdf_name"]  = pdf_file.name

pdf_ready = "pdf_bytes" in st.session_state

if pdf_ready:
    col_info, col_clear = st.columns([5, 1])
    col_info.success(f"📄 Ready: **{st.session_state['pdf_name']}**")
    if col_clear.button("🗑️ Clear"):
        for k in ["pdf_bytes", "pdf_name", "df", "class_names"]:
            st.session_state.pop(k, None)
        st.rerun()

# ─────────────────────────────────────────────
#  RUN INFERENCE
# ─────────────────────────────────────────────
if st.button("🚀 Run Classification",
             disabled=(not pdf_ready or not model_ready)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(st.session_state["pdf_bytes"])
        pdf_path = f.name

    if model_source:
        final_model_path = model_source
        cleanup_model    = False
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp.write(model_file.read()); tmp.close()
        final_model_path = tmp.name
        cleanup_model    = True

    with st.spinner("Loading model..."):
        model, class_names, idx_to_class = get_model(final_model_path)
    st.success(f"✅ Model loaded  |  Classes: {class_names}")

    total_pages = get_total_pages(pdf_path)
    first_page  = skip_first + 1
    last_page   = total_pages - skip_last
    total_cards = (last_page - first_page + 1) * COLS * ROWS

    st.info(
        f"📄 PDF has **{total_pages}** pages  |  "
        f"Processing pages **{first_page} → {last_page}**  |  "
        f"Expected cards: **{total_cards}**"
    )

    progress_bar = st.progress(0)
    status_text  = st.empty()
    records      = []
    cards_done   = 0
    start_time   = time.time()

    for page_num, c_idx, crop_pil in iter_page_crops(
        pdf_path, first_page, last_page,
        COLS, ROWS, HEADER_PX, FOOTER_PX, MARGIN_L, MARGIN_R
    ):
        pred, conf, probs, raw, fallback = classify_card(crop_pil, model, idx_to_class)
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
    status_text.success(f"🎉 Done! Classified {len(df)} cards in {time.time()-start_time:.1f}s")
    st.session_state["df"]          = df
    st.session_state["class_names"] = class_names

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

    # ── Core counts ──
    n_active  = int(counts.get("active",       0))
    n_deleted = int(counts.get("deleted",       0))
    n_adj     = int(counts.get("adjudication",  0))
    n_empty   = int(counts.get("empty",         0))
    n_total   = len(df)

    # Total VOTERS = active + deleted + under adjudication (excludes empty)
    n_voters  = n_active + n_deleted + n_adj

    st.markdown("---")
    st.header("📊 Results")

    # ══════════════════════════════════════════
    #  BIG KPI ROW — Total cards + Total voters
    # ══════════════════════════════════════════
    top1, top2 = st.columns(2)

    with top1:
        st.markdown(f"""
        <div class="kpi-box kpi-total">
            <div class="kpi-label">📦 Total Cards Processed</div>
            <div class="kpi-value">{n_total:,}</div>
            <div class="kpi-pct">All cards including empty slots</div>
        </div>
        """, unsafe_allow_html=True)

    with top2:
        st.markdown(f"""
        <div class="kpi-box kpi-voters">
            <div class="kpi-label">🗳️ Total Voters</div>
            <div class="kpi-value">{n_voters:,}</div>
            <div class="kpi-pct">Active + Deleted + Under Adjudication</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  4 CLASS KPI CARDS
    # ══════════════════════════════════════════
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        pct = n_active / n_voters * 100 if n_voters else 0
        st.markdown(f"""
        <div class="kpi-box kpi-active">
            <div class="kpi-label">✅ Active</div>
            <div class="kpi-value">{n_active:,}</div>
            <div class="kpi-pct">{pct:.1f}% of voters</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        pct = n_deleted / n_voters * 100 if n_voters else 0
        st.markdown(f"""
        <div class="kpi-box kpi-deleted">
            <div class="kpi-label">🗑️ Deleted</div>
            <div class="kpi-value">{n_deleted:,}</div>
            <div class="kpi-pct">{pct:.1f}% of voters</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        pct = n_adj / n_voters * 100 if n_voters else 0
        st.markdown(f"""
        <div class="kpi-box kpi-adj">
            <div class="kpi-label">⚖️ Under Adjudication</div>
            <div class="kpi-value">{n_adj:,}</div>
            <div class="kpi-pct">{pct:.1f}% of voters</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        pct = n_empty / n_total * 100 if n_total else 0
        st.markdown(f"""
        <div class="kpi-box kpi-empty">
            <div class="kpi-label">⬜ Empty Slots</div>
            <div class="kpi-value">{n_empty:,}</div>
            <div class="kpi-pct">{pct:.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Voter equation ──
    st.markdown(f"""
    <div style='text-align:center; padding:14px; background:#f8f9fa;
                border-radius:10px; margin:16px 0; font-size:1.1rem; color:#555;'>
        🗳️ <b>Voter Count:</b>
        &nbsp; <span style='color:#2e7d32;font-weight:700'>{n_active:,} Active</span>
        &nbsp;+&nbsp;
        <span style='color:#c62828;font-weight:700'>{n_deleted:,} Deleted</span>
        &nbsp;+&nbsp;
        <span style='color:#e65100;font-weight:700'>{n_adj:,} Under Adjudication</span>
        &nbsp;=&nbsp;
        <span style='color:#0d47a1;font-weight:800;font-size:1.25rem'>{n_voters:,} Total Voters</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  CHARTS
    # ══════════════════════════════════════════
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🥧 Class Distribution")
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map=CLASS_COLORS, hole=0.4
        )
        fig_pie.update_traces(textinfo="label+percent+value")
        # Rename adjudication → Under Adjudication in chart
        fig_pie.for_each_trace(lambda t: t.update(
            labels=["Under Adjudication" if l == "adjudication" else l.capitalize()
                    for l in t.labels]
        ) if hasattr(t, "labels") else t)
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

    # Rename adjudication → Under Adjudication in display
    df_display = df.copy()
    df_display["prediction"] = df_display["prediction"].replace(
        {"adjudication": "Under Adjudication"}
    )

    filter_cls = st.multiselect(
        "Filter by class",
        options=["active", "deleted", "Under Adjudication", "empty"],
        default=["active", "deleted", "Under Adjudication", "empty"]
    )
    st.dataframe(
        df_display[df_display["prediction"].isin(filter_cls)],
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
