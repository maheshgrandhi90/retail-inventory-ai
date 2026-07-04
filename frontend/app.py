"""Smart Shelf Analytics & BI Dashboard (Module 7 — Streamlit).

Upload a shelf image -> YOLO detects products -> SWIN+FAISS classifies each crop -> the app
shows KPIs, interactive analytics charts, and a natural-language Business-Intelligence panel
that answers questions over the accumulated inventory (SQLite).

Run from the repo root:
    source .venv/bin/activate
    KMP_DUPLICATE_LIB_OK=TRUE streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure the repo root is importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from backend import inventory_db as db
from bi_interface import bi_engine
from retrieval import pipeline

st.set_page_config(page_title="Smart Shelf Analytics & BI", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
      div[data-testid="stMetric"] {
          background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
          border: 1px solid #374151; border-radius: 14px; padding: 16px 18px;
      }
      div[data-testid="stMetric"] label p { color: #9ca3af !important; font-size: .8rem; }
      div[data-testid="stMetricValue"] { color: #f9fafb !important; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      .pill { display:inline-block; padding:2px 10px; border-radius:999px;
              background:#1e3a8a; color:#dbeafe; font-size:.75rem; margin-left:6px; }
      .muted { color:#9ca3af; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = px.colors.qualitative.Set3


def _style_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e5e7eb"), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#374151")
    fig.update_yaxes(gridcolor="#374151")
    return fig


st.title("🛒 Smart Shelf Analytics & BI")
llm_on = bi_engine.ollama_available()
st.markdown(
    "<span class='muted'>YOLO detection · SWIN + FAISS retrieval classification · "
    "inventory analytics + natural-language BI</span>"
    f"<span class='pill'>{'LLM: Ollama ✓' if llm_on else 'BI: rule-based'}</span>",
    unsafe_allow_html=True,
)

db.init_db()

with st.sidebar:
    st.header("① Upload & detect")
    if pipeline.classifier_ready():
        st.success("SWIN+FAISS index loaded ✓")
    else:
        st.error("Classifier not ready — check retrieval/assets (see retrieval/README.md).")

    uploaded = st.file_uploader("Shelf image", type=["jpg", "jpeg", "png", "bmp"])
    conf = st.slider("YOLO confidence", 0.05, 0.9, 0.25, 0.05)
    max_crops = st.slider("Max products to classify (0 = all)", 0, 300, 60, 10,
                          help="Cap for speed on CPU. 0 classifies every detected box.")
    save_to_db = st.checkbox("Save scan to inventory history", value=True)
    run = st.button("🔍 Analyze shelf", type="primary", use_container_width=True,
                    disabled=uploaded is None)

    st.divider()
    st.caption("Inventory history")
    s = db.stats()
    st.write(f"Scans: **{s['total_scans']}** · Items: **{s['total_items']}** · "
             f"Categories: **{s['distinct_categories']}**")
    if st.button("🗑️ Clear inventory history", use_container_width=True):
        db.clear_all()
        st.rerun()


if run and uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Detecting products and classifying crops…"):
        result = pipeline.analyze_image(image, conf=conf, max_crops=max_crops)
        records = pipeline.detections_to_records(result)
    st.session_state["result"] = result
    st.session_state["records"] = records
    if save_to_db:
        scan_id = db.save_scan(result, uploaded.name, records)
        st.session_state["last_scan_id"] = scan_id
        st.toast(f"Saved scan #{scan_id} to inventory")

result = st.session_state.get("result")
records = st.session_state.get("records", [])

tab_analyze, tab_analytics, tab_bi, tab_history = st.tabs(
    ["🖼️ Detection", "📊 Analytics", "💬 Business Intelligence", "🗂️ Inventory History"]
)

with tab_analyze:
    if result is None:
        st.info("Upload a shelf image and click **Analyze shelf** to begin.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Products detected", result.num_items)
        c2.metric("Distinct categories", result.distinct_categories)
        c3.metric("Empty shelf space", f"{result.empty_pct*100:.0f}%", result.empty_label)
        c4.metric("Needs review", result.review_count)
        c5.metric("Shelf type", result.shelf_type)

        left, right = st.columns([3, 2])
        with left:
            st.image(result.annotated_image, caption="Detected & classified products",
                     use_container_width=True)
            st.caption(f"YOLO {result.timings.get('yolo_s')}s · "
                       f"classify {result.timings.get('classify_s')}s · "
                       f"{result.timings.get('boxes')} boxes")
        with right:
            df = pd.DataFrame(records)
            st.markdown("**Detected items**")
            st.dataframe(df[["crop_id", "category", "subcategory", "score"]],
                         height=420, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download detections CSV", df.to_csv(index=False).encode(),
                               file_name="detections.csv", use_container_width=True)

with tab_analytics:
    if not records:
        st.info("Run an analysis to see analytics.")
    else:
        df = pd.DataFrame(records)
        cat_counts = bi_engine.category_counts(df)

        a, b = st.columns(2)
        with a:
            st.subheader("Category distribution")
            if not cat_counts.empty:
                cc = cat_counts.rename_axis("category").reset_index(name="count")
                fig = px.bar(cc, x="count", y="category", orientation="h",
                             color="category", color_discrete_sequence=PALETTE)
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(_style_fig(fig, 440), use_container_width=True)
            else:
                st.caption("No confidently classified products.")
        with b:
            st.subheader("Shelf composition")
            if not cat_counts.empty:
                cc = cat_counts.rename_axis("category").reset_index(name="count")
                fig = px.pie(cc, values="count", names="category", hole=0.5,
                             color_discrete_sequence=PALETTE)
                st.plotly_chart(_style_fig(fig, 440), use_container_width=True)

        st.subheader("Category → subcategory breakdown")
        known = df[df["category"].str.lower() != "unknown"]
        if not known.empty:
            grp = known.groupby(["category", "subcategory"]).size().reset_index(name="count")
            fig = px.treemap(grp, path=["category", "subcategory"], values="count",
                             color="count", color_continuous_scale="Tealgrn")
            st.plotly_chart(_style_fig(fig, 460), use_container_width=True)
        else:
            st.caption("No subcategory data available.")

        if result is not None:
            import plotly.graph_objects as go

            g = go.Figure(go.Indicator(
                mode="gauge+number", value=result.empty_pct * 100,
                title={"text": "Empty shelf space (%)"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#e5484d" if result.empty_pct >= 0.55
                               else "#f5a623" if result.empty_pct >= 0.25 else "#30a46c"},
                       "steps": [{"range": [0, 25], "color": "#14532d"},
                                 {"range": [25, 55], "color": "#78350f"},
                                 {"range": [55, 100], "color": "#7f1d1d"}]},
            ))
            st.plotly_chart(_style_fig(g, 300), use_container_width=True)

with tab_bi:
    st.subheader("Ask about the inventory")
    st.caption(
        "Natural-language questions over your saved inventory. "
        + (f"Using Ollama ({bi_engine.OLLAMA_MODEL})." if llm_on
           else "Rule-based engine (install Ollama for free-form answers).")
    )
    items_df = db.get_items_df()
    scans_df = db.get_scans_df()
    if items_df.empty and records:
        items_df = pd.DataFrame(records)

    if items_df.empty:
        st.info("No inventory yet. Analyze a shelf image first (enable 'Save scan').")
    else:
        cols = st.columns(4)
        for i, sug in enumerate(bi_engine.SUGGESTED_QUESTIONS[:8]):
            if cols[i % 4].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["bi_q"] = sug

        q = st.text_input("Your question", value=st.session_state.get("bi_q", ""),
                          placeholder="e.g. How many soft drinks are on the shelf?")
        if q:
            ans = bi_engine.answer(q, items_df, scans_df, use_llm=llm_on)
            st.markdown(f"> {ans.text}")
            st.caption(f"source: {ans.source}")
            if ans.table is not None and not ans.table.empty:
                tc1, tc2 = st.columns(2)
                tc1.dataframe(ans.table, hide_index=True, use_container_width=True)
                label_cols = [c for c in ans.table.columns if ans.table[c].dtype == object]
                num_cols = [c for c in ans.table.columns if ans.table[c].dtype != object]
                if label_cols and num_cols:
                    fig = px.bar(ans.table, x=num_cols[0], y=label_cols[0],
                                 orientation="h", color_discrete_sequence=PALETTE)
                    fig.update_layout(showlegend=False)
                    tc2.plotly_chart(_style_fig(fig, 320), use_container_width=True)

with tab_history:
    scans_df = db.get_scans_df()
    if scans_df.empty:
        st.info("No saved scans yet. Enable 'Save scan to inventory history' and analyze an image.")
    else:
        st.subheader("Scans over time")
        s = scans_df.sort_values("id")
        fig = px.line(s, x="ts", y="num_items", markers=True,
                      labels={"ts": "time", "num_items": "products detected"})
        st.plotly_chart(_style_fig(fig, 320), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Empty space per scan")
            fig = px.bar(s, x="id", y="empty_pct", color_discrete_sequence=["#f5a623"],
                         labels={"id": "scan", "empty_pct": "empty fraction"})
            st.plotly_chart(_style_fig(fig, 300), use_container_width=True)
        with c2:
            st.subheader("Aggregate category mix (all scans)")
            cc = bi_engine.category_counts(db.get_items_df())
            if not cc.empty:
                ccdf = cc.head(12).rename_axis("category").reset_index(name="count")
                fig = px.bar(ccdf, x="count", y="category", orientation="h",
                             color="category", color_discrete_sequence=PALETTE)
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(_style_fig(fig, 300), use_container_width=True)

        st.subheader("Scan log")
        st.dataframe(scans_df, hide_index=True, use_container_width=True)
