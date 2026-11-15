from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wadipulse.agent.geo_agent import WadiPulseGeoAgent
from wadipulse.config import STUDY_REGIONS, Settings
from wadipulse.storage.duckdb_store import WadiPulseStore
from wadipulse.vision.inference import WadiPulseInferenceEngine

st.set_page_config(
    page_title="WadiPulse-GeoAgent | Arid Irrigation Telemetry",
    page_icon="🛰️",
    layout="wide",
)

STATUS_COLORS = {
    "healthy": "#2e7d32",
    "moderate_stress": "#f9a825",
    "severe_stress": "#c62828",
}


@st.cache_resource
def get_services() -> tuple[Settings, WadiPulseStore, WadiPulseInferenceEngine, WadiPulseGeoAgent]:
    settings = Settings()
    store = WadiPulseStore(settings.db_path)
    engine = WadiPulseInferenceEngine(settings)
    agent = WadiPulseGeoAgent(store=store, settings=settings)
    if not store.get_region_summary():
        engine.bootstrap_from_cached_tiles(store=store)
    return settings, store, engine, agent


settings, store, engine, agent = get_services()

st.title("WadiPulse-GeoAgent")
st.caption(
    "Few-Shot Geospatial Foundation Model (Prithvi-EO-2.0-300M) & Tool-Calling Agent "
    "for Center-Pivot Irrigation Monitoring in Saudi Arabia"
)

with st.sidebar:
    st.header("Basin & STAC Controls")
    selected_region = st.selectbox(
        "Target Agricultural Basin",
        options=["all", *STUDY_REGIONS.keys()],
        format_func=lambda k: (
            "All Basins (Saudi Arabia)" if k == "all" else str(STUDY_REGIONS[k]["name"])
        ),
    )
    status_filter = st.selectbox(
        "Canopy Moisture Status",
        options=["all", "severe_stress", "moderate_stress", "healthy"],
    )

    st.divider()
    st.subheader("Live Sentinel-2 STAC Inference")
    live_region = st.selectbox(
        "Basin to Query",
        options=list(STUDY_REGIONS.keys()),
        format_func=lambda k: str(STUDY_REGIONS[k]["name"]),
    )
    date_range = st.text_input("STAC Date Window", value="2026-03-01/2026-06-30")
    if st.button("Run Live STAC + Prithvi Inference", width='stretch'):
        with st.spinner("Querying Planetary Computer STAC & running Prithvi-EO-2.0..."):
            try:
                res = engine.analyze_region_live(
                    region=live_region,
                    date_range=date_range,
                    store=store,
                )
                st.success(
                    f"Segmented {res['field_count']} pivots from {res['scene_id']} "
                    f"({res['acquisition_date']})"
                )
            except Exception as exc:
                st.error(f"STAC inference failed: {exc}")

    if st.button("Re-index Cached Saudi Tiles", width='stretch'):
        with st.spinner("Running Prithvi-EO-2.0 over 27 cached chips..."):
            inserted = engine.bootstrap_from_cached_tiles(store=store)
            st.success(f"Indexed {inserted} pivot fields into DuckDB.")

tab_map, tab_bench, tab_agent = st.tabs(
    ["Basin Telemetry & Map", "Few-Shot GFM Benchmark", "GeoAgent Assistant"]
)


with tab_map:
    region_arg = None if selected_region == "all" else selected_region
    status_arg = None if status_filter == "all" else status_filter

    summaries = store.get_region_summary(region=region_arg)
    fields = store.get_fields(region=region_arg, status=status_arg, limit=500)

    if summaries:
        sum_df = pd.DataFrame(summaries)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detected Pivot Fields", int(sum_df["total_pivots"].sum()))
        c2.metric("Irrigated Area (ha)", f"{sum_df['total_irrigated_ha'].sum():,.1f}")
        c3.metric("Mean Canopy NDVI", f"{sum_df['avg_ndvi'].mean():.3f}")
        c4.metric("Mean Moisture NDMI", f"{sum_df['avg_ndmi'].mean():.3f}")

    if fields:
        fields_df = pd.DataFrame(fields)
        center_lat = float(fields_df["centroid_lat"].mean())
        center_lon = float(fields_df["centroid_lon"].mean())
        zoom = 6 if selected_region == "all" else 11

        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron")
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite Imagery",
        ).add_to(m)

        for row in fields:
            color = STATUS_COLORS.get(row["stress_status"], "#1565c0")
            radius_m = max(120.0, (float(row["area_ha"]) * 10_000.0 / 3.14159) ** 0.5)
            popup_html = (
                f"<b>{row['field_id']}</b><br>"
                f"Area: {row['area_ha']} ha<br>"
                f"NDVI: {row['mean_ndvi']} | NDMI: {row['mean_ndmi']}<br>"
                f"Stress Score: {row['stress_score']} ({row['stress_status']})"
            )
            folium.Circle(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=radius_m,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=470, returned_objects=[])

        st.subheader("Per-Pivot Biophysical Telemetry (DuckDB)")
        st.dataframe(fields_df, width='stretch', hide_index=True)
    else:
        st.info(
            "No pivot records match the current filter. Click 'Re-index Cached Saudi Tiles' in the sidebar."
        )

with tab_bench:
    ablation_path = settings.artifacts_dir / "ablation_results.json"
    if ablation_path.exists():
        ablation_rows = json.loads(ablation_path.read_text(encoding="utf-8"))
        ab_df = pd.DataFrame(ablation_rows)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_iou = px.line(
                ab_df,
                x="shots",
                y="test_iou",
                color="model",
                markers=True,
                title="Cross-Basin Segmentation IoU vs. Labeled Training Chips",
                labels={"shots": "Labeled Training Chips (224x224)", "test_iou": "Test IoU"},
            )
            fig_iou.update_layout(yaxis_range=[0.4, 1.0])
            st.plotly_chart(fig_iou, width='stretch')

        with col_b:
            fig_mape = px.bar(
                ab_df,
                x="shots",
                y="area_mape_pct",
                color="model",
                barmode="group",
                title="Irrigated Area Estimation Error (MAPE %, Lower is Better)",
                labels={
                    "shots": "Labeled Training Chips (224x224)",
                    "area_mape_pct": "Area MAPE (%)",
                },
            )
            st.plotly_chart(fig_mape, width='stretch')

        st.subheader("Ablation Summary Table (MLflow Logged)")
        st.dataframe(ab_df, width='stretch', hide_index=True)
    else:
        st.warning(
            "Run `uv run python scripts/run_few_shot_ablation.py` to generate benchmark metrics."
        )

with tab_agent:
    st.markdown(
        "Ask natural-language hydrological questions grounded in the DuckDB pivot telemetry "
        "(e.g., *'Compare irrigated area and moisture stress between Wadi ad-Dawasir and Al-Kharj'*)."
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask WadiPulse GeoAgent...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Querying DuckDB telemetry tools..."):
                reply = agent.ask(prompt)
                footer = f"\n\n*Tools executed: `{', '.join(reply['tool_calls'])}` | Mode: `{reply['mode']}`*"
                full_text = reply["answer"] + footer
                st.markdown(full_text)
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
