# WadiPulse-GeoAgent

**Few-Shot Geospatial Foundation Model (`Prithvi-EO-2.0-300M`) & Tool-Calling Hydrological Agent for Center-Pivot Irrigation Monitoring in Saudi Arabia**

`WadiPulse-GeoAgent` is an end-to-end geospatial machine learning and telemetry system designed to map center-pivot irrigated fields, quantify canopy moisture stress, and support natural-language hydrological queries across arid agricultural basins (**Wadi ad-Dawasir** and **Al-Kharj**, Saudi Arabia).

---

## Key Capabilities

1. **Automated Sentinel-2 L2A Ingestion (Planetary Computer STAC):**
   - Queries cloud-filtered Sentinel-2 Level-2A imagery (`B02`, `B03`, `B04`, `B8A`, `B11`, `B12` + `SCL` scene classification mask) directly via Microsoft Planetary Computer STAC APIs.
2. **Few-Shot Center-Pivot Segmentation with `Prithvi-EO-2.0-300M`:**
   - Adapts IBM/NASA's `Prithvi-EO-2.0-300M` Geospatial Foundation Model with a lightweight convolutional decoder (`prithvi_decoder_20shot.pt`) for cross-basin center-pivot delineation under scarce labeled data.
3. **Biophysical Telemetry & Spatial Storage (`DuckDB`):**
   - Vectorizes predicted field masks into GeoJSON polygons and computes per-pivot irrigated area (`ha`), mean **NDVI** (canopy vigor), mean **NDMI** (canopy water content), and composite **water-stress scores**, indexing **321 fields (10,236.9 ha)** across both basins.
4. **Grounded Tool-Calling GeoAgent (`FastAPI` + `Ollama`):**
   - Connects a local tool-calling LLM (`qwen2.5:7b-instruct` via OpenAI-compatible API) directly to deterministic DuckDB SQL tools (`query_region_water_budget`, `find_high_stress_pivots`) with an automatic deterministic fallback when offline.

---

## Cross-Basin Few-Shot Benchmark (`Prithvi-EO-2.0` vs. `U-Net`)

To evaluate cross-regional generalization across desert agricultural oases, we benchmarked `Prithvi-EO-2.0-300M` against a 6-band `U-Net` baseline trained from scratch on `5`, `10`, and `20` labeled `224×224` Sentinel-2 chips (tracked via MLflow):

| Model | Labeled Chips | Test IoU | Test F1 | Irrigated Area MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| `UNet-6Band (Baseline)` | 5 | 0.5595 | 0.7123 | 83.99% |
| **`Prithvi-EO-2.0-300M`** | **5** | **0.6255** | **0.7663** | **60.09%** |
| `UNet-6Band (Baseline)` | 10 | 0.8906 | 0.9416 | 2.41% |
| **`Prithvi-EO-2.0-300M`** | **10** | **0.7848** | **0.8781** | **8.37%** |
| `UNet-6Band (Baseline)` | 20 | 0.5580 | 0.7148 | 43.32% |
| **`Prithvi-EO-2.0-300M`** | **20** | **0.7832** | **0.8763** | **20.19%** |

- **Key Finding:** While a randomly initialized `U-Net` suffers severe overfitting and cross-basin variance (`0.5580` IoU at 20 shots), `Prithvi-EO-2.0-300M` preserves stable segmentation accuracy (`0.7832` IoU, **+0.225 IoU** over U-Net) and cuts irrigated area estimation error in half (`20.19%` vs. `43.32%` MAPE).

---

## Dashboard & Agent Preview

### Interactive Basin Telemetry & Stress Map
![Basin Telemetry Map](docs/images/map_telemetry.png)

### Few-Shot Ablation (`Prithvi-EO-2.0-300M` vs. `U-Net`)
![Few-Shot Benchmark](docs/images/ablation_benchmark.png)

### Grounded GeoAgent Querying over DuckDB
![GeoAgent Chat](docs/images/geoagent_chat.png)

---

## Repository Structure

```text
WadiPulse-GeoAgent/
├── app/
│   └── dashboard.py              # Streamlit interactive map, benchmark charts & GeoAgent UI
├── data/
│   ├── artifacts/                # Saved 20-shot Prithvi decoder weights & ablation metrics
│   └── few_shot_tiles/           # Cached Sentinel-2 6-band chips & ground-truth pivot masks
├── docs/
│   └── images/                   # Dashboard & benchmark screenshots
├── scripts/
│   └── run_few_shot_ablation.py  # Reproducible few-shot training & MLflow logging script
├── src/wadipulse/
│   ├── agent/                    # Tool-calling LLM agent + deterministic DuckDB fallback
│   ├── api/                      # FastAPI REST endpoints (/health, /fields, /summary, /ask)
│   ├── data/                     # Planetary Computer STAC loader & NDVI/NDMI computation
│   ├── mlops/                    # MLflow experiment tracking wrapper
│   ├── storage/                  # DuckDB analytical telemetry store & GeoJSON export
│   └── vision/                   # Prithvi-EO-2.0 segmentor, U-Net baseline & inference engine
├── tests/
│   └── test_pipeline.py          # Pytest unit & FastAPI integration tests
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### Option 1: Docker Compose (Recommended)

```bash
# 1. Start FastAPI backend (port 8000) and Streamlit dashboard (port 8501)
docker compose up -d --build

# 2. (Optional) Enable local LLM tool-calling via Ollama on host
ollama pull qwen2.5:7b-instruct
```

- **Interactive Dashboard:** `http://localhost:8501`
- **FastAPI Swagger Docs:** `http://localhost:8000/docs`

### Option 2: Local Development with `uv`

```bash
# 1. Install dependencies
uv sync

# 2. Run unit and API integration tests
uv run pytest -v

# 3. Launch API and dashboard
uv run uvicorn wadipulse.api.main:app --port 8000 &
uv run streamlit run app/dashboard.py
```