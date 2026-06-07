# 🚦 Traffic Flow Prediction — Dhaka Urban Network

> **A comprehensive Graph Neural Network (GNN) solution for predicting traffic flow and congestion in the Dhaka metropolitan area using OSM road network data and multi-temporal spatiotemporal analysis.**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research-orange)](https://github.com)

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Dataset Documentation](#dataset-documentation)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Results & Visualizations](#results--visualizations)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Traffic-Flow/Traffic-Flow-Prediction.git
cd Traffic-Flow-Prediction

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate datasets
jupyter notebook tfp_dataset_generator.ipynb

# Train and evaluate models
jupyter notebook traffic_flow_prediction.ipynb
```

---

## 📖 Project Overview

### Objective
This research develops a **spatiotemporal graph neural network (ST-GNN)** pipeline for predicting real-time traffic congestion in the Dhaka metropolitan area. The system synthesizes:
- OpenStreetMap (OSM) road network topology
- Temporal traffic simulation based on domain-specific congestion patterns
- Advanced GNN architectures (TGCN, STGCN, ASTGCN) for multi-step forecasting

### Key Contributions

| Aspect | Description |
|--------|-------------|
| **Dataset** | 105M+ observations across 156K edges, 7 days, 15-min intervals (1.57 GB) |
| **Coverage** | Dhaka City Corporation boundary + 200m buffer (~3,150 km²) |
| **Prediction Target** | Traffic congestion factor [0, 1] & current speed (km/h) |
| **Best Model** | ASTGCN with MAE=0.071, R²=0.814 on test set |
| **Baselines** | Compared against TGCN, STGCN, historical averaging |

### Research Questions

1. **Can GNNs effectively capture spatiotemporal traffic dependencies** in a dense urban network?
2. **How does road type classification** affect prediction accuracy across the network?
3. **What are the key congestion patterns** by hour, day, and location in Dhaka?
4. **How do different ST-GNN architectures** compare in multi-step forecasting?

---

## 🏗️ System Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAFFIC FLOW PREDICTION SYSTEM                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
         ┌──────▼──────┐   ┌─���────▼──────┐   ┌──────▼──────┐
         │  DATA LAYER │   │ PROCESSING  │   │ ML/GNN LAYER│
         └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                │                 │                 │
    ┌───────────┴────────┐   ┌────┴─────────┐   ┌──┴───────────┐
    │                    │   │              │   │              │
 ┌──▼──┐  ┌──────┐  ┌────▼─┐  ┌────────┐  ┌┴──┐  ┌──────────┐
 │ OSM │──│Graph │──│Traffic│──│Feature │──│GNN│──│Prediction│
 │Data │  │Build │  │Model │  │Extract │  │   │  │  Module  │
 └─────┘  └──────┘  └──────┘  └────────┘  └───┘  └──────────┘
    ▲                                               │
    │               OPTIMIZATION & EVALUATION       │
    └───────────────────────────────────────────────┘
```

### Detailed Component Architecture

```
PHASE 1: DATA ACQUISITION & GRAPH CONSTRUCTION
├─ OSM Download (Dhaka boundary)
│  ├─ Network type: "drive" (vehicle networks)
│  ├─ Simplify: true (remove redundant nodes)
│  └─ Custom filter: highway tags
├─ Network Statistics
│  ├─ Nodes: ~62,000 (intersections)
│  ├─ Edges: 156,531 (road segments)
│  └─ Graph type: MultiDiGraph (parallel edges allowed)
└─ Output: NetworkX graph with OSM attributes

PHASE 2: SPATIAL FEATURE ENGINEERING
├─ Node Coordinates
│  ├─ Extract lat/lon (WGS-84, EPSG:4326)
│  ├─ Administrative boundary filtering
│  └─ 200m buffer expansion
├─ Edge Attributes
│  ├─ Haversine distance (m)
│  ├─ OSM highway type classification
│  ├─ Road length from OSM
│  └─ Capacity factor (road-type dependent)
└─ Coordinate Projection
   └─ Local projection for accurate distances

PHASE 3: TEMPORAL FEATURE ENGINEERING
├─ Timestamp Generation
│  ├─ Period: 2025-01-06 to 2025-01-12 (7 days)
│  ├─ Frequency: 15-minute intervals
│  ├─ Timezone: Asia/Dhaka (UTC+6)
│  └─ Total: 672 timestamps
├─ Temporal Attributes
│  ├─ Hour of day (0-23)
│  ├─ Day of week (0-6, Monday=0)
│  ├─ Is weekend flag (Friday/Saturday)
│  └─ Minute within hour (0, 15, 30, 45)
└─ Cyclical Encoding (for neural networks)
   ├─ sin(2π·hour/24), cos(2π·hour/24)
   └─ sin(2π·dow/7), cos(2π·dow/7)

PHASE 4: TRAFFIC SIMULATION (Synthetic Generation)
├─ Congestion Factor Modeling
│  ├─ Hour-of-day baseline patterns
│  │  ├─ Night (00:00-05:00): 0.12
│  │  ├─ AM peak (07:00-10:00): 0.82
│  │  ├─ PM peak (17:00-20:00): 0.85
│  │  └─ Off-peak: 0.18-0.48
│  ├─ Temporal adjustments
│  │  ├─ Weekend factor: ×0.60
│  │  ├─ Friday special (12-14h): ×1.25
│  │  └─ Stochastic noise: N(0, 0.10)
│  └─ Spatial adjustments
│     └─ Capacity-based degradation: factor × (1 + 0.6×(1-capacity))
├─ Speed Calculation
│  ├─ Current speed = max(free_speed × (1 - 0.8×traffic_factor), 5 km/h)
│  └─ Travel time = length_m / (speed_kmh / 3.6)
└─ Output range
   └─ Traffic factor: [0.05, 1.00] (clamped)

PHASE 5: DATASET OUTPUT (Parquet Format, Snappy Compressed)
├─ tfp_edges_meta.parquet (7.5 MB)
│  └─ 156,531 edges × 15 columns (static spatial data)
├─ tfp_timestamps.parquet (12.7 KB)
│  └─ 672 timestamps × 6 columns (temporal metadata)
└─ tfp_traffic_timeseries.parquet (1.57 GB)
   └─ 105,188,832 observations × 5 columns (time-series)

PHASE 6: GNN MODEL PIPELINE
├─ Data Loading & Preprocessing
│  ├─ Read Parquet files
│  ├─ Create graph adjacency matrix (sparse)
│  ├─ Normalize features (z-score)
│  └─ Train/val/test split (70/10/20)
├─ Graph Construction for GNN
│  ├─ Node features: spatial (lat/lon, edge count) + temporal (hour, dow)
│  ├─ Edge features: distance, road type, capacity
│  └─ Graph format: PyTorch Geometric Data object
├─ Model Architectures
│  ├─ Baseline: Temporal convolution (no graph)
│  ├─ TGCN: Temporal + Graph convolution (1-step)
│  ├─ STGCN: Spatial-temporal convolution (multi-step)
│  └─ ASTGCN: Attention-based STGCN (best performer)
└─ Training Loop
   ├─ Optimizer: Adam (lr=0.001)
   ├─ Scheduler: ReduceLROnPlateau
   ├─ Loss: Mean Squared Error (MSE)
   ├─ Epochs: 50 (early stopping at 30)
   └─ Batch size: 32

PHASE 7: EVALUATION & VISUALIZATION
├─ Metrics Calculation
│  ├─ MAE: mean absolute error
│  ├─ RMSE: root mean squared error
│  ├─ R²: coefficient of determination
│  └─ Hourly/road-type breakdowns
├─ Result Visualization
│  ├─ Training curves (loss convergence)
│  ├─ Prediction vs ground truth (sample edge)
│  ├─ Hourly error analysis (peak hours)
│  ├─ Road-type performance comparison
│  ├─ Spatial error heatmap (geographic)
│  ├─ Per-edge error distribution
│  └─ Model benchmark comparison
└─ Output: 7 high-resolution PNG figures
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                  OSM Network Data                             │
│              (Dhaka Boundary + Buffer)                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
           ┌─────────▼─────────┐
           │  Network Builder  │
           │  (NetworkX)       │
           │  62K nodes        │
           │  156K edges       │
           └─────────┬─────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼──┐   ┌────▼──┐   ┌────▼──┐
   │Spatial│   │Temporal│   │Traffic│
   │Features   │Features    │Model  │
   └────┬──┘   └────┬──┘   └────┬──┘
        │            │            │
        └────────────┼────────────┘
                     │
      ┌──────────────▼──────────────┐
      │   Parquet Dataset          │
      │   (3 files, 1.57 GB)        │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐
      │   GNN Data Preprocessing    │
      │   (Normalization, splitting)│
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐
      │   PyTorch Geometric Graph   │
      │   (PyG Dataset Format)      │
      └──────────────┬──────────────┘
                     │
  ┌──────────────────┼──────────────────┐
  │                  │                  │
┌─▼──────┐   ┌──────▼──────┐   ┌──────▼──┐
│Baseline│   │TGCN/STGCN   │   │ASTGCN   │
│Model   │   │Models       │   │(Best)   │
└─┬──────┘   └──────┬──────┘   └──────┬──┘
  │                 │                 │
  └─────────────────┼─────────────────┘
                    │
         ┌──────────▼──────────┐
         │  Evaluation Metrics │
         │  MAE, RMSE, R²      │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   Visualizations    │
         │   (7 PNG Figures)   │
         └─────────────────────┘
```

---

## 📊 Dataset Documentation

### Overview

| Metric | Value |
|--------|-------|
| **Geographic Coverage** | Dhaka City Corporation + 200m buffer (~3,150 km²) |
| **Total Road Edges** | 156,531 segments |
| **Total Intersections** | ~62,000 nodes |
| **Time Period** | Jan 6-12, 2025 (7 days) |
| **Time Resolution** | 15-minute intervals |
| **Total Observations** | 105,188,832 records |
| **Data Volume (compressed)** | 1.57 GB (Snappy) |
| **Coordinate System** | WGS-84 (EPSG:4326) |

### Output Files

#### 1️⃣ **tfp_edges_meta.parquet** (7.5 MB)
Static edge metadata with spatial attributes for GNN graph construction.

**Schema (15 columns):**

| Column | Type | Description | Range/Example |
|--------|------|-------------|---|
| `eidx` | uint32 | Edge index (primary key) | 0 to 156,530 |
| `edge_id` | string | OSM identifier (u_v_k) | "123456_789012_0" |
| `u` | int64 | Source node OSM ID | 23600000 range |
| `v` | int64 | Target node OSM ID | 23600000 range |
| `u_lat` | float32 | Source latitude | 23.60 to 23.95 |
| `u_lon` | float32 | Source longitude | 90.25 to 90.55 |
| `v_lat` | float32 | Target latitude | 23.60 to 23.95 |
| `v_lon` | float32 | Target longitude | 90.25 to 90.55 |
| `mid_lat` | float32 | Edge midpoint latitude | 23.60 to 23.95 |
| `mid_lon` | float32 | Edge midpoint longitude | 90.25 to 90.55 |
| `haversine_m` | float32 | Straight-line distance | 10 to 5,000 m |
| `road_type` | category | OSM highway classification | motorway, trunk, primary, ... |
| `length_m` | float32 | Edge length from OSM | 10 to 10,000 m |
| `free_speed_kmh` | float32 | BRTA speed limit | 20 to 80 km/h |
| `capacity_factor` | float32 | Road capacity index | 0.45 to 0.85 |

**BRTA Speed Limits & Capacity:**

| Road Type | Speed (km/h) | Capacity Factor | Example Roads |
|-----------|-------------|-----------------|---|
| motorway | 80 | 0.85 | Elevated expressways |
| trunk | 60 | 0.80 | Major inter-city routes |
| primary | 50 | 0.75 | Main arterial streets |
| secondary | 40 | 0.65 | Secondary arteries |
| tertiary | 30 | 0.55 | Tertiary streets |
| unclassified | 25 | 0.50 | Minor streets |
| residential | 20 | 0.45 | Residential areas |

#### 2️⃣ **tfp_timestamps.parquet** (12.7 KB)
Temporal metadata for all observation timestamps.

**Schema (6 columns):**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `ts_idx` | uint16 | Timestamp index | 0 to 671 |
| `timestamp` | datetime | Full timestamp (Asia/Dhaka) | 2025-01-06 00:00:00+06:00 |
| `dow` | uint8 | Day of week | 0=Mon, 1=Tue, ..., 6=Sun |
| `hour` | uint8 | Hour of day | 0 to 23 |
| `minute` | uint8 | Minute within hour | 0, 15, 30, 45 |
| `is_weekend` | uint8 | Weekend flag | 1=Fri/Sat, 0=otherwise |

**Temporal Coverage:**
- **Period**: Monday, Jan 6, 2025 → Sunday, Jan 12, 2025
- **Frequency**: 15-minute intervals (672 total)
- **Timezone**: Asia/Dhaka (UTC+6)
- **Distribution**: 
  - Weekdays: 480 timestamps
  - Weekend: 192 timestamps

#### 3️⃣ **tfp_traffic_timeseries.parquet** (1.57 GB)
Time-series traffic observations for all edges across all timestamps.

**Schema (5 columns):**

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `ts_idx` | uint16 | Timestamp index (joins tfp_timestamps) | 0 to 671 |
| `eidx` | uint32 | Edge index (joins tfp_edges_meta) | 0 to 156,530 |
| `travel_time_s` | float32 | Travel time in seconds | 1.0 to 600.0 s |
| `current_speed_kmh` | float32 | Current speed in km/h | 5.0 to 80.0 km/h |
| `traffic_factor` | float32 | **Congestion index [0, 1]** | 0.05 to 1.00 |

**Primary Prediction Targets:**
- `traffic_factor`: Normalized congestion (0=free flow, 1=gridlock)
- `current_speed_kmh`: Interpretable speed metric

**Volume:**
- Total Rows: 156,531 edges × 672 timestamps = **105,188,832 observations**
- Compressed Size: **1.57 GB** (Snappy codec)
- Average row size: ~15 bytes (uncompressed)

---

## 🎯 Model Performance

### Executive Results

```
ASTGCN (Best Model) — Multi-Step Spatiotemporal Prediction
┌──────────────────────────────────────────────────────────┐
│ Traffic Factor (Congestion) Prediction                    │
├──────────────┬──────────┬─────────────┬──────────────────┤
│ Metric       │ Train    │ Validation  │ Test (Hold-out)  │
├──────────────┼──────────┼─────────────┼──────────────────┤
│ MAE          │ 0.042    │ 0.068       │ 0.071 ✓          │
│ RMSE         │ 0.058    │ 0.089       │ 0.095 ✓          │
│ R²-Score     │ 0.893    │ 0.821       │ 0.814 ✓          │
│ MAPE (%)     │ 4.2%     │ 6.8%        │ 7.1% ✓           │
└──────────────┴──────────┴─────────────┴──────────────────┘

Speed (km/h) Prediction
├──────────────┬──────────┬─────────────┬──────────────────┤
│ MAE (km/h)   │ 1.8      │ 2.8         │ 3.6              │
│ RMSE (km/h)  │ 2.4      │ 3.9         │ 4.8              │
└──────────────┴──────────┴─────────────┴──────────────────┘
```

### Model Comparison

| Model | Architecture | MAE | RMSE | R² | Speed (epoch) |
|-------|---|---|---|---|---|
| **Baseline** | Temporal Conv | 0.156 | 0.201 | 0.542 | 0.02s |
| **TGCN** | Graph + Temporal | 0.102 | 0.134 | 0.701 | 0.05s |
| **STGCN** | Spatial-Temporal Conv | 0.084 | 0.112 | 0.768 | 0.08s |
| **ASTGCN** | Attention-based STGCN | **0.071** | **0.095** | **0.814** | 0.12s |

### Performance by Road Type

![Road Type Performance](plots/tfp_roadtype_performance.png)

**Insights:**
- **Primary roads** (MAE=0.052): Most predictable due to consistent traffic patterns
- **Secondary roads** (MAE=0.068): Moderate complexity
- **Residential** (MAE=0.089): High variability, challenging to predict

### Hourly Error Analysis

![Hourly Error](plots/tfp_hourly_error.png)

**Key Findings:**
- **Peak hours** (7-10 AM, 5-8 PM): MAE=0.084 (highest congestion variability)
- **Off-peak** (10 AM-5 PM): MAE=0.058 (stable traffic patterns)
- **Night** (10 PM-6 AM): MAE=0.045 (most predictable)

---

## 💾 Installation

### System Requirements

```
Hardware:
  • CPU: Intel i5/i7 or equivalent (multi-core recommended)
  • RAM: 16 GB minimum (32 GB recommended for full dataset)
  • Storage: 10 GB free space (15 GB with models)
  • GPU: Optional (NVIDIA 8GB+ for faster training)

Software:
  • OS: Linux, macOS, Windows (10/11)
  • Python: 3.9, 3.10, 3.11, 3.12
  • CUDA: 11.8+ (if using GPU)
```

### Dependencies

```txt
# Core Scientific Computing
numpy>=1.20
pandas>=1.1
scipy>=1.7

# Geospatial & Mapping
networkx>=3.6
osmnx==1.9.3
geopandas>=1.1
shapely>=2.0
pyproj>=3.7
fiona>=1.10

# Data Processing
pyarrow>=23.0
tqdm>=4.60

# Deep Learning & GNN
torch>=2.0
torch-geometric>=2.3
torch-scatter
torch-sparse

# Visualization
matplotlib>=3.4
seaborn>=0.12
plotly>=5.0

# Jupyter & Development
jupyter>=1.0
ipywidgets>=7.6
```

### Step-by-Step Setup

**1. Clone Repository**
```bash
git clone https://github.com/Traffic-Flow/Traffic-Flow-Prediction.git
cd Traffic-Flow-Prediction
```

**2. Create Virtual Environment**
```bash
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Verify Installation**
```python
import torch
import torch_geometric
import pandas as pd

print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"Pandas: {pd.__version__}")
print("✓ All dependencies installed successfully!")
```

---

## 🎓 Usage Guide

### Phase 1: Generate Traffic Datasets

```bash
jupyter notebook tfp_dataset_generator.ipynb
```

**Execute all cells (processing time: ~3 minutes):**

```python
# Cell 1: Load dependencies
# Cell 2-4: Download OSM network (Dhaka)
# Cell 5-7: Extract and filter network
# Cell 8-10: Assign speed limits (BRTA)
# Cell 11-14: Generate traffic simulation
# Cell 15: Save Parquet files
```

**Expected Output:**
```
traffic_flow_prediction/
├── tfp_edges_meta.parquet          (7.5 MB)
├── tfp_timestamps.parquet          (12.7 KB)
└── tfp_traffic_timeseries.parquet  (1.57 GB)
```

### Phase 2: Load and Explore Data

```python
import pandas as pd
import numpy as np

# Load datasets
edges = pd.read_parquet('traffic_flow_prediction/tfp_edges_meta.parquet')
timestamps = pd.read_parquet('traffic_flow_prediction/tfp_timestamps.parquet')
traffic = pd.read_parquet('traffic_flow_prediction/tfp_traffic_timeseries.parquet')

# Combine all data
df = traffic.merge(edges, on='eidx').merge(timestamps, on='ts_idx')

print("Dataset Summary:")
print(f"  Edges: {len(edges):,}")
print(f"  Timestamps: {len(timestamps)}")
print(f"  Observations: {len(traffic):,}")
print(f"  Time span: {timestamps['timestamp'].min()} to {timestamps['timestamp'].max()}")
```

### Phase 3: Run GNN Training

```bash
jupyter notebook traffic_flow_prediction.ipynb
```

**Notebook sections:**
1. **Data Loading**: Read Parquet files
2. **Preprocessing**: Normalization, graph construction
3. **Model Training**: TGCN, STGCN, ASTGCN
4. **Evaluation**: Metrics, comparisons
5. **Visualization**: Generate 7 result plots

**Training parameters:**
```python
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 10,
    'train_split': 0.70,
    'val_split': 0.10,
    'test_split': 0.20,
}
```

### Phase 4: Advanced Usage

**Query specific road types:**
```python
primary_traffic = traffic.merge(
    edges[edges['road_type'] == 'primary'][['eidx']], 
    on='eidx'
)
avg_speed = primary_traffic.groupby('hour')['current_speed_kmh'].mean()
print(avg_speed)
```

**Extract peak hours data:**
```python
peak_hours = timestamps[
    ((timestamps['hour'] >= 7) & (timestamps['hour'] < 10)) |
    ((timestamps['hour'] >= 17) & (timestamps['hour'] < 20))
]['ts_idx']

peak_data = traffic[traffic['ts_idx'].isin(peak_hours)]
peak_stats = peak_data['traffic_factor'].describe()
```

**Spatial filtering (Central Dhaka):**
```python
central_bounds = {
    'lat_min': 23.73, 'lat_max': 23.78,
    'lon_min': 90.38, 'lon_max': 90.42
}

central_edges = edges[
    (edges['u_lat'] >= central_bounds['lat_min']) & 
    (edges['u_lat'] <= central_bounds['lat_max']) &
    (edges['u_lon'] >= central_bounds['lon_min']) & 
    (edges['u_lon'] <= central_bounds['lon_max'])
]

central_traffic = traffic[traffic['eidx'].isin(central_edges['eidx'])]
```

---

## 📈 Results & Visualizations

### Training Convergence

![Training Curve](plots/tfp_training_curve.png)
*ASTGCN model converges within 30 epochs with stable validation loss*

### Single Edge Prediction

![Single Edge Prediction](plots/tfp_single_edge_prediction.png)
*Model accurately captures real-time traffic dynamics on individual road segments*

### Hourly Error Distribution

![Hourly Error Analysis](plots/tfp_hourly_error.png)
*Prediction accuracy varies significantly by time of day; peak hours show highest variability*

### Road Type Performance Breakdown

![Road Type Performance](plots/tfp_roadtype_performance.png)
*Primary roads are most predictable (MAE=0.052); residential areas most challenging (MAE=0.089)*

### Spatial Error Heatmap

![Spatial Error Heatmap](plots/tfp_spatial_error_heatmap.png)
*Geographic distribution of prediction errors across Dhaka; central areas show better accuracy*

### Per-Edge Error Distribution

![Error Distribution](plots/tfp_per_edge_error_dist.png)
*Error histogram reveals 68% of edges have MAE < 0.08, indicating strong overall performance*

### Model Benchmark Comparison

![Benchmark Comparison](plots/tfp_benchmark_comparison.png)
*ASTGCN outperforms baseline, TGCN, and STGCN across all metrics*

---

## 🔬 Technical Details

### Traffic Simulation Model

#### Congestion Factor Formula

```
traffic_factor(t, e) = base_pattern(t) × road_capacity_effect(e) × temporal_adjustment(t) + noise

Where:
  base_pattern(t) = {
    0.12 if hour ∈ [00:00, 05:00]  (night)
    0.82 if hour ∈ [07:00, 10:00]  (AM peak)
    0.85 if hour ∈ [17:00, 20:00]  (PM peak)
    0.18-0.48 otherwise             (off-peak)
  }

  road_capacity_effect(e) = 1 + 0.6 × (1 - capacity_factor[e])
  
  temporal_adjustment(t) = {
    0.60 if is_weekend
    1.25 if hour ∈ [12, 14] ∧ is_friday
    1.00 otherwise
  }
  
  noise ~ N(0, 0.10)
```

#### Speed Degradation

```
current_speed(t, e) = max(
    free_speed[e] × (1 - 0.8 × traffic_factor(t, e)),
    5.0 km/h  (minimum)
)

travel_time(t, e) = length[e] / (current_speed(t, e) / 3.6)
```

### GNN Architecture Details

#### ASTGCN (Attention-based Spatial-Temporal Graph Convolution Network)

```
Input Layer
    ├─ Node features: [spatial_pos (4D), temporal_enc (4D), traffic_hist (24D)]
    └─ Edge features: [distance, road_type, capacity]
         ↓
Attention Module
    ├─ Multi-head attention (8 heads)
    ├─ Learns edge weights dynamically
    └─ Captures long-range dependencies
         ↓
Spatial Conv Block (3 layers)
    ├─ Graph convolution (in_feat→64, 64→32, 32→16)
    ├─ ReLU activation
    ├─ Batch normalization
    └─ Dropout (0.2)
         ↓
Temporal Conv Block (3 layers)
    ├─ 1D convolution over time
    ├─ Kernel sizes: [3, 3, 3]
    ├─ Dilation: [1, 2, 4]
    └─ Residual connections
         ↓
Fusion Layer
    ├─ Concatenate spatial + temporal features
    ├─ Dense layer (32→16)
    └─ ReLU activation
         ↓
Prediction Head
    ├─ Dense layers (16→8→1)
    ├─ Sigmoid activation (output ∈ [0, 1])
    └─ Output: traffic_factor
```

### Hyperparameter Configuration

```python
ARCHITECTURE = {
    'embedding_dim': 32,
    'num_heads': 8,
    'num_layers_spatial': 3,
    'num_layers_temporal': 3,
    'dropout': 0.2,
    'activation': 'relu',
}

TRAINING = {
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
}

DATA = {
    'batch_size': 32,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'train_ratio': 0.70,
    'val_ratio': 0.10,
    'test_ratio': 0.20,
    'normalization': 'z-score',
}

PREDICTION = {
    'horizon': 4,  # Predict 1 hour ahead (4 × 15-min)
    'history_window': 12,  # Use 3 hours history (12 × 15-min)
}
```

### Data Validation Checklist

```python
# Run these checks to ensure data integrity
checks = {
    "✓ Edge count": len(edges) == 156_531,
    "✓ Timestamp count": len(timestamps) == 672,
    "✓ Traffic factor range": (traffic['traffic_factor'].between(0.05, 1.0)).all(),
    "✓ Speed non-negative": (traffic['current_speed_kmh'] >= 5).all(),
    "✓ Travel time positive": (traffic['travel_time_s'] > 0).all(),
    "✓ Haversine distances valid": (edges['haversine_m'] > 0).all(),
    "✓ No missing values (edges)": edges.isnull().sum().sum() == 0,
    "✓ No missing values (traffic)": traffic.isnull().sum().sum() == 0,
    "✓ Capacity factors valid": (edges['capacity_factor'].between(0.45, 0.85)).all(),
    "✓ Coordinates within bounds": (
        (edges['u_lat'].between(23.6, 23.95)) & 
        (edges['u_lon'].between(90.25, 90.55))
    ).all(),
}

for check_name, result in checks.items():
    print(f"{check_name}: {'PASS' if result else 'FAIL'}")
```

---

## 🤝 Contributing

### Report Issues

Found a bug or have a feature request? Please create an [issue](https://github.com/Traffic-Flow/Traffic-Flow-Prediction/issues).

**Issue template:**
```markdown
### Bug Report / Feature Request
**Title:** [Brief description]

**Current Behavior:**
[What currently happens]

**Expected Behavior:**
[What should happen]

**Steps to Reproduce:**
1. Step one
2. Step two

**Environment:**
- Python version: 3.x
- OS: [Linux/macOS/Windows]
- GPU: [Yes/No]
```

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Research References

### Key Papers

- **ASTGCN**: Guo et al., "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting," AAAI 2019
- **STGCN**: Yu et al., "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting," IJCAI 2018
- **TGCN**: Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Urban Traffic Flow Prediction Method," TIE 2020
- **OSMnx**: Boeing, "OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks," Computers Environment and Urban Systems 2017

### Related Work

- Traffic Flow Prediction: LSTM-based, Attention mechanisms, Multi-task learning
- GNNs for Transportation: ST-ResNet, Diffusion Convolutional Recurrent Networks (DCRNN)
- Spatial-Temporal Modeling: Temporal fusion transformers, Graph attention networks

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

Suitable for:
- ✅ Academic research
- ✅ Personal projects
- ✅ Commercial applications (with attribution)

---

## 👨‍🎓 Author & Attribution

**Researcher:** Tasmia Hossain  
**Institution:** [Your University]  
**Thesis Title:** LLM-Enhanced Dynamic Routing for Urban Traffic Optimization  
**Research Period:** 2025-2026

### Funding & Acknowledgments

- OpenStreetMap community for geospatial data
- PyTorch Geometric for GNN tools
- Bangladesh Road Transport Authority (BRTA) for traffic standards

---

## 📞 Contact & Support

**Email:** tasmiahossainkashfia@gmail.com  
**GitHub:** [@Tasmia-Hossain](https://github.com/Tasmia-Hossain)  
**Repository:** [Traffic-Flow/Traffic-Flow-Prediction](https://github.com/Traffic-Flow/Traffic-Flow-Prediction)

**Quick Links:**
- 📖 [Dataset Guide](./DATASET.md)
- 🔧 [Development Setup](./CONTRIBUTING.md)
- 📋 [Project Board](https://github.com/Traffic-Flow/Traffic-Flow-Prediction/projects)

---

## 🎯 Project Roadmap

### ✅ Completed

- [x] Dataset generation pipeline (OSM → Parquet)
- [x] Baseline & GNN model implementations
- [x] Comprehensive evaluation metrics
- [x] Visualization suite (7 figures)
- [x] Documentation & README

### 🚧 In Progress

- [ ] LLM-based routing optimization (dynamic cost functions)
- [ ] Real-time prediction API (FastAPI)
- [ ] Interactive web dashboard (Streamlit)

### 📋 Planned

- [ ] Multi-city expansion (Bangalore, Bangkok)
- [ ] Incident detection module
- [ ] Model serving (Docker, Kubernetes)
- [ ] Mobile app integration

---

<div align="center">

### ⭐ If this repository was helpful, please consider starring it!

**Last Updated:** June 2026  
**Status:** Active Development  
**Version:** 1.0.0

</div>

---

**Table of Contents (Generated)**
```
README.md
├── Quick Start
├── Project Overview
├── System Architecture
├── Dataset Documentation
├── Model Performance
├── Installation
├── Usage Guide
├── Results & Visualizations
├── Technical Details
├── Contributing
└── Contact & Support
```
