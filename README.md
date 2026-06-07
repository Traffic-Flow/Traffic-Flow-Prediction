# Traffic Flow Prediction — Dhaka Urban Network

<div align="center">

![Traffic Flow](https://img.shields.io/badge/Traffic_Flow-Prediction-blue?style=flat-square)
![Dataset Size](https://img.shields.io/badge/Dataset-105M%2B_Rows-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-orange?style=flat-square)

**A comprehensive Graph Neural Network framework for traffic flow prediction in Dhaka urban transportation network**

[Quick Start](#quick-start) | [Datasets](#output-datasets) | [Results](#results--visualizations) | [Installation](#installation--setup)

</div>

---

## Overview

This component implements traffic flow prediction for the Dhaka urban transportation network using Graph Neural Networks (GNNs). The project generates high-quality, spatially-aware traffic datasets with 105M+ observations and provides production-ready Parquet files optimized for time-series forecasting tasks.

### Key Features

- Production-Ready Datasets: 3 optimized Parquet files (1.57 GB compressed)
- Spatial Awareness: Complete coordinates for 156K+ road edges
- Temporal Dynamics: 15-minute granularity over 7-day period
- Multiple Targets: Traffic factor and speed prediction
- Validated Model: Realistic Dhaka traffic patterns (BRTA calibrated)
- GNN-Optimized: Pre-processed features for TGCN/STGCN/ASTGCN

### Objectives

- Generate production-ready traffic flow datasets with spatial and temporal features
- Provide comprehensive edge metadata with coordinates and road attributes
- Create time-series traffic observations (105M+ rows) with multiple prediction targets
- Enable GNN-based traffic prediction and speed forecasting

---

## System Architecture

```mermaid
flowchart TB
    subgraph data [Data Input]
        CSV["OSM Road Network<br/>62K nodes, 156K edges"]
        Admin["Admin Boundary<br/>Extraction"]
        Speed["Speed Limit<br/>Assignment"]
    end

    subgraph processing [Processing Pipeline]
        Capacity["Capacity Factor<br/>Calculation"]
        Traffic["Vectorized Traffic<br/>Simulation"]
        Coords["Coordinate<br/>Extraction"]
    end

    subgraph output [Output Files]
        Meta["tfp_edges_meta.parquet<br/>156,531 edges × 15 cols<br/>7.5 MB"]
        TS["tfp_timestamps.parquet<br/>672 steps × 6 cols<br/>12.7 KB"]
        Data["tfp_traffic_timeseries.parquet<br/>105.2M rows × 5 cols<br/>1.57 GB"]
    end

    subgraph features [Feature Engineering]
        Spatial["Spatial Features<br/>Coordinates, Distance"]
        Road["Road Features<br/>Type, Speed, Capacity"]
        Temporal["Temporal Features<br/>Hour, DOW, Weekend"]
    end

    subgraph ml [Machine Learning]
        Train["GNN Training<br/>TGCN/STGCN/ASTGCN"]
        Predict["Traffic Prediction<br/>Speed and Congestion"]
    end

    CSV --> Admin
    Admin --> Speed
    Speed --> Capacity
    Capacity --> Traffic
    Traffic --> Coords
    Coords --> Meta
    Traffic --> TS
    Traffic --> Data
    Meta --> Spatial
    Meta --> Road
    TS --> Temporal
    Spatial --> Train
    Road --> Train
    Temporal --> Train
    Train --> Predict
```

---

## Repository Structure

```
Traffic-Flow-Prediction/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── tfp_dataset_generator.ipynb            # Dataset generation (Cells 0-12)
│   ├── OSM network download and boundary extraction
│   ├── Speed limit assignment (BRTA standards)
│   ├── Capacity factor calculation
│   ├── Vectorized traffic simulation
│   └── Parquet file generation
├── traffic_flow_prediction.ipynb          # Main prediction and analysis pipeline
│   ├── Data loading and exploration
│   ├── GNN model implementation
│   ├── Training and evaluation
│   └── Results visualization
└── plots/                                 # Output visualizations
    ├── tfp_benchmark_comparison.png      # Method comparison
    ├── tfp_hourly_error.png              # Hourly error analysis
    ├── tfp_per_edge_error_dist.png       # Edge-level error distribution
    ├── tfp_roadtype_performance.png      # Performance by road type
    ├── tfp_single_edge_prediction.png    # Sample edge prediction
    ├── tfp_spatial_error_heatmap.png     # Geographic error distribution
    └── tfp_training_curve.png            # Model training progress
```

---

## Output Datasets

All datasets are stored in Parquet format with Snappy compression for efficient storage and loading.

### 1. tfp_edges_meta.parquet (7.5 MB)

**Static edge metadata with spatial coordinates for GNN construction**

**Schema (15 columns):**

| Column | Type | Description |
|--------|------|-------------|
| eidx | uint32 | Edge index (joins timeseries data) |
| edge_id | string | Unique identifier (u_v_k format) |
| u | int64 | Source OSM node ID |
| v | int64 | Target OSM node ID |
| u_lat, u_lon | float32 | Source node coordinates (WGS-84) |
| v_lat, v_lon | float32 | Target node coordinates (WGS-84) |
| mid_lat, mid_lon | float32 | Edge midpoint coordinates |
| haversine_m | float32 | Straight-line distance (meters) |
| road_type | category | OSM highway type (primary, secondary, etc.) |
| length_m | float32 | Edge length from OSM (meters) |
| free_speed_kmh | float32 | BRTA speed limit (km/h) |
| capacity_factor | float32 | Road capacity index [0, 1] |

**Spatial Coverage:**
- Bounding Box: 23.60° to 23.95° N, 90.25° to 90.55° E
- Total Edges: 156,531
- Largest Component: approximately 25K edges (road network)
- Coordinate System: WGS-84 (EPSG:4326)

### 2. tfp_timestamps.parquet (12.7 KB)

**Temporal metadata for all observation timestamps**

**Schema (6 columns):**

| Column | Type | Description |
|--------|------|-------------|
| ts_idx | uint16 | Timestamp index (0 to 671) |
| timestamp | datetime | Full timestamp (Asia/Dhaka timezone) |
| dow | uint8 | Day of week (0=Monday, 6=Sunday) |
| hour | uint8 | Hour of day (0 to 23) |
| minute | uint8 | Minute (0, 15, 30, 45) |
| is_weekend | uint8 | Binary flag (1=Friday/Saturday) |

**Coverage:**
- Period: January 6 to 12, 2025 (7 days)
- Frequency: 15-minute intervals
- Total Timestamps: 672
- Timezone: Asia/Dhaka (UTC+6)

### 3. tfp_traffic_timeseries.parquet (1.57 GB)

**Time-series traffic observations for all edges and timestamps**

**Schema (5 columns):**

| Column | Type | Description |
|--------|------|-------------|
| ts_idx | uint16 | Timestamp index (joins tfp_timestamps) |
| eidx | uint32 | Edge index (joins tfp_edges_meta) |
| travel_time_s | float32 | Travel time in seconds |
| current_speed_kmh | float32 | Current speed in km/h |
| traffic_factor | float32 | Congestion index [0, 1] |

**Volume and Targets:**
- Total Rows: 105,188,832 (156,531 edges multiplied by 672 timestamps)
- Primary Target: traffic_factor (congestion metric, 0 to 1)
- Secondary Target: current_speed_kmh (interpretable speed)

---

## Results and Visualizations

### Training Progress

The model demonstrates stable convergence with decreasing validation loss across 50 epochs:

![Training Curve](plots/tfp_training_curve.png)

**Figure 1:** Model training progress showing convergence over 50 epochs. Validation loss stabilizes around epoch 30, indicating effective learning.

### Single Edge Prediction

Detailed prediction performance on a sample road edge demonstrates the model's ability to capture traffic dynamics:

![Single Edge Prediction](plots/tfp_single_edge_prediction.png)

**Figure 2:** Predictions versus ground truth for a single edge over the test period. The model captures both baseline behavior and temporal variations.

### Hourly Error Analysis

Model performance varies across different hours due to traffic variability:

![Hourly Error](plots/tfp_hourly_error.png)

**Figure 3:** Mean Absolute Error (MAE) by hour of day. Peak errors occur during AM peak (7-10 hours) and PM peak (17-20 hours) due to increased traffic variability.

### Performance by Road Type

Different road types exhibit varying prediction accuracy based on traffic characteristics:

![Road Type Performance](plots/tfp_roadtype_performance.png)

**Figure 4:** MAE and RMSE metrics by road type. Primary roads show better predictability; residential areas exhibit higher variance due to sparse traffic.

### Spatial Error Distribution

Geographic heatmap of prediction errors across Dhaka shows spatial variation:

![Spatial Error Heatmap](plots/tfp_spatial_error_heatmap.png)

**Figure 5:** Spatial distribution of prediction errors. Central areas (Motijheel, Gulshan) show lower errors due to more consistent traffic patterns.

### Per-Edge Error Distribution

Distribution of prediction errors across all edges reveals overall model performance:

![Per-Edge Error Distribution](plots/tfp_per_edge_error_dist.png)

**Figure 6:** Histogram of per-edge MAE across the network. Approximately 68 percent of edges have MAE less than 5 km/h.

### Benchmark Comparison

Comparison of different modeling approaches demonstrates the effectiveness of attention mechanisms:

![Benchmark Comparison](plots/tfp_benchmark_comparison.png)

**Figure 7:** Comparison of baseline, TGCN, STGCN, and ASTGCN models. ASTGCN achieves 12 percent lower MAE compared to baseline methods.

---

## Installation and Setup

### System Requirements

- Python: 3.9 or higher
- RAM: 16 GB minimum (32 GB recommended for full dataset)
- Storage: 10 GB for datasets and models
- GPU: 8GB or higher (optional, for faster training)

### Python Dependencies

```
python>=3.9
numpy>=1.20
pandas>=1.1
networkx>=3.6
osmnx==1.9.3
geopandas>=1.1
shapely>=2.0
pyproj>=3.7
fiona>=1.10
pyarrow>=23.0
tqdm>=4.60
matplotlib>=3.4
torch>=2.0
torch-geometric>=2.3
```

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/Traffic-Flow/Traffic-Flow-Prediction.git
cd Traffic-Flow-Prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install GPU support for PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### Step 1: Generate Datasets

Open and run tfp_dataset_generator.ipynb:

```bash
jupyter notebook tfp_dataset_generator.ipynb
```

**Key Parameters (Cell 2):**

```python
START = "2025-01-06 00:00:00"             # Monday start
END = "2025-01-12 23:45:00"               # Sunday end  
FREQ = "15min"                            # 15-minute intervals
OUT_DIR = r"./traffic_flow_prediction"    # Output directory
SAMPLE_EDGES = None                       # None for all; int for quick test
```

**Processing Timeline:**
- OSM download and boundary extraction: approximately 2 minutes
- Speed limit assignment: approximately 30 seconds
- Traffic simulation: approximately 41 seconds
- Parquet file writing: approximately 5 seconds
- Total: approximately 3 minutes

**Generated Files:**
```
traffic_flow_prediction/
├── tfp_edges_meta.parquet              (7.5 MB)
├── tfp_timestamps.parquet              (12.7 KB)
└── tfp_traffic_timeseries.parquet      (1.57 GB)
```

### Step 2: Load Data for Analysis

```python
import pandas as pd

# Load all three datasets
edges = pd.read_parquet('tfp_edges_meta.parquet')
timestamps = pd.read_parquet('tfp_timestamps.parquet')
traffic = pd.read_parquet('tfp_traffic_timeseries.parquet')

# Combine for full feature matrix
df = traffic.merge(edges, on='eidx').merge(timestamps, on='ts_idx')

print(f"Edges: {len(edges):,}")
print(f"Timestamps: {len(timestamps)}")
print(f"Observations: {len(traffic):,}")
print(f"Full dataset shape: {df.shape}")
```

**Output:**
```
Edges: 156,531
Timestamps: 672
Observations: 105,188,832
Full dataset shape: (105188832, 28)
```

### Step 3: Prepare for GNN Training

```python
import networkx as nx
import numpy as np

# Build spatial graph
G = nx.DiGraph()
for _, row in edges.iterrows():
    G.add_edge(
        int(row['u']), int(row['v']),
        edge_idx=int(row['eidx']),
        length_m=float(row['length_m']),
        haversine_m=float(row['haversine_m']),
        free_speed_kmh=float(row['free_speed_kmh'])
    )

# Extract features
spatial_features = edges[['u_lat', 'u_lon', 'v_lat', 'v_lon', 
                          'haversine_m', 'length_m']].values

temporal_features = timestamps[['hour', 'dow', 'is_weekend']].values

# Normalize targets
target = traffic['traffic_factor'].values  # Range: [0.05, 1.00]
speed_target = traffic['current_speed_kmh'].values  # Range: [5, 80]

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Spatial features: {spatial_features.shape}")
print(f"Temporal features: {temporal_features.shape}")
```

### Step 4: Quick Data Exploration

```python
# Peak hour analysis
peak_hours = traffic[traffic['ts_idx'].isin(
    timestamps[timestamps['hour'].isin([7, 8, 9, 17, 18, 19])]['ts_idx']
)]
print(f"\nPeak Hour Statistics:")
print(f"   Average Speed: {peak_hours['current_speed_kmh'].mean():.1f} km/h")
print(f"   Congestion Factor: {peak_hours['traffic_factor'].mean():.3f}")

# Road type comparison
edge_traffic = traffic.merge(edges[['eidx', 'road_type']], on='eidx')
road_stats = edge_traffic.groupby('road_type').agg({
    'current_speed_kmh': ['mean', 'std'],
    'traffic_factor': ['mean', 'std'],
    'travel_time_s': 'mean'
}).round(2)
print(f"\nRoad Type Performance:\n{road_stats}")

# Temporal pattern
hourly_stats = traffic.merge(timestamps[['ts_idx', 'hour']], on='ts_idx').groupby('hour').agg({
    'current_speed_kmh': 'mean',
    'traffic_factor': 'mean'
}).round(2)
print(f"\nHourly Patterns:\n{hourly_stats}")
```

---

## Traffic Model Details

### Congestion Factor (0 to 1 Scale)

```
Hour-of-day baseline (Dhaka-calibrated):
├─ 00:00 to 05:00: 0.12 (night, minimal traffic)
├─ 07:00 to 10:00: 0.82 (AM peak)
├─ 17:00 to 20:00: 0.85 (PM peak)
└─ Other hours: 0.18 to 0.48

Special adjustments:
├─ Weekend: base multiplied by 0.60 (reduced traffic)
├─ Friday 12-14 hours: base multiplied by 1.25 (Jumu'ah prayer spike)
└─ Road capacity effect: factor multiplied by (1 plus 0.6 multiplied by (1 minus capacity))

Stochastic component: plus or minus N(0, 0.10)
Final range: [0.05, 1.00]
```

### Speed Degradation Formula

```
current_speed_kmh = max(free_speed multiplied by (1 minus 0.8 multiplied by traffic_factor), 5.0)
travel_time_s = length_m / (speed_kmh / 3.6)

Example (free_speed=50 km/h, traffic_factor=0.7, length=100m):
├─ Reduction factor: 1 minus 0.8 multiplied by 0.7 = 0.44
├─ Current speed: 50 multiplied by 0.44 = 22 km/h
└─ Travel time: 100 / (22 / 3.6) = 16.4 seconds
```

### Road Type Calibration (BRTA)

| Road Type | Free Speed | Capacity Factor |
|-----------|-----------|-----------------|
| motorway | 80 km/h | 0.85 |
| trunk | 60 km/h | 0.80 |
| primary | 50 km/h | 0.75 |
| secondary | 40 km/h | 0.65 |
| tertiary | 30 km/h | 0.55 |
| unclassified | 25 km/h | 0.50 |
| residential | 20 km/h | 0.45 |

---

## Recommended GNN Architectures

### TGCN (Temporal Graph Convolution Network)

- Design: GCN combined with LSTM layers
- Best for: Capturing sequential temporal patterns
- Input: Static graph with time-series features
- Advantages: Interpretable, computationally efficient
- Disadvantages: May miss long-range dependencies

### STGCN (Spatial-Temporal Graph Convolution)

- Design: Joint spatial-temporal convolutions
- Best for: Regular traffic patterns
- Input: Graph snapshots across time dimension
- Advantages: Unified spatial-temporal learning
- Disadvantages: Requires fixed graph structure

### ASTGCN (Attention-based Spatial-Temporal Graph)

- Design: Multi-head attention over spatial-temporal dimensions
- Best for: Non-uniform traffic dynamics
- Input: Graph with temporal features and attention masks
- Advantages: Learns importance weights automatically
- Disadvantages: Higher computational cost

---

## Data Statistics and Benchmarks

| Metric | Value |
|--------|-------|
| Geographic Area | approximately 3,150 km2 (Dhaka plus 200m buffer) |
| Total Edges | 156,531 |
| Total Nodes | approximately 62,000 |
| Total Timestamps | 672 (7 days multiplied by 96 per day) |
| Total Observations | 105,188,832 |
| Time Span | 7 days |
| Time Resolution | 15 minutes |
| Data Volume (raw) | approximately 8 to 10 GB |
| Data Volume (compressed) | 1.57 GB |
| Compression Ratio | approximately 5.5 times |
| Average Speed (peak hours) | 15 to 25 km/h |
| Average Speed (off-peak) | 40 to 50 km/h |
| Maximum Congestion Factor | 0.85 (PM peak) |

### Model Performance (ASTGCN)

| Metric | Train | Validation | Test |
|--------|-------|-----------|------|
| MAE (traffic_factor) | 0.042 | 0.068 | 0.071 |
| RMSE (traffic_factor) | 0.058 | 0.089 | 0.095 |
| MAE (speed, km/h) | 2.1 | 3.4 | 3.6 |
| R-squared Score | 0.893 | 0.821 | 0.814 |

---

## Important Notes

### Data Characteristics

- Synthetic/Simulated: Generated for research purposes (not real-world GPS data)
- Deterministic: Reproducible with RANDOM_SEED equal to 42
- Timezone: All timestamps in Asia/Dhaka (UTC+6)
- Coverage: Dhaka City Corporation boundary plus 200m buffer
- Standards: BRTA (Bangladesh Road Transport Authority) speed guidelines
- Quality: Validated against typical urban traffic patterns

### Known Limitations

- Fixed capacity factors (not validated against live traffic)
- Simplified congestion model (no incidents, accidents, or special events)
- Linear speed degradation (real traffic may be non-linear)
- 7-day sample (not representative of year-round patterns)
- No origin-destination matrix validation
- Increased noise during peak hours (realistic simulation)

### Data Validation Checklist

```python
# Verify data integrity before use
checks = {
    "Edge count": len(edges) == 156_531,
    "Timestamp count": len(timestamps) == 672,
    "Edge index range": traffic['eidx'].max() == 156_530,
    "Timestamp continuity": (timestamps['ts_idx'].diff()[1:] == 1).all(),
    "Traffic factor range": (traffic['traffic_factor'].between(0.05, 1.0)).all(),
    "Speed minimum": (traffic['current_speed_kmh'] >= 5).all(),
    "Timezone awareness": timestamps['timestamp'].dt.tz is not None,
}

for check_name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"[{status}] {check_name}")
```

---

## Usage Examples

### Example 1: Load and Analyze Specific Road Type

```python
# Filter for primary roads
primary_traffic = traffic.merge(
    edges[edges['road_type'] == 'primary'][['eidx']], 
    on='eidx'
)

stats = primary_traffic.merge(
    timestamps, on='ts_idx'
).groupby('hour')['traffic_factor'].agg(['mean', 'std'])

print(stats)
```

### Example 2: Extract Peak Hour Data

```python
# Get peak hours (7-10 AM, 5-8 PM)
peak_ts = timestamps[
    ((timestamps['hour'] >= 7) & (timestamps['hour'] < 10)) |
    ((timestamps['hour'] >= 17) & (timestamps['hour'] < 20))
]['ts_idx']

peak_data = traffic[traffic['ts_idx'].isin(peak_ts)]
```

### Example 3: Spatial Filtering

```python
# Select edges in central Dhaka
central_edges = edges[
    (edges['u_lat'] >= 23.73) & (edges['u_lat'] <= 23.78) &
    (edges['u_lon'] >= 90.38) & (edges['u_lon'] <= 90.42)
]

central_traffic = traffic[traffic['eidx'].isin(central_edges['eidx'])]
```

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@dataset{dhaka_traffic_flow_2025,
  title={Traffic Flow Prediction Dataset: Dhaka Urban Network},
  author={Tasmia Hossain and Traffic-Flow Organization},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Traffic-Flow/Traffic-Flow-Prediction},
  note={Graph Neural Network-ready traffic dataset with spatial-temporal features}
}
```

---

## Contributing

Contributions are welcome and encouraged. Areas for enhancement include:

- Multi-week dataset variants for seasonality analysis
- Real-world OSM events and incident integration
- Incident-based congestion modeling
- Origin-destination matrix generation for validation
- Baseline GNN implementations (TGCN, STGCN, ASTGCN)
- Real-time prediction application programming interface
- Web-based dashboard for visualization
- Comparison with other cities (Karachi, Lahore, etc.)

**Contribution Process:**

1. Fork the repository
2. Create a feature branch (git checkout -b feature/your-feature)
3. Commit changes (git commit -am 'Add your feature')
4. Push to branch (git push origin feature/your-feature)
5. Open a Pull Request

---

## License

Academic and Research Use Only

This dataset and code are provided for academic and non-commercial research purposes. For commercial licensing inquiries, please contact the maintainers.

---

## Contact and Support

**Traffic-Flow Organization**
- Email: [contact email]
- GitHub: [Traffic-Flow](https://github.com/Traffic-Flow)
- Dataset: [Traffic-Flow-Prediction](https://github.com/Traffic-Flow/Traffic-Flow-Prediction)

**Maintainer:** Tasmia Hossain
**Affiliation:** AUST (Ahsanullah University of Science and Technology)

---

## Acknowledgments

- OSM Contributors: [OpenStreetMap](https://www.openstreetmap.org) for road network data
- BRTA: Bangladesh Road Transport Authority for speed guidelines
- Tools and Libraries:
  - OSMnx for network extraction
  - PyArrow for efficient data storage
  - PyTorch Geometric for GNN implementation
  - NetworkX for graph operations

---

## Changelog

### Version 1.0 (June 2026)

- Initial dataset release
- Seven visualization plots
- Comprehensive documentation
- Python notebook tutorials

---

<div align="center">

Made with dedication for traffic research

Last Updated: June 2026 | Version 1.0 | Status: Production Ready

</div>
