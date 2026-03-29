# Drone Swarm Irrigation System

A multi-agent drone swarm simulation for precision irrigation in smart agriculture. Drones autonomously detect thermally stressed crop zones and coordinate to irrigate them efficiently, minimising water usage while maximising field coverage.

---

## 📁 Project Structure

```
drone-swarm-irrigation/
├── thermal_simulation.py     # Phase 1 — Generate a synthetic thermal field
├── swarm_algorithm.py        # Phase 2 — Full drone swarm with leader/member comms
├── demo.py                   # Phase 3 — Integrated demo + final dashboard
├── drone_irrigation1.nlogox  # NetLogo model (visualisation / extended simulation)
├── output/                   # Auto-generated simulation output files
│   ├── thermal_field.npy         # Saved thermal field (NumPy binary)
│   ├── thermal_data.csv          # Thermal field as CSV (for NetLogo)
│   ├── thermal_map.png           # Heatmap + distribution plot (Phase 1)
│   ├── thermal_final.csv         # Post-irrigation thermal field
│   ├── irrigated_map.csv         # Binary map of irrigated cells
│   ├── drone_paths.csv           # Per-tick drone positions
│   ├── simulation_history.csv    # Tick-by-tick metrics
│   └── final_dashboard.png       # Comprehensive 6-panel dashboard (Phase 3)
└── .gitignore
```

---

## ⚙️ How It Works

The simulation runs in three sequential phases:

### Phase 1 — Thermal Field Generation (`thermal_simulation.py`)

- Creates a **50×50 grid** representing a crop field.
- Randomly places **6 hotspots** (stressed crop zones) with bell-curve heat profiles.
- Applies a **Gaussian blur** to simulate realistic heat spread.
- Temperature range: baseline **28 °C**, hotspot peak up to **~200 °C** above baseline before blur.
- Saves `thermal_field.npy` and `thermal_data.csv` for use by subsequent phases.
- Produces `thermal_map.png` showing the heatmap and temperature distribution histogram.

### Phase 2 — Swarm Algorithm (`swarm_algorithm.py`)

Implements a **leader–member drone swarm** with direct inter-drone communication.

#### `Drone` class
| Attribute | Value | Description |
|-----------|-------|-------------|
| `sense_radius` | 15 cells | Local scan radius for stressed cells |
| `speed` | 0.8 cells/tick | Movement speed |
| `water_capacity` | 300 units | Max water per drone |

Key behaviours:
- **`sense_environment()`** — scans local radius for the hottest un-irrigated stressed cell (≥ 35 °C).
- **`report_to_leader()`** — sends its hottest local cell to the leader each tick.
- **`move_toward()`** — moves toward target without overshooting.
- **`irrigate()`** — cools a stressed cell by **8 °C** and marks it as irrigated.

#### `DroneSwarm` class
Key behaviours:
- **Leader–member communication** — every tick, member drones report their hottest local cell to the designated leader. The leader sorts reports by temperature and assigns **unique targets** (no two drones chase the same cell).
- **Leader failover** — if the leader drone exhausts its water supply, it is demoted and the drone with the lowest water usage is automatically promoted.
- **Collision avoidance** — drones maintain a minimum separation of 4 cells via a repulsion mechanism.
- **Global fallback** — if local sensing finds no target, drones navigate to the globally hottest un-irrigated stressed cell.

Output files saved by `save_results()`:
- `thermal_final.csv`, `irrigated_map.csv`, `drone_paths.csv`, `simulation_history.csv`, `comm_log.csv`, `failover_log.txt`

### Phase 3 — Integrated Demo & Dashboard (`demo.py`)

Runs Phase 1 and Phase 2 end-to-end in a single script and generates a **6-panel final dashboard** (`output/final_dashboard.png`):

| Panel | Contents |
|-------|----------|
| Thermal Field — BEFORE | Heatmap before irrigation, hotspot markers |
| Thermal Field — AFTER | Heatmap after irrigation, final drone positions |
| Irrigated Map | Red (un-irrigated) / green (irrigated) cell map |
| Drone Paths | Scatter of last 500 steps per drone |
| Irrigation Progress | % irrigated vs. tick |
| Mean Field Temperature | Mean temp over time vs. stress threshold |
| Cumulative Water Used | Litres consumed vs. tick |
| Simulation Stats | Summary text box with key metrics |

**Key configuration parameters** (editable at the top of `demo.py`):

```python
GRID_SIZE        = 50       # Field dimensions (cells)
NUM_HOTSPOTS     = 6        # Number of stressed crop zones
BASE_TEMP        = 28.0     # Baseline temperature (°C)
STRESS_THRESHOLD = 35.0     # Temperature above which a cell needs irrigation
HOTSPOT_TEMP     = 42.0     # Peak hotspot temperature (°C)
HOTSPOT_RADIUS   = 4        # Hotspot radius (cells)
NUM_DRONES       = 5        # Number of drones in the swarm
SENSE_RADIUS     = 7        # Drone sensing radius (cells)
IRRIGATE_COOL    = 8.0      # Temperature reduction on irrigation (°C)
REPULSION_DIST   = 3        # Minimum distance between drones
MAX_TICKS        = 300      # Maximum simulation steps
RANDOM_SEED      = 42       # Fixed seed for reproducibility
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib scipy
```

### Run the Full Simulation

```bash
python demo.py
```

This runs all three phases and saves all outputs to the `output/` folder.

### Run Individual Phases

```bash
# Phase 1 only — generate thermal field
python thermal_simulation.py

# Phase 2 only — run the swarm (requires thermal_field.npy in output/)
python swarm_algorithm.py
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `output/thermal_field.npy` | NumPy binary of the raw thermal field |
| `output/thermal_data.csv` | CSV of the thermal field (readable by NetLogo) |
| `output/thermal_map.png` | Heatmap + histogram from Phase 1 |
| `output/thermal_final.csv` | Thermal field after all irrigation |
| `output/irrigated_map.csv` | 50×50 binary matrix (1 = irrigated) |
| `output/drone_paths.csv` | Columns: `tick, drone_id, row, col` |
| `output/simulation_history.csv` | Columns: `tick, irrigated_pct, mean_temp_C, water_used_L` |
| `output/comm_log.csv` | Full inter-drone communication log |
| `output/failover_log.txt` | Leader failover events |
| `output/final_dashboard.png` | 6-panel summary dashboard |

---

## 🌐 NetLogo Integration

The file `drone_irrigation1.nlogox` is a **NetLogo model** that can read `thermal_data.csv` to visualise the field and drone behaviour in NetLogo's agent-based environment.

Open it in [NetLogo](https://ccl.northwestern.edu/netlogo/) (version 6+).

---

## 🧠 Algorithm Summary

```
Each tick:
  1. Member drones scan their local radius → report hottest cell to leader
  2. Leader sorts reports by temperature, assigns unique targets (no overlap)
  3. Each drone moves toward its target at speed 0.8 cells/tick
  4. On arrival, drone cools the cell by 8 °C and marks it irrigated
  5. Collision avoidance adjusts positions if drones get within 4 cells
  6. If leader is out of water → failover: drone with least water used is promoted
  7. Simulation ends when all stressed cells (≥ 35 °C) are irrigated or MAX_TICKS is reached
```

---

## 📄 License

This project is for educational and research purposes.
