import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import os

# ── Configuration ─────────────────────────────────────────────────────────────
GRID_SIZE    = 50          # 50×50 field
NUM_HOTSPOTS = 6           # number of dry/stressed crop zones
BASE_TEMP    = 28.0        # baseline field temperature (°C)
HOTSPOT_PEAK = 200.0        # how much hotter a hotspot is above baseline
SIGMA        = 2.0         # how wide the heat spreads (Gaussian radius)
RANDOM_SEED  = 42          # fixed seed so results are reproducible
OUTPUT_DIR   = "output"    # folder where files are saved

# ── Setup ──────────────────────────────────────────────────────────────────────
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)   # creates output/ if it doesn't exist

# ── Build the thermal field ────────────────────────────────────────────────────
field = np.full((GRID_SIZE, GRID_SIZE), BASE_TEMP)  # start: every cell = 28°C

hotspot_locations = []
for _ in range(NUM_HOTSPOTS):
    row = np.random.randint(5, GRID_SIZE - 5)   # keep hotspots away from edges
    col = np.random.randint(5, GRID_SIZE - 5)
    hotspot_locations.append((row, col))
    field[row, col] += HOTSPOT_PEAK             # spike the temperature

# Spread each spike using a Gaussian (bell-curve) blur
field = gaussian_filter(field, sigma=SIGMA)

print(f"✅ Thermal field created")
print(f"   Grid size    : {GRID_SIZE}×{GRID_SIZE}")
print(f"   Hotspots     : {NUM_HOTSPOTS} at {hotspot_locations}")
print(f"   Temp range   : {field.min():.1f}°C – {field.max():.1f}°C")

# ── Save outputs ───────────────────────────────────────────────────────────────

# 1. Save as .npy (Python binary — used by swarm_algorithm.py on Day 2)
npy_path = os.path.join(OUTPUT_DIR, "thermal_field.npy")
np.save(npy_path, field)
print(f"✅ Saved: {npy_path}")

# 2. Save as .csv (NetLogo will read this on Day 3)
csv_path = os.path.join(OUTPUT_DIR, "thermal_data.csv")
np.savetxt(csv_path, field, delimiter=",", fmt="%.2f")
print(f"✅ Saved: {csv_path}")

# ── Visualise ──────────────────────────────────────────────────────────────────

# Colour map: green (cool) → yellow (warm) → red (hot)
cmap = plt.cm.RdYlGn_r

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Day 1 — Thermal Field Overview", fontsize=14, fontweight='bold')

# Left: heatmap
im = axes[0].imshow(field, cmap=cmap, interpolation='bilinear')
axes[0].set_title("Thermal Field (°C)")
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")
plt.colorbar(im, ax=axes[0], label="Temperature (°C)")

# Mark hotspot centres
for (r, c) in hotspot_locations:
    axes[0].plot(c, r, 'b^', markersize=8, label='Hotspot centre')
axes[0].legend(loc='upper right', fontsize=8)

# Right: temperature histogram
axes[1].hist(field.flatten(), bins=40, color='tomato', edgecolor='white', linewidth=0.5)
axes[1].set_title("Temperature Distribution")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Number of cells")
axes[1].axvline(35, color='darkred', linestyle='--', linewidth=1.5, label='Stress threshold (35°C)')
axes[1].legend()

plt.tight_layout()

# Save heatmap as PNG
png_path = os.path.join(OUTPUT_DIR, "thermal_map.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {png_path}")

print("\n🎉 Day 1 complete! Check your output/ folder.")
print("   Close the plot window to exit.")

plt.show()


