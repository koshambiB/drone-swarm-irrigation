"""
main_demo.py  —  Day 3: Full integrated demo + dashboard
Runs thermal generation + swarm algorithm together and produces
a comprehensive final_dashboard.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import csv

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
GRID_SIZE        = 50
NUM_HOTSPOTS     = 6
BASE_TEMP        = 28.0
STRESS_THRESHOLD = 35.0
HOTSPOT_TEMP     = 42.0      # peak temp inside each hotspot (well above 35)
HOTSPOT_RADIUS   = 4         # radius of each hotspot in cells
NUM_DRONES       = 5
SENSE_RADIUS     = 7
IRRIGATE_COOL    = 8.0
REPULSION_DIST   = 3
MAX_TICKS        = 300
RANDOM_SEED      = 42
OUTPUT_DIR       = "output"

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — GENERATE THERMAL FIELD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("  PHASE 1 — Generating Thermal Field")
print("=" * 55)

field_original = np.full((GRID_SIZE, GRID_SIZE), BASE_TEMP, dtype=float)

# Place hotspots as filled circles with smooth falloff
hotspot_locations = []
for _ in range(NUM_HOTSPOTS):
    cr = np.random.randint(HOTSPOT_RADIUS + 2, GRID_SIZE - HOTSPOT_RADIUS - 2)
    cc = np.random.randint(HOTSPOT_RADIUS + 2, GRID_SIZE - HOTSPOT_RADIUS - 2)
    hotspot_locations.append((cr, cc))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            dist = np.sqrt((r - cr)**2 + (c - cc)**2)
            if dist <= HOTSPOT_RADIUS:
                # Smooth bell curve: hottest at centre, fades to BASE_TEMP at edge
                heat = (HOTSPOT_TEMP - BASE_TEMP) * (1 - (dist / HOTSPOT_RADIUS) ** 2)
                field_original[r, c] = max(field_original[r, c], BASE_TEMP + heat)

stressed_count = int(np.sum(field_original >= STRESS_THRESHOLD))

# Save for NetLogo / standalone Day 2 use
np.save(os.path.join(OUTPUT_DIR, "thermal_field.npy"), field_original)
np.savetxt(os.path.join(OUTPUT_DIR, "thermal_data.csv"),
           field_original, delimiter=",", fmt="%.2f")

print(f"  Grid        : {GRID_SIZE}x{GRID_SIZE}")
print(f"  Hotspots    : {NUM_HOTSPOTS} (radius={HOTSPOT_RADIUS} cells, peak={HOTSPOT_TEMP}C)")
print(f"  Temp range  : {field_original.min():.1f}C - {field_original.max():.1f}C")
print(f"  Stressed    : {stressed_count} cells (>={STRESS_THRESHOLD}C)")
print("  Thermal field ready\n")

if stressed_count == 0:
    raise RuntimeError("Still no stressed cells — check HOTSPOT_TEMP and STRESS_THRESHOLD.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SWARM ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("  PHASE 2 — Running Drone Swarm")
print("=" * 55)

field      = field_original.copy()
irrigated  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
water_used = 0.0

drone_positions = np.array([
    [float(np.random.randint(0, GRID_SIZE)),
     float(np.random.randint(0, GRID_SIZE))]
    for _ in range(NUM_DRONES)
])

path_records = []
history      = []

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def find_target(dr, dc):
    """Hottest unirrigated stressed cell within SENSE_RADIUS."""
    best_temp, best_cell = -np.inf, None
    r_int = int(round(dr)); c_int = int(round(dc))
    r0 = max(0, r_int - SENSE_RADIUS);  r1 = min(GRID_SIZE, r_int + SENSE_RADIUS + 1)
    c0 = max(0, c_int - SENSE_RADIUS);  c1 = min(GRID_SIZE, c_int + SENSE_RADIUS + 1)
    for r in range(r0, r1):
        for c in range(c0, c1):
            if not irrigated[r, c] and field[r, c] >= STRESS_THRESHOLD:
                if field[r, c] > best_temp:
                    best_temp, best_cell = field[r, c], (r, c)
    return best_cell

def global_hottest():
    """Fallback: hottest stressed unirrigated cell anywhere."""
    masked = np.where(~irrigated, field, -np.inf)
    if masked.max() < STRESS_THRESHOLD:
        return None
    return np.unravel_index(np.argmax(masked), masked.shape)

for tick in range(MAX_TICKS):
    remaining = int(np.sum((field >= STRESS_THRESHOLD) & ~irrigated))
    if remaining == 0:
        print(f"  All stressed cells irrigated at tick {tick}!")
        break

    irrigated_pct = 100.0 * irrigated.sum() / max(stressed_count, 1)
    history.append((tick, irrigated_pct, field.mean(), water_used))

    if tick % 25 == 0:
        print(f"  Tick {tick:>4} | {irrigated_pct:5.1f}% irrigated "
              f"| {field.mean():.2f}C mean | {water_used:.0f} L "
              f"| {remaining} cells left")

    for i, pos in enumerate(drone_positions):
        dr, dc = pos
        target = find_target(dr, dc) or global_hottest()

        if target is not None:
            tr, tc = target
            diff_r = tr - dr;  diff_c = tc - dc
            dist   = max(np.sqrt(diff_r**2 + diff_c**2), 0.001)
            step_r = diff_r / dist;  step_c = diff_c / dist
        else:
            angle  = np.random.uniform(0, 2 * np.pi)
            step_r = np.sin(angle);  step_c = np.cos(angle)

        rep_r, rep_c = 0.0, 0.0
        for j, other in enumerate(drone_positions):
            if j == i:
                continue
            d = np.sqrt((dr - other[0])**2 + (dc - other[1])**2)
            if 0 < d < REPULSION_DIST:
                rep_r += (dr - other[0]) / d
                rep_c += (dc - other[1]) / d

        move_r = step_r + 0.5 * rep_r
        move_c = step_c + 0.5 * rep_c
        mag    = max(np.sqrt(move_r**2 + move_c**2), 0.001)
        move_r /= mag;  move_c /= mag

        new_r = clamp(dr + move_r, 0, GRID_SIZE - 1)
        new_c = clamp(dc + move_c, 0, GRID_SIZE - 1)
        drone_positions[i] = [new_r, new_c]

        ri = clamp(int(round(new_r)), 0, GRID_SIZE - 1)
        ci = clamp(int(round(new_c)), 0, GRID_SIZE - 1)
        if field[ri, ci] >= STRESS_THRESHOLD and not irrigated[ri, ci]:
            field[ri, ci]    -= IRRIGATE_COOL
            irrigated[ri, ci] = True
            water_used        += 1.0

        path_records.append((tick, i, new_r, new_c))

final_tick        = tick + 1
irrigated_pct_end = 100.0 * irrigated.sum() / max(stressed_count, 1)
history.append((final_tick, irrigated_pct_end, field.mean(), water_used))

print(f"\n  Swarm Summary")
print(f"     Ticks        : {final_tick}")
print(f"     Irrigated    : {irrigated.sum()} / {stressed_count} cells ({irrigated_pct_end:.1f}%)")
print(f"     Mean temp    : {field.mean():.2f}C  (was {field_original.mean():.2f}C)")
print(f"     Water used   : {water_used:.0f} L")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SAVE ALL CSV FILES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  PHASE 3 — Saving outputs")
print("=" * 55)

with open(os.path.join(OUTPUT_DIR, "drone_paths.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tick", "drone_id", "row", "col"])
    for rec in path_records:
        w.writerow([rec[0], rec[1], f"{rec[2]:.3f}", f"{rec[3]:.3f}"])
print("  drone_paths.csv saved")

np.savetxt(os.path.join(OUTPUT_DIR, "irrigated_map.csv"),
           irrigated.astype(int), delimiter=",", fmt="%d")
print("  irrigated_map.csv saved")

with open(os.path.join(OUTPUT_DIR, "simulation_history.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tick", "irrigated_pct", "mean_temp_C", "water_used_L"])
    for row in history:
        w.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.4f}", f"{row[3]:.1f}"])
print("  simulation_history.csv saved")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — FINAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  PHASE 4 — Generating Final Dashboard")
print("=" * 55)

ticks_arr = [h[0] for h in history]
irr_arr   = [h[1] for h in history]
temp_arr  = [h[2] for h in history]
water_arr = [h[3] for h in history]

cmap = plt.cm.RdYlGn_r
vmin = field_original.min()
vmax = field_original.max()

fig = plt.figure(figsize=(18, 11))
fig.suptitle("Drone Swarm Irrigation - Final Dashboard",
             fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

# ── Row 0: Thermal maps ────────────────────────────────────────────────────
ax_before = fig.add_subplot(gs[0, 0:2])
im0 = ax_before.imshow(field_original, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation="bilinear")
ax_before.set_title("Thermal Field - BEFORE", fontweight="bold")
ax_before.set_xlabel("Column"); ax_before.set_ylabel("Row")
for (r, c) in hotspot_locations:
    ax_before.plot(c, r, "b^", markersize=7)
plt.colorbar(im0, ax=ax_before, label="C", fraction=0.046)

ax_after = fig.add_subplot(gs[0, 2:4])
im1 = ax_after.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="bilinear")
ax_after.set_title("Thermal Field - AFTER", fontweight="bold")
ax_after.set_xlabel("Column")
for pos in drone_positions:
    ax_after.plot(pos[1], pos[0], "b^", markersize=7)
plt.colorbar(im1, ax=ax_after, label="C", fraction=0.046)

# ── Row 1: Irrigated map + Drone paths ────────────────────────────────────
ax_irr = fig.add_subplot(gs[1, 0:2])
irr_display = np.zeros((GRID_SIZE, GRID_SIZE, 3))
irr_display[~irrigated] = [0.85, 0.2, 0.2]
irr_display[irrigated]  = [0.2, 0.75, 0.3]
ax_irr.imshow(irr_display, interpolation="nearest")
ax_irr.set_title(
    f"Irrigated Map  ({irrigated.sum()}/{stressed_count} cells  {irrigated_pct_end:.1f}%)",
    fontweight="bold")
ax_irr.set_xlabel("Column"); ax_irr.set_ylabel("Row")

ax_paths = fig.add_subplot(gs[1, 2:4])
colors_d = plt.cm.tab10(np.linspace(0, 0.5, NUM_DRONES))
recent   = path_records[-min(500, len(path_records)):]
for d_id in range(NUM_DRONES):
    pts = [(r[2], r[3]) for r in recent if r[1] == d_id]
    if pts:
        rs, cs = zip(*pts)
        ax_paths.scatter(cs, rs, s=3, color=colors_d[d_id],
                         label=f"Drone {d_id}", alpha=0.7)
ax_paths.set_xlim(0, GRID_SIZE); ax_paths.set_ylim(GRID_SIZE, 0)
ax_paths.set_title("Drone Paths (last 500 steps)", fontweight="bold")
ax_paths.set_xlabel("Column"); ax_paths.set_ylabel("Row")
ax_paths.legend(loc="upper right", fontsize=7, markerscale=4)

# ── Row 2: Time-series + Stats ─────────────────────────────────────────────
ax_pct = fig.add_subplot(gs[2, 0])
ax_pct.plot(ticks_arr, irr_arr, color="steelblue", linewidth=2)
ax_pct.fill_between(ticks_arr, irr_arr, alpha=0.2, color="steelblue")
ax_pct.set_title("Irrigation Progress", fontweight="bold")
ax_pct.set_xlabel("Tick"); ax_pct.set_ylabel("Irrigated (%)")
ax_pct.set_ylim(0, 105)

ax_temp = fig.add_subplot(gs[2, 1])
ax_temp.plot(ticks_arr, temp_arr, color="tomato", linewidth=2)
ax_temp.axhline(STRESS_THRESHOLD, color="darkred", linestyle="--",
                linewidth=1.2, label=f"Stress ({STRESS_THRESHOLD}C)")
ax_temp.set_title("Mean Field Temperature", fontweight="bold")
ax_temp.set_xlabel("Tick"); ax_temp.set_ylabel("C")
ax_temp.legend(fontsize=8)

ax_water = fig.add_subplot(gs[2, 2])
ax_water.plot(ticks_arr, water_arr, color="mediumseagreen", linewidth=2)
ax_water.fill_between(ticks_arr, water_arr, alpha=0.2, color="mediumseagreen")
ax_water.set_title("Cumulative Water Used", fontweight="bold")
ax_water.set_xlabel("Tick"); ax_water.set_ylabel("Litres")

ax_stats = fig.add_subplot(gs[2, 3])
ax_stats.axis("off")
temp_drop = field_original.mean() - field.mean()
stats_text = (
    f"SIMULATION STATS\n"
    f"{'─'*28}\n"
    f"Drones          :  {NUM_DRONES}\n"
    f"Grid size       :  {GRID_SIZE}x{GRID_SIZE}\n"
    f"Total ticks     :  {final_tick}\n"
    f"Stressed cells  :  {stressed_count}\n"
    f"Irrigated cells :  {irrigated.sum()}\n"
    f"Coverage        :  {irrigated_pct_end:.1f}%\n"
    f"Water used      :  {water_used:.0f} L\n"
    f"Mean temp before:  {field_original.mean():.2f}C\n"
    f"Mean temp after :  {field.mean():.2f}C\n"
    f"Temp reduction  :  {temp_drop:.2f}C"
)
ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
              fontsize=9.5, verticalalignment="top",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                        edgecolor="goldenrod", linewidth=1.5))

dashboard_path = os.path.join(OUTPUT_DIR, "final_dashboard.png")
plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {dashboard_path}")

print("\n" + "=" * 55)
print("  Day 3 Complete!")
print("=" * 55)
print("  All files saved to output/")
print("  Close the plot window to exit.")

plt.show()