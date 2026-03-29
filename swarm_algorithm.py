"""
swarm_algorithm.py
DAY 2 - Drone Swarm Logic
Each drone finds hot zones and moves toward them.
Run AFTER thermal_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import os

# ──────────────────────────────────────────────
# DRONE CLASS - One drone agent
# ──────────────────────────────────────────────
class Drone:
    """
    Represents a single irrigation drone.
    
    Each drone:
    - Has a position (x, y) on the field grid
    - Senses temperature in a radius around itself
    - Moves toward the hottest nearby cell
    - Irrigates cells it visits (marks them as watered)
    - Avoids other drones (simple collision avoidance)
    """

    def __init__(self, drone_id, x, y, sense_radius=5, speed=1.5, water_capacity=100):
        self.id            = drone_id
        self.x             = float(x)
        self.y             = float(y)
        self.sense_radius  = sense_radius    # How far the drone can "see"
        self.speed         = speed           # How fast it moves per tick
        self.water_capacity = water_capacity # Total water units it carries
        self.water_used     = 0             # Water used so far
        self.path           = [(x, y)]      # Record of positions visited
        self.status         = "searching"   # searching / irrigating / returning / done

    def sense_environment(self, thermal_field):
        """
        Look at nearby cells and return the hottest one.
        Returns: (best_x, best_y, max_temp) or None if nothing hot nearby
        """
        grid_h, grid_w = thermal_field.shape
        best_x, best_y, best_temp = None, None, -999

        for dx in range(-self.sense_radius, self.sense_radius + 1):
            for dy in range(-self.sense_radius, self.sense_radius + 1):
                nx = int(self.x) + dx
                ny = int(self.y) + dy

                # Stay within grid bounds
                if 0 <= nx < grid_h and 0 <= ny < grid_w:
                    temp = thermal_field[nx, ny]
                    if temp > best_temp:
                        best_temp = temp
                        best_x, best_y = nx, ny

        return (best_x, best_y, best_temp)

    def move_toward(self, target_x, target_y):
        """Move one step toward the target position."""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0.5:  # Only move if not already there
            # Normalize direction, then multiply by speed
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
            self.path.append((self.x, self.y))

    def irrigate(self, thermal_field, irrigated_map, cooling=8.0):
        """
        Cool down the current cell (simulate irrigation).
        Reduces temperature by 'cooling' degrees.
        """
        xi, yi = int(self.x), int(self.y)
        grid_h, grid_w = thermal_field.shape

        if 0 <= xi < grid_h and 0 <= yi < grid_w:
            if not irrigated_map[xi, yi] and thermal_field[xi, yi] >= 33:
                thermal_field[xi, yi] -= cooling
                thermal_field[xi, yi] = max(thermal_field[xi, yi], 25)
                irrigated_map[xi, yi] = True
                self.water_used += 1
                self.status = "irrigating"
                return True

        self.status = "searching"
        return False

    def is_out_of_water(self):
        return self.water_used >= self.water_capacity

    @property
    def position(self):
        return (self.x, self.y)


# ──────────────────────────────────────────────
# SWARM CLASS - Manages all drones together
# ──────────────────────────────────────────────
class DroneSwarm:
    """
    Manages a fleet of drones working together to irrigate a field.
    
    Key behaviors:
    1. Drones spread out to cover different zones
    2. Each drone independently finds and irrigates hot cells
    3. Simple collision avoidance (drones repel each other)
    """

    def __init__(self, thermal_field, num_drones=5, seed=42):
        self.thermal_field  = thermal_field.copy()   # Working copy
        self.original_field = thermal_field.copy()   # Keep original for comparison
        self.grid_h, self.grid_w = thermal_field.shape
        self.irrigated_map  = np.zeros_like(thermal_field, dtype=bool)
        self.ticks          = 0
        self.history        = []  # For animation

        # Spawn drones at different starting positions
        np.random.seed(seed)
        self.drones = []
        for i in range(num_drones):
            # Spread drones across the field initially
            x = np.random.randint(2, self.grid_h - 2)
            y = np.random.randint(2, self.grid_w - 2)
            drone = Drone(
                drone_id      = i,
                x             = x,
                y             = y,
                sense_radius  = 7,
                speed         = 1.2,
                water_capacity= 150
            )
            self.drones.append(drone)

        print(f"🚁 Swarm initialized: {num_drones} drones on {self.grid_h}×{self.grid_w} field")

    def apply_collision_avoidance(self):
        """
        Very simple: if two drones are too close, push them apart slightly.
        This mimics real swarm repulsion behaviour.
        """
        MIN_DIST = 3.0
        for i, d1 in enumerate(self.drones):
            for j, d2 in enumerate(self.drones):
                if i >= j:
                    continue
                dx = d1.x - d2.x
                dy = d1.y - d2.y
                dist = np.sqrt(dx**2 + dy**2)

                if dist < MIN_DIST and dist > 0:
                    # Push drones apart
                    push = (MIN_DIST - dist) / 2
                    d1.x += (dx / dist) * push
                    d1.y += (dy / dist) * push
                    d2.x -= (dx / dist) * push
                    d2.y -= (dy / dist) * push

                    # Clamp to grid
                    d1.x = np.clip(d1.x, 0, self.grid_h - 1)
                    d1.y = np.clip(d1.y, 0, self.grid_w - 1)
                    d2.x = np.clip(d2.x, 0, self.grid_h - 1)
                    d2.y = np.clip(d2.y, 0, self.grid_w - 1)

    def step(self):
        """Advance the simulation by one time-step (tick)."""
        self.ticks += 1

        for drone in self.drones:
            if drone.is_out_of_water():
                drone.status = "done"
                continue

            # 1. Sense nearby environment
            tx, ty, temp = drone.sense_environment(self.thermal_field)

            if tx is None:
                continue

            # 2. Move toward hottest zone
            drone.move_toward(tx, ty)

            # 3. Try to irrigate current location
            drone.irrigate(self.thermal_field, self.irrigated_map)

        # 4. Collision avoidance (keep drones spread out)
        self.apply_collision_avoidance()

        # 5. Record snapshot for analysis
        self.history.append({
            'tick'          : self.ticks,
            'irrigated_pct' : self.irrigated_map.mean() * 100,
            'mean_temp'     : self.thermal_field.mean(),
            'drone_positions': [(d.x, d.y) for d in self.drones],
        })

    def run(self, max_ticks=200, threshold=35.0):
        """
        Run the full simulation.
        Stops when all stressed cells are irrigated OR max_ticks reached.
        """
        print(f"\n🏃 Running swarm simulation (max {max_ticks} ticks)...")
        print(f"   Irrigation threshold: {threshold}°C")
        print(f"   Initial stressed cells: {(self.thermal_field >= threshold).sum()}")

        for t in range(max_ticks):
            self.step()

            # Print progress every 20 ticks
            if t % 20 == 0:
                stressed = (self.thermal_field >= threshold).sum()
                irrigated_pct = self.irrigated_map.mean() * 100
                print(f"   Tick {t:3d}: {stressed:3d} stressed cells | "
                      f"{irrigated_pct:.1f}% irrigated | "
                      f"Mean temp: {self.thermal_field.mean():.1f}°C")

            # Stop early if no more stressed cells
            if (self.thermal_field >= threshold).sum() == 0:
                print(f"\n✅ All stressed zones irrigated at tick {t}!")
                break

        print(f"\n📊 Simulation Complete ({self.ticks} ticks)")
        self.print_final_report()

    def print_final_report(self):
        total_irrigated = self.irrigated_map.sum()
        total_cells     = self.grid_h * self.grid_w
        total_water     = sum(d.water_used for d in self.drones)

        print(f"\n{'='*50}")
        print(f"  FINAL SWARM REPORT")
        print(f"{'='*50}")
        print(f"  Total ticks run     : {self.ticks}")
        print(f"  Cells irrigated     : {total_irrigated}/{total_cells} ({total_irrigated/total_cells*100:.1f}%)")
        print(f"  Total water used    : {total_water} units")
        print(f"  Mean field temp now : {self.thermal_field.mean():.1f}°C")
        print(f"  Remaining stress    : {(self.thermal_field >= 35).sum()} cells ≥35°C")
        print(f"{'='*50}")
        for d in self.drones:
            print(f"  Drone {d.id}: water_used={d.water_used}, "
                  f"path_length={len(d.path)}, status={d.status}")

    def save_results(self, output_dir="output"):
        """Save all results needed by NetLogo and main_demo.py"""
        os.makedirs(output_dir, exist_ok=True)

        # Save final thermal state
        np.savetxt(f"{output_dir}/thermal_final.csv",
                   self.thermal_field, delimiter=",", fmt="%.2f")

        # Save irrigated map (True/False → 1/0)
        np.savetxt(f"{output_dir}/irrigated_map.csv",
                   self.irrigated_map.astype(int), delimiter=",", fmt="%d")

        # Save drone paths
        all_paths = []
        for d in self.drones:
            for step_num, (px, py) in enumerate(d.path):
                all_paths.append([d.id, step_num, px, py])

        path_arr = np.array(all_paths)
        np.savetxt(f"{output_dir}/drone_paths.csv", path_arr,
                   delimiter=",", fmt="%.2f",
                   header="drone_id,step,x,y", comments="")

        # Save tick-by-tick history
        ticks       = [h['tick'] for h in self.history]
        irr_pct     = [h['irrigated_pct'] for h in self.history]
        mean_temps  = [h['mean_temp'] for h in self.history]

        history_arr = np.column_stack([ticks, irr_pct, mean_temps])
        np.savetxt(f"{output_dir}/simulation_history.csv", history_arr,
                   delimiter=",", fmt="%.4f",
                   header="tick,irrigated_pct,mean_temp", comments="")

        print(f"\n✅ All results saved to '{output_dir}/'")

    def visualize_result(self, save_path=None):
        """Show before/after comparison + drone paths."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Swarm Irrigation Simulation Results", fontsize=16, fontweight='bold')

        cmap = plt.cm.RdYlBu_r

        # ── Before ──────────────────────────────
        ax1 = axes[0]
        im1 = ax1.imshow(self.original_field, cmap=cmap, origin='lower', vmin=25, vmax=50)
        plt.colorbar(im1, ax=ax1, label='°C')
        ax1.set_title('Before Irrigation', fontsize=13)

        # ── After ───────────────────────────────
        ax2 = axes[1]
        im2 = ax2.imshow(self.thermal_field, cmap=cmap, origin='lower', vmin=25, vmax=50)
        plt.colorbar(im2, ax=ax2, label='°C')
        ax2.set_title('After Irrigation', fontsize=13)

        # Draw drone paths on the after-map
        colors = ['cyan', 'lime', 'magenta', 'yellow', 'white',
                  'orange', 'pink', 'aqua', 'red', 'blue']
        for d in self.drones:
            if len(d.path) > 1:
                xs = [p[0] for p in d.path]
                ys = [p[1] for p in d.path]
                c = colors[d.id % len(colors)]
                ax2.plot(ys, xs, '-', color=c, linewidth=1.5, alpha=0.8,
                         label=f'Drone {d.id}')
                ax2.plot(ys[0], xs[0], 's', color=c, markersize=7)   # start
                ax2.plot(ys[-1], xs[-1], '*', color=c, markersize=10) # end
        ax2.legend(fontsize=7, loc='upper right')

        # ── Efficiency chart ────────────────────
        ax3 = axes[2]
        ticks      = [h['tick'] for h in self.history]
        irr_pct    = [h['irrigated_pct'] for h in self.history]
        mean_temps = [h['mean_temp'] for h in self.history]

        ax3.plot(ticks, irr_pct, 'g-', linewidth=2, label='% Irrigated')
        ax3b = ax3.twinx()
        ax3b.plot(ticks, mean_temps, 'r--', linewidth=2, label='Mean Temp (°C)')
        ax3.set_xlabel('Simulation Tick')
        ax3.set_ylabel('Cells Irrigated (%)', color='green')
        ax3b.set_ylabel('Mean Temperature (°C)', color='red')
        ax3.set_title('Irrigation Progress Over Time', fontsize=13)
        ax3.legend(loc='upper left')
        ax3b.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Results chart saved → {save_path}")
        plt.show()


# ──────────────────────────────────────────────
# MAIN - Run this file directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 2: Drone Swarm Algorithm")
    print("=" * 55)

    # Load thermal field from Day 1
    try:
        thermal_field = np.load("output/thermal_field.npy")
        print("✅ Loaded thermal_field.npy from Day 1")
    except FileNotFoundError:
        print("⚠️  thermal_field.npy not found — generating fresh data...")
        from thermal_simulation import generate_thermal_field
        thermal_field, _ = generate_thermal_field()

    # Create and run the swarm
    swarm = DroneSwarm(thermal_field, num_drones=5, seed=42)
    swarm.run(max_ticks=250, threshold=35.0)

    # Save results for NetLogo
    swarm.save_results("output")

    # Visualize
    swarm.visualize_result(save_path="output/swarm_results.png")

    print("\n✅ Day 2 Complete!")
    print("   Next: Open NetLogo and load drone_irrigation.nlogo")
