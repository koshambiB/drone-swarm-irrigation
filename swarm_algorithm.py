"""
swarm_algorithm.py
DAY 2 - Drone Swarm Logic with Leader Failover
Each drone finds hot zones and moves toward them.
If the leader drone fails, the next best drone is promoted automatically.
Run AFTER thermal_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
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
    - Can be the leader (targets globally hottest zone)
      or a member drone (targets locally hottest zone)
    """

    def __init__(self, drone_id, x, y, sense_radius=7, speed=1.2, water_capacity=150):
        self.id             = drone_id
        self.x              = float(x)
        self.y              = float(y)
        self.sense_radius   = sense_radius    # How far the drone can "see"
        self.speed          = speed           # How fast it moves per tick
        self.water_capacity = water_capacity  # Total water units it carries
        self.water_used     = 0              # Water used so far
        self.path           = [(x, y)]       # Record of positions visited
        self.status         = "searching"    # searching / irrigating / done
        self.is_leader      = False          # Leader flag

    def sense_environment(self, thermal_field):
        """
        Look at nearby cells and return the hottest one.
        Returns: (best_x, best_y, max_temp)
        """
        grid_h, grid_w = thermal_field.shape
        best_x, best_y, best_temp = None, None, -999

        for dx in range(-self.sense_radius, self.sense_radius + 1):
            for dy in range(-self.sense_radius, self.sense_radius + 1):
                nx = int(self.x) + dx
                ny = int(self.y) + dy

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

        if dist > 0.5:
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
    1. Drone 0 starts as the LEADER — targets the globally hottest cell
    2. Member drones target the hottest cell in their local sense radius
    3. If the leader runs out of water (fails), the drone with the most
       water remaining is immediately promoted as the new leader
    4. Simple collision avoidance keeps drones spread out
    """

    def __init__(self, thermal_field, num_drones=5, seed=42):
        self.thermal_field  = thermal_field.copy()
        self.original_field = thermal_field.copy()
        self.grid_h, self.grid_w = thermal_field.shape
        self.irrigated_map  = np.zeros_like(thermal_field, dtype=bool)
        self.ticks          = 0
        self.history        = []
        self.failover_log   = []   # records all leader failover events

        np.random.seed(seed)
        self.drones = []
        for i in range(num_drones):
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

        # Designate first drone as leader
        self.drones[0].is_leader = True
        print(f"🚁 Swarm initialized: {num_drones} drones on {self.grid_h}×{self.grid_w} field")
        print(f"👑 Drone 0 is the initial leader")

    # ─────────────────────────────────────────────
    # EDGE CASE HANDLER: Leader Failover
    # ─────────────────────────────────────────────
    def check_leader_failover(self):
        """
        EDGE CASE: If the leader drone runs out of water (fails),
        immediately promote the drone with the most water remaining.

        Steps:
        1. Find the current leader
        2. Check if it has failed (out of water)
        3. Strip its leader status
        4. Find all drones still with water (candidates)
        5. Promote the one with the least water_used (most remaining)
        """
        # Find current leader
        leader = next((d for d in self.drones if d.is_leader), None)

        # Edge sub-case: no leader exists at all (e.g. very start)
        if leader is None:
            candidates = [d for d in self.drones if not d.is_out_of_water()]
            if candidates:
                new_leader = min(candidates, key=lambda d: d.water_used)
                new_leader.is_leader = True
                msg = (f"Tick {self.ticks}: No leader found — "
                       f"Drone {new_leader.id} auto-promoted")
                print(f"👑 {msg}")
                self.failover_log.append(msg)
            return

        # Leader is still operational — nothing to do
        if not leader.is_out_of_water():
            return

        # ── Leader has FAILED ─────────────────────
        leader.is_leader = False
        leader.status    = "done"

        fail_msg = (f"Tick {self.ticks}: Leader Drone {leader.id} "
                    f"ran out of water and FAILED")
        print(f"\n⚠️  {fail_msg}")
        self.failover_log.append(fail_msg)

        # Find candidates — drones still with water left
        candidates = [d for d in self.drones if not d.is_out_of_water()]

        if not candidates:
            no_drones_msg = f"Tick {self.ticks}: All drones failed — swarm is done"
            print(f"❌ {no_drones_msg}")
            self.failover_log.append(no_drones_msg)
            return

        # Promote the drone with the most water remaining
        new_leader = min(candidates, key=lambda d: d.water_used)
        new_leader.is_leader = True

        water_left = new_leader.water_capacity - new_leader.water_used
        promo_msg  = (f"Tick {self.ticks}: Drone {new_leader.id} promoted to leader "
                      f"(water remaining: {water_left} units)")
        print(f"👑 {promo_msg}\n")
        self.failover_log.append(promo_msg)

    # ─────────────────────────────────────────────
    # Collision avoidance
    # ─────────────────────────────────────────────
    def apply_collision_avoidance(self):
        """Push drones apart if they get too close."""
        MIN_DIST = 3.0
        for i, d1 in enumerate(self.drones):
            for j, d2 in enumerate(self.drones):
                if i >= j:
                    continue
                dx   = d1.x - d2.x
                dy   = d1.y - d2.y
                dist = np.sqrt(dx**2 + dy**2)

                if dist < MIN_DIST and dist > 0:
                    push = (MIN_DIST - dist) / 2
                    d1.x += (dx / dist) * push
                    d1.y += (dy / dist) * push
                    d2.x -= (dx / dist) * push
                    d2.y -= (dy / dist) * push

                    d1.x = np.clip(d1.x, 0, self.grid_h - 1)
                    d1.y = np.clip(d1.y, 0, self.grid_w - 1)
                    d2.x = np.clip(d2.x, 0, self.grid_h - 1)
                    d2.y = np.clip(d2.y, 0, self.grid_w - 1)

    # ─────────────────────────────────────────────
    # Main simulation step
    # ─────────────────────────────────────────────
    def step(self):
        """Advance the simulation by one time-step (tick)."""
        self.ticks += 1

        for drone in self.drones:
            if drone.is_out_of_water():
                drone.status = "done"
                continue

            # ── Leader behaviour ──────────────────
            if drone.is_leader:
                # Leader scans the ENTIRE field for the hottest unirrigated cell
                mask = ~self.irrigated_map
                if mask.any():
                    masked_field = np.where(mask, self.thermal_field, -999)
                    hot_idx      = np.unravel_index(
                        np.argmax(masked_field), masked_field.shape
                    )
                    tx, ty = hot_idx
                else:
                    # All irrigated — fall back to local sensing
                    tx, ty, _ = drone.sense_environment(self.thermal_field)
                    if tx is None:
                        continue

            # ── Member drone behaviour ────────────
            else:
                # Member drones only sense within their local radius
                tx, ty, _ = drone.sense_environment(self.thermal_field)
                if tx is None:
                    continue

            drone.move_toward(tx, ty)
            drone.irrigate(self.thermal_field, self.irrigated_map)

        # Collision avoidance
        self.apply_collision_avoidance()

        # Check for leader failover (the edge case)
        self.check_leader_failover()

        # Record snapshot for charts
        current_leader = next((d for d in self.drones if d.is_leader), None)
        self.history.append({
            'tick'           : self.ticks,
            'irrigated_pct'  : self.irrigated_map.mean() * 100,
            'mean_temp'      : self.thermal_field.mean(),
            'leader_id'      : current_leader.id if current_leader else -1,
            'drone_positions': [(d.x, d.y) for d in self.drones],
        })

    # ─────────────────────────────────────────────
    # Run loop
    # ─────────────────────────────────────────────
    def run(self, max_ticks=250, threshold=35.0):
        """Run the full simulation."""
        print(f"\n🏃 Running swarm simulation (max {max_ticks} ticks)...")
        print(f"   Irrigation threshold  : {threshold}°C")
        print(f"   Initial stressed cells: {(self.thermal_field >= threshold).sum()}")
        print()

        for t in range(max_ticks):
            self.step()

            if t % 20 == 0:
                stressed      = (self.thermal_field >= threshold).sum()
                irrigated_pct = self.irrigated_map.mean() * 100
                leader        = next((d for d in self.drones if d.is_leader), None)
                leader_str    = f"Leader=Drone {leader.id}" if leader else "No leader"
                print(f"   Tick {t:3d}: {stressed:3d} stressed | "
                      f"{irrigated_pct:.1f}% irrigated | "
                      f"Temp: {self.thermal_field.mean():.1f}°C | "
                      f"{leader_str}")

            if (self.thermal_field >= threshold).sum() == 0:
                print(f"\n✅ All stressed zones irrigated at tick {t}!")
                break

        print(f"\n📊 Simulation complete ({self.ticks} ticks)")
        self.print_final_report()

    # ─────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────
    def print_final_report(self):
        total_irrigated = self.irrigated_map.sum()
        total_cells     = self.grid_h * self.grid_w
        total_water     = sum(d.water_used for d in self.drones)

        print(f"\n{'='*55}")
        print(f"  FINAL SWARM REPORT")
        print(f"{'='*55}")
        print(f"  Total ticks         : {self.ticks}")
        print(f"  Cells irrigated     : {total_irrigated}/{total_cells} "
              f"({total_irrigated/total_cells*100:.1f}%)")
        print(f"  Total water used    : {total_water} units")
        print(f"  Mean field temp     : {self.thermal_field.mean():.1f}°C")
        print(f"  Remaining stress    : {(self.thermal_field >= 35).sum()} cells >=35°C")
        print(f"{'='*55}")

        for d in self.drones:
            role = "LEADER" if d.is_leader else "member"
            print(f"  Drone {d.id} [{role}]: "
                  f"water_used={d.water_used}, "
                  f"path_len={len(d.path)}, "
                  f"status={d.status}")

        if self.failover_log:
            print(f"\n  Leader Failover Events ({len(self.failover_log)}):")
            for entry in self.failover_log:
                print(f"    -> {entry}")
        else:
            print(f"\n  No leader failover events occurred.")
        print(f"{'='*55}")

    # ─────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────
    def save_results(self, output_dir="output"):
        """Save all results needed by NetLogo and main_demo.py"""
        os.makedirs(output_dir, exist_ok=True)

        np.savetxt(f"{output_dir}/thermal_final.csv",
                   self.thermal_field, delimiter=",", fmt="%.2f")

        np.savetxt(f"{output_dir}/irrigated_map.csv",
                   self.irrigated_map.astype(int), delimiter=",", fmt="%d")

        all_paths = []
        for d in self.drones:
            for step_num, (px, py) in enumerate(d.path):
                all_paths.append([d.id, step_num, px, py])

        path_arr = np.array(all_paths)
        np.savetxt(f"{output_dir}/drone_paths.csv", path_arr,
                   delimiter=",", fmt="%.2f",
                   header="drone_id,step,x,y", comments="")

        ticks      = [h['tick'] for h in self.history]
        irr_pct    = [h['irrigated_pct'] for h in self.history]
        mean_temps = [h['mean_temp'] for h in self.history]
        leader_ids = [h['leader_id'] for h in self.history]

        history_arr = np.column_stack([ticks, irr_pct, mean_temps, leader_ids])
        np.savetxt(f"{output_dir}/simulation_history.csv", history_arr,
                   delimiter=",", fmt="%.4f",
                   header="tick,irrigated_pct,mean_temp,leader_id", comments="")

        # Save failover log as readable text file
        if self.failover_log:
            with open(f"{output_dir}/failover_log.txt", "w") as f:
                f.write("LEADER FAILOVER LOG\n")
                f.write("=" * 40 + "\n")
                for entry in self.failover_log:
                    f.write(entry + "\n")
            print(f" Failover log saved -> {output_dir}/failover_log.txt")

        print(f" All results saved to '{output_dir}/'")

    # ─────────────────────────────────────────────
    # Visualize
    # ─────────────────────────────────────────────
    def visualize_result(self, save_path=None):
        """Show before/after comparison + drone paths + leader change markers."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Swarm Irrigation Simulation Results (with Leader Failover)",
                     fontsize=14, fontweight='bold')

        cmap = plt.cm.RdYlBu_r

        # ── Before ──────────────────────────────
        ax1 = axes[0]
        im1 = ax1.imshow(self.original_field, cmap=cmap, origin='lower', vmin=25, vmax=50)
        plt.colorbar(im1, ax=ax1, label='°C')
        ax1.set_title('Before Irrigation', fontsize=12)

        # ── After + drone paths ──────────────────
        ax2 = axes[1]
        im2 = ax2.imshow(self.thermal_field, cmap=cmap, origin='lower', vmin=25, vmax=50)
        plt.colorbar(im2, ax=ax2, label='°C')
        ax2.set_title('After Irrigation + Drone Paths', fontsize=12)

        path_colors = ['cyan', 'lime', 'magenta', 'yellow', 'white',
                       'orange', 'pink', 'aqua', 'red', 'blue']

        for d in self.drones:
            if len(d.path) > 1:
                xs = [p[0] for p in d.path]
                ys = [p[1] for p in d.path]
                c  = path_colors[d.id % len(path_colors)]
                lw = 2.5 if d.is_leader else 1.2
                label = f'Drone {d.id} (leader)' if d.is_leader else f'Drone {d.id}'
                ax2.plot(ys, xs, '-', color=c, linewidth=lw, alpha=0.85, label=label)
                ax2.plot(ys[0],  xs[0],  's', color=c, markersize=7)
                ax2.plot(ys[-1], xs[-1], '*', color=c, markersize=11)

        ax2.legend(fontsize=7, loc='upper right')

        # ── Progress + leader change markers ─────
        ax3 = axes[2]
        ticks      = [h['tick'] for h in self.history]
        irr_pct    = [h['irrigated_pct'] for h in self.history]
        mean_temps = [h['mean_temp'] for h in self.history]
        leader_ids = [h['leader_id'] for h in self.history]

        ax3.plot(ticks, irr_pct, 'g-', linewidth=2, label='% Irrigated')
        ax3b = ax3.twinx()
        ax3b.plot(ticks, mean_temps, 'r--', linewidth=2, label='Mean Temp (C)')

        # Draw vertical orange lines where the leader changed
        prev = leader_ids[0] if leader_ids else -1
        for i, lid in enumerate(leader_ids):
            if lid != prev:
                ax3.axvline(x=ticks[i], color='orange',
                            linestyle='--', linewidth=1.5, alpha=0.9)
                ax3.text(ticks[i] + 1, max(irr_pct) * 0.4,
                         f'->D{lid}', fontsize=8, color='orange')
                prev = lid

        ax3.set_xlabel('Simulation Tick')
        ax3.set_ylabel('Cells Irrigated (%)', color='green')
        ax3b.set_ylabel('Mean Temperature (C)', color='red')
        ax3.set_title('Progress (orange = leader change)', fontsize=11)
        ax3.legend(loc='upper left')
        ax3b.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Results chart saved -> {save_path}")
        plt.show()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 2: Drone Swarm Algorithm (with Leader Failover)")
    print("=" * 55)

    # Load thermal field from Day 1
    try:
        thermal_field = np.load("output/thermal_field.npy")
        print("Loaded thermal_field.npy from Day 1")
    except FileNotFoundError:
        print("Warning: thermal_field.npy not found — generating fresh data...")
        from thermal_simulation import generate_thermal_field
        thermal_field, _ = generate_thermal_field()

    # Create and run the swarm
    swarm = DroneSwarm(thermal_field, num_drones=5, seed=42)
    swarm.run(max_ticks=250, threshold=35.0)

    # Save results for NetLogo
    swarm.save_results("output")

    # Visualize
    swarm.visualize_result(save_path="output/swarm_results.png")

    print("\n Day 2 Complete!")
    print("   Next: Open NetLogo and load drone_irrigation.nlogo")