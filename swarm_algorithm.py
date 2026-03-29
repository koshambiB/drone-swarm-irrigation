"""
swarm_algorithm.py  —  FINAL VERSION with Direct Communication
Each member drone reports its hottest local cell to the leader.
Leader assigns unique targets back to each member.
No two drones chase the same hotspot.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


class Drone:
    def __init__(self, drone_id, x, y, sense_radius=15, speed=0.8, water_capacity=300):
        self.id              = drone_id
        self.x               = float(x)
        self.y               = float(y)
        self.sense_radius    = sense_radius
        self.speed           = speed
        self.water_capacity  = water_capacity
        self.water_used      = 0
        self.path            = [(x, y)]
        self.status          = "searching"
        self.is_leader       = False
        self.assigned_target = None   # set by leader
        self.last_report     = None   # (tx, ty, temp) sent to leader this tick

    def sense_environment(self, thermal_field, irrigated_map):
        """Scan local radius, skip irrigated cells, return hottest stressed cell."""
        grid_h, grid_w = thermal_field.shape
        best_x, best_y, best_temp = None, None, -999
        cx, cy = int(self.x), int(self.y)

        for dx in range(-self.sense_radius, self.sense_radius + 1):
            for dy in range(-self.sense_radius, self.sense_radius + 1):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < grid_h and 0 <= ny < grid_w):
                    continue
                if irrigated_map[nx, ny]:
                    continue
                temp = thermal_field[nx, ny]
                if temp >= 35.0 and temp > best_temp:
                    best_temp = temp
                    best_x, best_y = nx, ny

        return (best_x, best_y, best_temp)

    def report_to_leader(self, thermal_field, irrigated_map):
        """
        COMMUNICATION STEP 1:
        Member drone senses its local area and returns a report
        (tx, ty, temp) representing the hottest cell it found.
        Leader collects these reports each tick.
        """
        tx, ty, temp = self.sense_environment(thermal_field, irrigated_map)
        if tx is not None:
            self.last_report = (tx, ty, temp)
            return (tx, ty, temp)
        self.last_report = None
        return None

    def move_toward(self, target_x, target_y, thermal_field):
        """Move toward target without overshooting."""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.3:
            step = min(self.speed, dist)
            self.x += (dx / dist) * step
            self.y += (dy / dist) * step
            grid_h, grid_w = thermal_field.shape
            self.x = np.clip(self.x, 0, grid_h - 1)
            self.y = np.clip(self.y, 0, grid_w - 1)
            self.path.append((self.x, self.y))

    def irrigate(self, thermal_field, irrigated_map, threshold=35.0, cooling=8.0):
        """Cool down current cell if stressed and un-irrigated."""
        xi = int(round(self.x))
        yi = int(round(self.y))
        grid_h, grid_w = thermal_field.shape
        xi = np.clip(xi, 0, grid_h - 1)
        yi = np.clip(yi, 0, grid_w - 1)

        if (not irrigated_map[xi, yi]) and thermal_field[xi, yi] >= threshold:
            thermal_field[xi, yi] -= cooling
            thermal_field[xi, yi]  = max(thermal_field[xi, yi], 25.0)
            irrigated_map[xi, yi]  = True
            self.water_used       += 1
            self.status            = "irrigating"
            return True

        self.status = "searching"
        return False

    def is_out_of_water(self):
        return self.water_used >= self.water_capacity

    @property
    def position(self):
        return (self.x, self.y)


class DroneSwarm:
    def __init__(self, thermal_field, num_drones=5, seed=42):
        self.thermal_field  = thermal_field.copy()
        self.original_field = thermal_field.copy()
        self.grid_h, self.grid_w = thermal_field.shape
        self.irrigated_map  = np.zeros_like(thermal_field, dtype=bool)
        self.ticks          = 0
        self.history        = []
        self.failover_log   = []
        self.comm_log       = []   # communication events log

        np.random.seed(seed)
        spawn_positions = self._compute_spread_positions(num_drones)

        self.drones = []
        for i in range(num_drones):
            x, y = spawn_positions[i]
            drone = Drone(drone_id=i, x=x, y=y,
                          sense_radius=15, speed=0.8, water_capacity=300)
            self.drones.append(drone)

        self.drones[0].is_leader = True
        print(f"Swarm initialized: {num_drones} drones on {self.grid_h}x{self.grid_w} field")
        print(f"Drone 0 is the initial leader")

    def _compute_spread_positions(self, num_drones):
        positions = []
        cols = int(np.ceil(np.sqrt(num_drones)))
        rows = int(np.ceil(num_drones / cols))
        cell_h = self.grid_h / rows
        cell_w = self.grid_w / cols
        for i in range(num_drones):
            row = i // cols
            col = i % cols
            x = int((row + 0.5) * cell_h + np.random.uniform(-2, 2))
            y = int((col + 0.5) * cell_w + np.random.uniform(-2, 2))
            x = np.clip(x, 2, self.grid_h - 3)
            y = np.clip(y, 2, self.grid_w - 3)
            positions.append((x, y))
        return positions

    def _global_hottest_unirrigated(self):
        mask = (~self.irrigated_map) & (self.thermal_field >= 35.0)
        if not mask.any():
            return None, None
        masked = np.where(mask, self.thermal_field, -999)
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        return idx[0], idx[1]

    def leader_collect_and_assign(self):
        """
        COMMUNICATION STEP 2:
        Leader collects reports from all member drones.
        Sorts by temperature descending.
        Assigns each member a unique target (no overlap).
        This is the core direct communication mechanism.
        """
        leader = next((d for d in self.drones if d.is_leader), None)
        if leader is None:
            return

        # Collect reports from all members
        reports = []
        for drone in self.drones:
            if drone.is_leader or drone.is_out_of_water():
                continue
            report = drone.report_to_leader(self.thermal_field, self.irrigated_map)
            if report is not None:
                tx, ty, temp = report
                reports.append((temp, tx, ty, drone))
                self.comm_log.append({
                    'tick'     : self.ticks,
                    'event'    : 'report',
                    'from'     : drone.id,
                    'to'       : leader.id,
                    'cell'     : (tx, ty),
                    'temp'     : temp
                })

        # Sort hottest first
        reports.sort(key=lambda r: r[0], reverse=True)

        # Assign unique targets — no two drones get same cell
        assigned_cells = set()
        for temp, tx, ty, drone in reports:
            if (tx, ty) not in assigned_cells:
                drone.assigned_target = (tx, ty)
                assigned_cells.add((tx, ty))
                self.comm_log.append({
                    'tick'  : self.ticks,
                    'event' : 'assignment',
                    'from'  : leader.id,
                    'to'    : drone.id,
                    'cell'  : (tx, ty),
                    'temp'  : temp
                })
            else:
                # Find next best unassigned cell for this drone
                drone.assigned_target = None

    def check_leader_failover(self):
        leader = next((d for d in self.drones if d.is_leader), None)
        if leader is None:
            candidates = [d for d in self.drones if not d.is_out_of_water()]
            if candidates:
                new_leader = min(candidates, key=lambda d: d.water_used)
                new_leader.is_leader = True
                self.failover_log.append(f"Tick {self.ticks}: Drone {new_leader.id} auto-promoted")
            return

        if not leader.is_out_of_water():
            return

        leader.is_leader = False
        leader.status = "done"
        self.failover_log.append(f"Tick {self.ticks}: Leader Drone {leader.id} exhausted")

        candidates = [d for d in self.drones if not d.is_out_of_water()]
        if not candidates:
            self.failover_log.append(f"Tick {self.ticks}: All drones exhausted")
            return

        new_leader = min(candidates, key=lambda d: d.water_used)
        new_leader.is_leader = True
        print(f"Tick {self.ticks}: Drone {new_leader.id} promoted to leader")
        self.failover_log.append(f"Tick {self.ticks}: Drone {new_leader.id} promoted")

    def apply_collision_avoidance(self):
        MIN_DIST = 4.0
        for i, d1 in enumerate(self.drones):
            for j, d2 in enumerate(self.drones):
                if i >= j:
                    continue
                dx = d1.x - d2.x
                dy = d1.y - d2.y
                dist = np.sqrt(dx**2 + dy**2)
                if 0 < dist < MIN_DIST:
                    push = (MIN_DIST - dist) / 2
                    d1.x += (dx / dist) * push
                    d1.y += (dy / dist) * push
                    d2.x -= (dx / dist) * push
                    d2.y -= (dy / dist) * push
                    for d in (d1, d2):
                        d.x = np.clip(d.x, 0, self.grid_h - 1)
                        d.y = np.clip(d.y, 0, self.grid_w - 1)

    def step(self):
        self.ticks += 1

        # COMMUNICATION: leader collects reports and assigns targets
        self.leader_collect_and_assign()

        for drone in self.drones:
            if drone.is_out_of_water():
                drone.status = "done"
                continue

            if drone.is_leader:
                tx, ty = self._global_hottest_unirrigated()
                if tx is None:
                    drone.status = "done"
                    continue
            else:
                # Use leader-assigned target
                if drone.assigned_target is not None:
                    tx, ty = drone.assigned_target
                else:
                    # Fallback to global if no assignment
                    tx, ty = self._global_hottest_unirrigated()
                    if tx is None:
                        drone.status = "done"
                        continue

            drone.move_toward(tx, ty, self.thermal_field)
            drone.irrigate(self.thermal_field, self.irrigated_map, threshold=35.0)

        self.apply_collision_avoidance()
        self.check_leader_failover()

        current_leader = next((d for d in self.drones if d.is_leader), None)
        self.history.append({
            'tick'          : self.ticks,
            'irrigated_pct' : self.irrigated_map.mean() * 100,
            'mean_temp'     : self.thermal_field.mean(),
            'leader_id'     : current_leader.id if current_leader else -1,
        })

    def run(self, max_ticks=500, threshold=35.0):
        stressed_init = int((self.thermal_field >= threshold).sum())
        print(f"\nRunning swarm simulation (max {max_ticks} ticks)...")
        print(f"Threshold: {threshold}C | Initial stressed: {stressed_init}\n")

        for t in range(max_ticks):
            self.step()
            if t % 25 == 0:
                stressed = int((self.thermal_field >= threshold).sum())
                pct = self.irrigated_map.mean() * 100
                leader = next((d for d in self.drones if d.is_leader), None)
                print(f"Tick {t:3d}: {stressed:3d} stressed | {pct:5.1f}% irrigated | "
                      f"Leader=Drone {leader.id if leader else 'None'}")
            if (self.thermal_field >= threshold).sum() == 0:
                print(f"\nAll stressed zones irrigated at tick {t}!")
                break

        print(f"\nSimulation complete ({self.ticks} ticks)")
        self.print_final_report(threshold)

    def print_final_report(self, threshold=35.0):
        total_irrigated = int(self.irrigated_map.sum())
        stressed_orig   = int((self.original_field >= threshold).sum())
        total_water     = sum(d.water_used for d in self.drones)
        print(f"\n{'='*50}")
        print(f"  FINAL REPORT")
        print(f"{'='*50}")
        print(f"  Ticks          : {self.ticks}")
        print(f"  Stressed orig  : {stressed_orig}")
        print(f"  Irrigated      : {total_irrigated} ({total_irrigated/max(stressed_orig,1)*100:.1f}%)")
        print(f"  Water used     : {total_water}")
        print(f"  Temp reduction : {self.original_field.mean()-self.thermal_field.mean():.2f}C")
        print(f"  Comm events    : {len(self.comm_log)}")
        print(f"{'='*50}")
        for d in self.drones:
            role = "LEADER" if d.is_leader else "member"
            print(f"  Drone {d.id} [{role}]: water={d.water_used}, status={d.status}")
        if self.failover_log:
            print(f"\n  Failover Events:")
            for e in self.failover_log:
                print(f"    -> {e}")
        print(f"{'='*50}")

    def save_results(self, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)

        np.savetxt(f"{output_dir}/thermal_final.csv",
                   self.thermal_field, delimiter=",", fmt="%.2f")
        np.savetxt(f"{output_dir}/irrigated_map.csv",
                   self.irrigated_map.astype(int), delimiter=",", fmt="%d")

        all_paths = []
        for d in self.drones:
            for step_num, (px, py) in enumerate(d.path):
                all_paths.append([d.id, step_num, px, py])
        np.savetxt(f"{output_dir}/drone_paths.csv", np.array(all_paths),
                   delimiter=",", fmt="%.2f",
                   header="drone_id,step,x,y", comments="")

        ticks      = [h['tick'] for h in self.history]
        irr_pct    = [h['irrigated_pct'] for h in self.history]
        mean_temps = [h['mean_temp'] for h in self.history]
        leader_ids = [h['leader_id'] for h in self.history]
        np.savetxt(f"{output_dir}/simulation_history.csv",
                   np.column_stack([ticks, irr_pct, mean_temps, leader_ids]),
                   delimiter=",", fmt="%.4f",
                   header="tick,irrigated_pct,mean_temp,leader_id", comments="")

        # Save communication log
        if self.comm_log:
            with open(f"{output_dir}/comm_log.csv", "w") as f:
                f.write("tick,event,from,to,cell_x,cell_y,temp\n")
                for e in self.comm_log:
                    f.write(f"{e['tick']},{e['event']},{e['from']},{e['to']},"
                            f"{e['cell'][0]},{e['cell'][1]},{e['temp']:.2f}\n")

        if self.failover_log:
            with open(f"{output_dir}/failover_log.txt", "w") as f:
                f.write("LEADER FAILOVER LOG\n" + "="*40 + "\n")
                for entry in self.failover_log:
                    f.write(entry + "\n")

        print(f"All results saved to '{output_dir}/'")

    def visualize_result(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Swarm Irrigation — Direct Communication Version",
                     fontsize=14, fontweight='bold')
        cmap = plt.cm.RdYlBu_r

        axes[0].imshow(self.original_field, cmap=cmap, origin='lower', vmin=25, vmax=50)
        axes[0].set_title('Before Irrigation')

        axes[1].imshow(self.thermal_field, cmap=cmap, origin='lower', vmin=25, vmax=50)
        axes[1].set_title('After Irrigation + Paths')
        colors = ['cyan','lime','magenta','yellow','white','orange','pink']
        for d in self.drones:
            if len(d.path) > 1:
                xs = [p[0] for p in d.path]
                ys = [p[1] for p in d.path]
                c  = colors[d.id % len(colors)]
                axes[1].plot(ys, xs, '-', color=c, linewidth=1.5, alpha=0.8,
                             label=f'D{d.id}{"(L)" if d.is_leader else ""}')
                axes[1].plot(ys[0], xs[0], 's', color=c, markersize=6)
                axes[1].plot(ys[-1], xs[-1], '*', color=c, markersize=10)
        axes[1].legend(fontsize=7)

        ticks   = [h['tick'] for h in self.history]
        irr_pct = [h['irrigated_pct'] for h in self.history]
        temps   = [h['mean_temp'] for h in self.history]
        axes[2].plot(ticks, irr_pct, 'g-', linewidth=2, label='% Irrigated')
        ax2 = axes[2].twinx()
        ax2.plot(ticks, temps, 'r--', linewidth=2, label='Mean Temp')
        axes[2].set_title('Progress')
        axes[2].set_xlabel('Tick')
        axes[2].legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved -> {save_path}")
        plt.show()


if __name__ == "__main__":
    print("="*50)
    print("  Drone Swarm — Direct Communication Version")
    print("="*50)

    try:
        thermal_field = np.load("output/thermal_field.npy")
        print("Loaded thermal_field.npy")
    except FileNotFoundError:
        print("thermal_field.npy not found — generating...")
        from thermal_simulation import generate_thermal_field
        thermal_field, _ = generate_thermal_field()

    swarm = DroneSwarm(thermal_field, num_drones=5, seed=42)
    swarm.run(max_ticks=500, threshold=35.0)
    swarm.save_results("output")
    swarm.visualize_result(save_path="output/swarm_results.png")
    print("\nDone!")