# sensor_placement_discrete.py
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist

def greedy_sensor_placement(locs, w, M, target_coverage=0.85):
    """
    Greedy algorithm for sensor placement with explicit coverage target
    """
    N = len(locs)
    J = M.shape[0]
    selected = []
    covered_points = np.zeros(J, dtype=bool)
    
    while True:
        # Find uncovered points
        uncovered = ~covered_points
        if np.sum(uncovered) == 0:
            break
            
        # Calculate marginal coverage for each remaining sensor
        marginal_coverage = np.zeros(N)
        for i in range(N):
            if i in selected:
                continue
            # How many new points would this sensor cover?
            sensor_coverage = M[:, i] > 0  # Convert to boolean
            new_coverage = sensor_coverage & uncovered
            marginal_coverage[i] = np.sum(new_coverage) / w[i] if w[i] > 0 else 0
        
        # Select sensor with best marginal coverage
        best_sensor = np.argmax(marginal_coverage)
        if marginal_coverage[best_sensor] == 0:
            break
            
        selected.append(best_sensor)
        covered_points |= (M[:, best_sensor] > 0)  # Convert to boolean
        
        # Check if we've reached target coverage
        coverage_rate = np.sum(covered_points) / J
        if coverage_rate >= target_coverage:
            break
    
    return selected, np.sum(covered_points) / J

def iterative_improvement(locs, w, M, initial_selected, max_iterations=50):
    """
    Local search to improve sensor placement
    """
    N = len(locs)
    J = M.shape[0]
    selected = set(initial_selected)
    best_score = float('inf')
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try removing each sensor
        for sensor in list(selected):
            temp_selected = selected - {sensor}
            if len(temp_selected) == 0:
                continue
                
            # Calculate coverage without this sensor
            coverage = np.any(M[:, list(temp_selected)] > 0, axis=1) if temp_selected else np.zeros(J, dtype=bool)
            coverage_rate = np.sum(coverage) / J
            score = len(temp_selected) / coverage_rate if coverage_rate > 0 else float('inf')
            
            if score < best_score:
                best_score = score
                selected = temp_selected
                improved = True
                break
        
        # Try adding sensors
        if not improved:
            for sensor in range(N):
                if sensor in selected:
                    continue
                    
                temp_selected = selected | {sensor}
                coverage = np.any(M[:, list(temp_selected)] > 0, axis=1)
                coverage_rate = np.sum(coverage) / J
                score = len(temp_selected) / coverage_rate if coverage_rate > 0 else float('inf')
                
                if score < best_score:
                    best_score = score
                    selected = temp_selected
                    improved = True
                    break
        
        if not improved:
            break
    
    coverage = np.any(M[:, list(selected)] > 0, axis=1) if selected else np.zeros(J, dtype=bool)
    return list(selected), np.sum(coverage) / J

# PARAMETERS
Lx, Ly = 12, 8
r = min(Lx, Ly) * 0.2
nx = int(Lx * 1.5)  # Slightly denser grid
ny = int(Ly * 1.5)
samples = 1000
num_obstacles = np.random.randint(2, 8)  # Fewer obstacles

# 1) ROOM AND OBSTACLES
room = Polygon([(0,0), (Lx,0), (Lx,Ly), (0,Ly)])
obstacles = []
for _ in range(num_obstacles):
    w, h = 0.5 + np.random.rand(), 0.5 + np.random.rand()
    x, y = (Lx - w)*np.random.rand(), (Ly - h)*np.random.rand()
    rect = Polygon([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])
    obstacles.append(rect)
obstacles_union = unary_union(obstacles)
free_space = room.difference(obstacles_union)

# 2) CANDIDATE SENSOR LOCATIONS
xg, yg = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
locs2d = np.vstack([xg.ravel(), yg.ravel()]).T
in_free = np.array([free_space.contains(Point(p)) for p in locs2d])
locs = locs2d[in_free]
N = len(locs)

# 3) COVERAGE WEIGHTS (simplified)
theta = 2 * np.pi * np.random.rand(samples)
rad = r * np.sqrt(np.random.rand(samples))
disk_pts = np.vstack([np.cos(theta)*rad, np.sin(theta)*rad]).T
w = np.zeros(N)
for i, loc in enumerate(locs):
    pts = loc + disk_pts
    w[i] = np.pi*r**2 * np.mean([free_space.contains(Point(p)) for p in pts])

# Normalize weights
w = w / np.max(w) if np.max(w) > 0 else np.ones(N)

# 4) COVERAGE CONSTRAINTS
xg2, yg2 = np.meshgrid(np.linspace(0, Lx, 35), np.linspace(0, Ly, 25))
eval_pts = np.vstack([xg2.ravel(), yg2.ravel()]).T
in_free_eval = np.array([free_space.contains(Point(p)) for p in eval_pts])
eval_pts = eval_pts[in_free_eval]
J = len(eval_pts)
M = np.zeros((J, N))
for j in range(J):
    for i in range(N):
        if np.linalg.norm(eval_pts[j] - locs[i]) <= r:
            M[j,i] = 1


# 5) SOLVE WITH DIFFERENT APPROACHES
print("=== Trying Different Approaches ===")

# Approach 1: Greedy with different coverage targets
coverage_targets = [0.95]
results = []

for target in coverage_targets:
    selected, actual_coverage = greedy_sensor_placement(locs, w, M, target)
    results.append((target, selected, actual_coverage, len(selected)))
    print(f"Target: {target:.0%}, Actual: {actual_coverage:.1%}, Sensors: {len(selected)}")

# Choose best result based on efficiency
best_idx = np.argmax([r[2] / r[3] if r[3] > 0 else 0 for r in results])
_, selected, coverage_rate, num_sensors = results[best_idx]

print(f"\nBest solution: {coverage_rate:.1%} coverage with {num_sensors} sensors")

# Approach 2: Try to improve with local search
print("\nTrying local search improvement...")
improved_selected, improved_coverage = iterative_improvement(locs, w, M, selected)
print(f"After improvement: {improved_coverage:.1%} coverage with {len(improved_selected)} sensors")

# Use improved result if better
if len(improved_selected) <= len(selected) and improved_coverage >= coverage_rate * 0.95:
    selected = improved_selected
    coverage_rate = improved_coverage
    print("Using improved solution")

# 6) FINAL EVALUATION
test_xy = np.random.rand(10000, 2) * [Lx, Ly]
in_free = np.array([free_space.contains(Point(p)) for p in test_xy])
in_coverage = np.zeros(10000, dtype=bool)
for i in selected:
    dist = np.linalg.norm(test_xy - locs[i], axis=1)
    in_coverage |= (dist <= r)
actual_coverage = np.sum(in_free & in_coverage) / np.sum(in_free)

print(f"\nFinal Results:")
print(f"Grid coverage: {coverage_rate:.1%}")
print(f"Monte Carlo coverage: {actual_coverage:.1%}")
print(f"Sensors used: {len(selected)} / {N}")
print(f"Efficiency: {actual_coverage / (len(selected) / N):.2f}")

# 7) VISUALIZATION
import matplotlib.patches as patches

plt.figure(figsize=(12, 8))

# Plot free space
x, y = room.exterior.xy
plt.fill(x, y, color='lightgray', label='Free space')

# Plot obstacles
for obs in obstacles:
    x, y = obs.exterior.xy
    plt.fill(x, y, color='dimgray', label='Obstacle')

# Plot evaluation points
plt.scatter(*eval_pts.T, s=5, color='lightblue', alpha=0.5, label='Eval points')

# Plot candidate sensor locations
plt.scatter(*locs.T, s=15, color='black', alpha=0.3, label='Candidates')

# Plot selected sensors
plt.scatter(*locs[selected].T, s=120, color='red', label='Selected', edgecolor='black')

# Plot sensor coverage
ax = plt.gca()
for idx in selected:
    circ = patches.Circle(locs[idx], r, facecolor='red', edgecolor='black', alpha=0.2, linewidth=1)
    ax.add_patch(circ)

plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.title(f"Discrete Sensor Placement | Coverage: {actual_coverage:.1%} | Sensors: {len(selected)}/{N} | Efficiency: {actual_coverage/(len(selected)/N):.2f}")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
