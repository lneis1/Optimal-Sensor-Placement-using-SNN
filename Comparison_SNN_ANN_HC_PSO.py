import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp
import matplotlib.patches as patches
import time
import random

# --- SNN Solver (unchanged) ---
def snn_solver(A, b, C, d, t_end, x0, k0=0.09, k1=0.05):
    def constraint_violation(x_vec):
        return C @ x_vec + d

    def grad(x_vec):
        g = A @ x_vec + b
        norm_g = np.linalg.norm(g)
        return g if norm_g < 100 else g * 100 / norm_g

    def snn_dynamics(t, x_vec):
        return -k0 * grad(x_vec)

    t_current = 0.0
    t_vals = []
    X_vals = []
    x_current = np.copy(x0)

    x_current = np.clip(x_current, 0, 1)
    initial_violated_d = constraint_violation(x_current)
    violated_indices = initial_violated_d > 1e-6
    if np.any(violated_indices):
        x_current -= k1 * C[violated_indices, :].T @ initial_violated_d[violated_indices]
        x_current = np.clip(x_current, 0, 1)

    while t_current < t_end:
        violated_d = constraint_violation(x_current)
        violated_indices = violated_d > 1e-6
        if np.any(violated_indices):
            x_current -= k1 * C[violated_indices, :].T @ violated_d[violated_indices]
            x_current = np.clip(x_current, 0, 1)

        sol = solve_ivp(snn_dynamics, [t_current, t_end], x_current, max_step=0.05, method='RK45', atol=1e-6, rtol=1e-4)
        if len(sol.t) <= 1 and t_current < t_end:
            break

        t_vals.append(sol.t)
        X_vals.append(sol.y.T)
        t_current = sol.t[-1]
        x_current = sol.y[:, -1]
        x_current = np.clip(x_current, 0, 1)

        if np.all(constraint_violation(x_current) <= 1e-3):
            break

    if not t_vals:
        return np.array([t_current]), np.array([x0]).reshape(1, -1)
    t_all = np.concatenate(t_vals)
    X_all = np.vstack(X_vals)
    return t_all, X_all

# --- Helper function for Monte Carlo Coverage Calculation ---
def calculate_coverage(selected_indices, locs, r, free_space, Lx, Ly, num_test_points=10000):
    test_xy = np.random.rand(num_test_points, 2) * [Lx, Ly]
    in_free = np.array([free_space.contains(Point(p)) for p in test_xy])
    in_coverage = np.zeros(num_test_points, dtype=bool)

    if not selected_indices.size: # Check if selected_indices is empty
        return 0.0

    for i in selected_indices:
        dist = np.linalg.norm(test_xy - locs[i], axis=1)
        in_coverage |= (dist <= r)

    total_free = np.sum(in_free)
    covered = np.sum(in_free & in_coverage)
    coverage_rate = (covered / total_free) * 100 if total_free > 0 else 0
    return coverage_rate

# --- Sensor Selection Methods ---

def solve_snn(N_target, A, b, C, d, t_end, x0, k0, k1, M, N):
    # A_new and b_new are defined inside the main loop to ensure they use correct N_target
    # and penalty_N_deviation. C and d are already constructed correctly.
    t_snn, X_snn = snn_solver(A, b, C, d, t_end, x0, k0, k1)
    y_final = X_snn[-1, :]

    top_indices = np.argsort(y_final)[::-1]
    selected_indices = top_indices[:N_target]
    num_selected = len(selected_indices)
    return selected_indices, num_selected

# --- New Method 1: ANN (Simplified Scoring) ---
def solve_ann_scoring(N_target, M, N, w, P, penalty_weight=1):
    if N == 0:
        return np.array([]), 0
    if N_target > N: N_target = N

    # Calculate a score for each potential sensor location
    # Score = (individual coverage potential) - (sum of penalties with all other sensors)
    # This mimics a simple neuron's output based on input (w) and recurrent inhibition (P)
    scores = np.zeros(N)
    for i in range(N):
        # Sum of penalties with other sensors
        # We assume other sensors are present for calculating the potential penalty
        # A more complex ANN would learn these weights, but this is a simplified heuristic
        total_penalty_with_others = np.sum(P[i, :]) # Sum of row i in P, which includes P_ii=0

        scores[i] = w[i] - (penalty_weight * total_penalty_with_others)

    # Select the top N_target sensors based on these scores
    top_indices = np.argsort(scores)[::-1] # Sort in descending order
    selected_indices = top_indices[:N_target]
    num_selected = len(selected_indices)
    return selected_indices, num_selected

# --- New Method 2: Hill Climbing ---
def solve_hill_climbing(N_target, M, N, w, P, max_iterations=1000,
                        coverage_weight=100, penalty_weight=1, target_deviation_penalty=50):
    if N == 0:
        return np.array([]), 0
    if N_target > N: N_target = N

    # Objective function for Hill Climbing (maximize this)
    def calculate_objective(chromosome):
        selected_indices = np.where(chromosome == 1)[0]
        num_selected = len(selected_indices)

        if num_selected == 0:
            return -1e9 # Very low objective for no sensors

        # Coverage score (number of covered eval points)
        total_covered_eval_pts = np.zeros(M.shape[0], dtype=bool)
        if selected_indices.size > 0:
            total_covered_eval_pts = np.any(M[:, selected_indices], axis=1)
        coverage_score = np.sum(total_covered_eval_pts)

        # Penalty from P matrix
        penalty_score = 0.0
        if num_selected > 1:
            for i in range(num_selected):
                for j in range(i + 1, num_selected):
                    penalty_score += P[selected_indices[i], selected_indices[j]]

        # Penalty for deviating from N_target (soft constraint)
        deviation_penalty = target_deviation_penalty * abs(num_selected - N_target)

        objective = (coverage_weight * coverage_score) - (penalty_weight * penalty_score) - deviation_penalty
        return objective

    # Generate initial random solution (N_target sensors selected)
    current_chromosome = np.zeros(N, dtype=int)
    if N > 0:
        initial_selection = np.random.choice(N, N_target, replace=False)
        current_chromosome[initial_selection] = 1

    current_objective = calculate_objective(current_chromosome)

    for _ in range(max_iterations):
        # Generate a neighbor by swapping one active sensor with one inactive sensor
        neighbor_chromosome = np.copy(current_chromosome)

        if N_target > 0 and N_target < N:
            active_sensors = np.where(neighbor_chromosome == 1)[0]
            inactive_sensors = np.where(neighbor_chromosome == 0)[0]

            if active_sensors.size > 0 and inactive_sensors.size > 0:
                idx_to_remove = np.random.choice(active_sensors)
                idx_to_add = np.random.choice(inactive_sensors)

                neighbor_chromosome[idx_to_remove] = 0
                neighbor_chromosome[idx_to_add] = 1
            else: # No valid swap possible (e.g., all sensors active or inactive)
                break
        elif N_target == N or N_target == 0: # No swaps possible if all or no sensors are targeted
            break

        neighbor_objective = calculate_objective(neighbor_chromosome)

        # Move to neighbor if it's better
        if neighbor_objective > current_objective:
            current_chromosome = np.copy(neighbor_chromosome)
            current_objective = neighbor_objective
        else:
            # If no improvement, break (simple hill climbing)
            break

    final_selected_indices = np.where(current_chromosome == 1)[0]
    return final_selected_indices, len(final_selected_indices)

# --- New Method 3: Particle Swarm Optimization (PSO) ---
def solve_pso(N_target, M, N, w, P,
              num_particles=30, max_iterations=100,
              c1=2.0, c2=2.0, w_inertia=0.9,
              coverage_weight=100, penalty_weight=1, target_deviation_penalty=50):

    if N == 0:
        return np.array([]), 0
    if N_target > N: N_target = N

    # Objective function for PSO (maximize this)
    def calculate_objective(chromosome_binary):
        selected_indices = np.where(chromosome_binary == 1)[0]
        num_selected = len(selected_indices)

        if num_selected == 0:
            return -1e9 # Very low objective for no sensors

        total_covered_eval_pts = np.zeros(M.shape[0], dtype=bool)
        if selected_indices.size > 0:
            total_covered_eval_pts = np.any(M[:, selected_indices], axis=1)
        coverage_score = np.sum(total_covered_eval_pts)

        penalty_score = 0.0
        if num_selected > 1:
            for i in range(num_selected):
                for j in range(i + 1, num_selected):
                    penalty_score += P[selected_indices[i], selected_indices[j]]

        deviation_penalty = target_deviation_penalty * abs(num_selected - N_target)

        objective = (coverage_weight * coverage_score) - (penalty_weight * penalty_score) - deviation_penalty
        return objective

    # Initialize particles
    particles_pos = np.zeros((num_particles, N), dtype=float) # Continuous positions
    particles_vel = np.zeros((num_particles, N), dtype=float) # Velocities

    # Initialize binary solutions for objective calculation
    particles_binary = np.zeros((num_particles, N), dtype=int)

    for i in range(num_particles):
        # Initialize with N_target randomly selected sensors
        initial_selection = np.random.choice(N, N_target, replace=False)
        particles_binary[i, initial_selection] = 1
        # Initialize continuous position to match binary state (e.g., 0.5 for active, 0 for inactive)
        # Or simply random between 0 and 1
        particles_pos[i] = np.random.rand(N) # Initialize randomly between 0 and 1
        particles_pos[i, initial_selection] = np.random.uniform(0.5, 1.0, size=N_target) # Make selected more likely to be 1

    pbest_pos = np.copy(particles_pos) # Personal best position
    pbest_obj = np.array([calculate_objective(p_bin) for p_bin in particles_binary]) # Personal best objective

    gbest_idx = np.argmax(pbest_obj)
    gbest_pos = np.copy(pbest_pos[gbest_idx]) # Global best position
    gbest_obj = pbest_obj[gbest_idx] # Global best objective

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Update velocity
            r1 = np.random.rand(N)
            r2 = np.random.rand(N)

            # Using particles_binary for pbest_pos in velocity update
            velocity_update = (w_inertia * particles_vel[i]) + \
                              (c1 * r1 * (pbest_pos[i] - particles_pos[i])) + \
                              (c2 * r2 * (gbest_pos - particles_pos[i]))

            particles_vel[i] = velocity_update

            # Update position (continuous)
            particles_pos[i] += particles_vel[i]

            # Clamp positions to [0, 1]
            particles_pos[i] = np.clip(particles_pos[i], 0, 1)

            # Convert continuous position to binary for objective calculation
            # A common approach for binary PSO is to use a sigmoid function
            # and then threshold, or directly threshold
            # Here, we'll simply threshold at 0.5 for simplicity
            new_binary_chromosome = (particles_pos[i] > 0.5).astype(int)

            # Enforce N_target constraint for the binary chromosome
            current_selected = np.sum(new_binary_chromosome)
            if current_selected > N_target:
                active_indices = np.where(new_binary_chromosome == 1)[0]
                to_deselect = np.random.choice(active_indices, current_selected - N_target, replace=False)
                new_binary_chromosome[to_deselect] = 0
            elif current_selected < N_target and N - current_selected > 0:
                inactive_indices = np.where(new_binary_chromosome == 0)[0]
                if inactive_indices.size > 0:
                    to_select = np.random.choice(inactive_indices, N_target - current_selected, replace=False)
                    new_binary_chromosome[to_select] = 1

            particles_binary[i] = new_binary_chromosome

            # Evaluate objective
            current_obj = calculate_objective(particles_binary[i])

            # Update personal best
            if current_obj > pbest_obj[i]:
                pbest_obj[i] = current_obj
                pbest_pos[i] = np.copy(particles_pos[i]) # Store continuous position for pbest

            # Update global best
            if current_obj > gbest_obj:
                gbest_obj = current_obj
                gbest_pos = np.copy(particles_pos[i]) # Store continuous position for gbest

        # If the global best solution is not N_target, ensure it is before returning
        # This is important because gbest_pos is continuous, and its binary conversion might not be exactly N_target
        gbest_final_binary = (gbest_pos > 0.5).astype(int)
        current_selected = np.sum(gbest_final_binary)
        if current_selected > N_target:
            active_indices = np.where(gbest_final_binary == 1)[0]
            to_deselect = np.random.choice(active_indices, current_selected - N_target, replace=False)
            gbest_final_binary[to_deselect] = 0
        elif current_selected < N_target and N - current_selected > 0:
            inactive_indices = np.where(gbest_final_binary == 0)[0]
            if inactive_indices.size > 0:
                to_select = np.random.choice(inactive_indices, N_target - current_selected, replace=False)
                gbest_final_binary[to_select] = 1

    final_selected_indices = np.where(gbest_final_binary == 1)[0]
    return final_selected_indices, len(final_selected_indices)


# --- Parameters ---
Lx, Ly = 12, 8
r = min(Lx, Ly) * 0.2
nx, ny = int(Lx), int(Ly)
k_spring = 20
samples = 10000 # For 'w' calculation
t_end, k0, k1 = 500, 0.5, 10
penalty_N_deviation = 100 # Used in SNN and as a conceptual penalty for other methods' fitness/energy

N_targets = [18, 19, 20, 21]  # Sensor counts to evaluate

num_room_structures = 50 # Reduced for quicker testing, set to 50 for full comparison
num_random_events = 50   # For SNN, ANN, Hill Climbing, PSO (due to stochastic nature)

# Dictionaries to store average coverage rates for each method
average_coverage = {
    'SNN': {n: [] for n in N_targets},
    'ANN (Simplified Scoring)': {n: [] for n in N_targets},
    'Hill Climbing': {n: [] for n in N_targets},
    'Particle Swarm Optimization': {n: [] for n in N_targets}
}

# Store execution times
execution_times = {
    'SNN': {n: [] for n in N_targets},
    'ANN (Simplified Scoring)': {n: [] for n in N_targets},
    'Hill Climbing': {n: [] for n in N_targets},
    'Particle Swarm Optimization': {n: [] for n in N_targets}
}

for room_idx in range(num_room_structures):
    print(f"\n--- Generating Room Structure {room_idx + 1}/{num_room_structures} ---")

    # Generate random room structure
    room = Polygon([(0,0), (Lx,0), (Lx,Ly), (0,Ly)])
    num_obstacles = np.random.randint(2, 11)
    obstacles = []
    for _ in range(num_obstacles):
        w_obs, h_obs = 0.5 + np.random.rand(), 0.5 + np.random.rand()
        x_obs, y_obs = (Lx - w_obs) * np.random.rand(), (Ly - h_obs) * np.random.rand()
        obs = Polygon([(x_obs,y_obs), (x_obs+w_obs,y_obs), (x_obs+w_obs,y_obs+h_obs), (x_obs,y_obs+h_obs)])
        obstacles.append(obs)
    obstacles_union = unary_union(obstacles)
    free_space = room.difference(obstacles_union)
    true_free_area = free_space.area

    xg, yg = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
    locs2d = np.vstack([xg.ravel(), yg.ravel()]).T
    in_free_check = np.array([free_space.contains(Point(p)) for p in locs2d])
    locs = locs2d[in_free_check]
    N = len(locs)

    # If no free space points, skip this room structure
    if N == 0:
        print("No free space points for sensor placement. Skipping this room structure.")
        continue

    # --- Weights (w) ---
    theta = 2 * np.pi * np.random.rand(samples)
    rad = r * np.sqrt(np.random.rand(samples))
    unit_disk_pts = np.vstack([np.cos(theta)*rad, np.sin(theta)*rad]).T
    w = np.array([
        np.pi * r**2 * np.mean([free_space.contains(Point(loc + pt)) for pt in unit_disk_pts])
        for loc in locs
    ])

    # --- Penalty matrix (P) ---
    D = cdist(locs, locs)
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] < 2 * r:
                P[i, j] = k_spring * (2 * r - D[i, j])**2
                P[j, i] = P[i, j]
    A = P # A is used by SNN as the penalty term

    # --- Coverage matrix (M) ---
    xg2, yg2 = np.meshgrid(np.linspace(0, Lx, 40), np.linspace(0, Ly, 24))
    eval_pts = np.vstack([xg2.ravel(), yg2.ravel()]).T
    in_free_eval = np.array([free_space.contains(Point(p)) for p in eval_pts])
    eval_pts = eval_pts[in_free_eval]
    J = len(eval_pts) # Number of evaluation points in free space

    M = np.zeros((J, N)) # M[j,i] = 1 if sensor i covers eval_point j
    if N > 0 and J > 0: # Ensure there are potential sensors and evaluation points
        for j in range(J):
            for i in range(N):
                if np.linalg.norm(eval_pts[j] - locs[i]) <= r:
                    M[j, i] = 1

    # If M has no columns (no sensor locations) or no rows (no eval points), skip
    if M.shape[1] == 0 or M.shape[0] == 0:
        print("No valid sensor locations or evaluation points. Skipping this room structure.")
        continue

    # Iterate for each sensor count (N_target)
    for N_target in N_targets:
        print(f"\n--- Solving for N_target = {N_target} ---")

        # --- SNN Solver ---
        current_snn_coverages = []
        current_snn_times = []
        for event_idx in range(num_random_events):
            start_time = time.time()
            x0_snn = np.random.rand(N) # Random initial guess for SNN solver

            # SNN's A and b incorporate penalty_N_deviation
            A_snn = A + penalty_N_deviation * np.ones((N, N))
            b_snn = -w + penalty_N_deviation * (-2 * N_target * np.ones(N))
            C_snn = np.vstack([-M, np.eye(N), -np.eye(N)]) # Constraints: cover all eval_pts, x in [0,1]
            d_snn = np.concatenate([-np.ones(J), -np.ones(N), np.zeros(N)]) # Constraints: C*x + d <= 0

            selected_snn_indices, num_selected_snn = solve_snn(N_target, A_snn, b_snn, C_snn, d_snn, t_end, x0_snn, k0, k1, M, N)

            coverage_snn = calculate_coverage(selected_snn_indices, locs, r, free_space, Lx, Ly)
            current_snn_coverages.append(coverage_snn)
            current_snn_times.append(time.time() - start_time)
        average_coverage['SNN'][N_target].append(np.mean(current_snn_coverages))
        execution_times['SNN'][N_target].append(np.mean(current_snn_times))
        print(f"  SNN Average Coverage: {np.mean(current_snn_coverages):.2f}% (Avg Time: {np.mean(current_snn_times):.4f}s)")

        # --- ANN (Simplified Scoring) ---
        current_ann_coverages = []
        current_ann_times = []
        for event_idx in range(num_random_events): # This method is deterministic, but run multiple times for consistency in comparison loop
            start_time = time.time()
            selected_ann_indices, num_selected_ann = solve_ann_scoring(N_target, M, N, w, P)
            coverage_ann = calculate_coverage(selected_ann_indices, locs, r, free_space, Lx, Ly)
            current_ann_coverages.append(coverage_ann)
            current_ann_times.append(time.time() - start_time)
        average_coverage['ANN (Simplified Scoring)'][N_target].append(np.mean(current_ann_coverages))
        execution_times['ANN (Simplified Scoring)'][N_target].append(np.mean(current_ann_times))
        print(f"  ANN (Simplified Scoring) Average Coverage: {np.mean(current_ann_coverages):.2f}% (Avg Time: {np.mean(current_ann_times):.4f}s)")

        # --- Hill Climbing ---
        current_hc_coverages = []
        current_hc_times = []
        for event_idx in range(num_random_events): # HC is stochastic due to initial solution and neighbor selection
            start_time = time.time()
            selected_hc_indices, num_selected_hc = solve_hill_climbing(N_target, M, N, w, P)
            coverage_hc = calculate_coverage(selected_hc_indices, locs, r, free_space, Lx, Ly)
            current_hc_coverages.append(coverage_hc)
            current_hc_times.append(time.time() - start_time)
        average_coverage['Hill Climbing'][N_target].append(np.mean(current_hc_coverages))
        execution_times['Hill Climbing'][N_target].append(np.mean(current_hc_times))
        print(f"  Hill Climbing Average Coverage: {np.mean(current_hc_coverages):.2f}% (Avg Time: {np.mean(current_hc_times):.4f}s)")

        # --- Particle Swarm Optimization ---
        current_pso_coverages = []
        current_pso_times = []
        for event_idx in range(num_random_events): # PSO is stochastic
            start_time = time.time()
            selected_pso_indices, num_selected_pso = solve_pso(N_target, M, N, w, P)
            coverage_pso = calculate_coverage(selected_pso_indices, locs, r, free_space, Lx, Ly)
            current_pso_coverages.append(coverage_pso)
            current_pso_times.append(time.time() - start_time)
        average_coverage['Particle Swarm Optimization'][N_target].append(np.mean(current_pso_coverages))
        execution_times['Particle Swarm Optimization'][N_target].append(np.mean(current_pso_times))
        print(f"  Particle Swarm Optimization Average Coverage: {np.mean(current_pso_coverages):.2f}% (Avg Time: {np.mean(current_pso_times):.4f}s)")


# Calculate overall average coverage for each N_target across all room structures
print("\n=== Overall Average Coverage Results ===")
method_names = ['SNN', 'ANN (Simplified Scoring)', 'Hill Climbing', 'Particle Swarm Optimization']

for method_name in method_names:
    print(f"\n--- {method_name} ---")
    for n_target in N_targets:
        if average_coverage[method_name][n_target]:
            overall_avg_coverage = np.mean(average_coverage[method_name][n_target])
            overall_avg_time = np.mean(execution_times[method_name][n_target])
            print(f"N_target {n_target}: Avg Coverage = {overall_avg_coverage:.2f}% (Avg Time = {overall_avg_time:.4f}s)")
        else:
            print(f"N_target {n_target}: No data collected.")

# --- Plotting Results ---
plt.figure(figsize=(12, 6))
for method_name in method_names:
    avg_coverages = [np.mean(average_coverage[method_name][n]) if average_coverage[method_name][n] else 0 for n in N_targets]
    plt.plot(N_targets, avg_coverages, marker='o', label=method_name)

plt.title('Average Coverage vs. Number of Sensors (Comparison of Methods)')
plt.xlabel('Number of Sensors (N_target)')
plt.ylabel('Average Coverage (%)')
plt.grid(True)
plt.legend()
plt.xticks(N_targets)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for method_name in method_names:
    avg_times = [np.mean(execution_times[method_name][n]) if execution_times[method_name][n] else 0 for n in N_targets]
    plt.plot(N_targets, avg_times, marker='o', label=method_name)

plt.title('Average Execution Time vs. Number of Sensors (Comparison of Methods)')
plt.xlabel('Number of Sensors (N_target)')
plt.ylabel('Average Execution Time (s)')
plt.grid(True)
plt.legend()
plt.xticks(N_targets)
plt.yscale('log') # Use log scale for time as metaheuristics can be slower
plt.tight_layout()
plt.show()
