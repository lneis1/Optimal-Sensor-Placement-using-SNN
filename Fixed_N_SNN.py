import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp
import matplotlib.patches as patches

# --- SNN Solver ---
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

# --- Parameters ---
Lx, Ly = 12, 8
r = min(Lx, Ly) * 0.2
nx, ny = int(Lx), int(Ly)
k_spring = 20
samples = 10000
num_obstacles = np.random.randint(2, 11)
t_end, k0, k1 = 500, 0.5, 10
penalty_N_deviation = 100
N_targets = [14, 15, 16, 17, 18, 19, 20]  # <=== Sensor counts to evaluate

room = Polygon([(0,0), (Lx,0), (Lx,Ly), (0,Ly)])
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

# --- Weights ---
theta = 2 * np.pi * np.random.rand(samples)
rad = r * np.sqrt(np.random.rand(samples))
unit_disk_pts = np.vstack([np.cos(theta)*rad, np.sin(theta)*rad]).T
w = np.array([
    np.pi * r**2 * np.mean([free_space.contains(Point(loc + pt)) for pt in unit_disk_pts])
    for loc in locs
])

# --- Penalty matrix ---
D = cdist(locs, locs)
P = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        if D[i, j] < 2 * r:
            P[i, j] = k_spring * (2 * r - D[i, j])**2
            P[j, i] = P[i, j]
A = P

# --- Coverage matrix ---
xg2, yg2 = np.meshgrid(np.linspace(0, Lx, 40), np.linspace(0, Ly, 24))
eval_pts = np.vstack([xg2.ravel(), yg2.ravel()]).T
in_free_eval = np.array([free_space.contains(Point(p)) for p in eval_pts])
eval_pts = eval_pts[in_free_eval]
J = len(eval_pts)
M = np.zeros((J, N))
for j in range(J):
    for i in range(N):
        if np.linalg.norm(eval_pts[j] - locs[i]) <= r:
            M[j, i] = 1

# --- Run for each N_target ---
for N_target in N_targets:
    print(f"\n=== Solving with SNN (Target = {N_target}) ===")

    A_new = A + penalty_N_deviation * np.ones((N, N))
    b_new = -w + penalty_N_deviation * (-2 * N_target * np.ones(N))

    C = np.vstack([-M, np.eye(N), -np.eye(N)])
    d = np.concatenate([-np.ones(J), -np.ones(N), np.zeros(N)])
    x0 = np.random.rand(N)

    t_snn, X_snn = snn_solver(A_new, b_new, C, d, t_end, x0, k0, k1)
    y_final = X_snn[-1, :]

    top_indices = np.argsort(y_final)[-N_target:][::-1]
    selected_snn_indices = top_indices
    num_selected_sensors = len(selected_snn_indices)

    # Monte Carlo Coverage
    test_xy = np.random.rand(10000, 2) * [Lx, Ly]
    in_free = np.array([free_space.contains(Point(p)) for p in test_xy])
    in_coverage = np.zeros(10000, dtype=bool)
    for i in selected_snn_indices:
        dist = np.linalg.norm(test_xy - locs[i], axis=1)
        in_coverage |= (dist <= r)

    total_free = np.sum(in_free)
    covered = np.sum(in_free & in_coverage)
    coverage_rate = (covered / total_free) * 100 if total_free > 0 else 0

    print(f"Selected sensors: {num_selected_sensors}")
    print(f"Coverage: {coverage_rate:.2f}% (Free area: {true_free_area:.2f} units)")
    print(f"Efficiency: {coverage_rate / (num_selected_sensors):.2f}")
