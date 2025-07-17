import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp

def snn_solver(A, b, C, d, t_end, x0, k0=0.09, k1=0.05):
    def constraint_violation(x):
        return C @ x + d

    def grad(x):
        g = A @ x + b
        norm = np.linalg.norm(g)
        return g if norm < 100 else g * 100 / norm  # Clip gradients

    def snn_dynamics(t, x):
        return -k0 * grad(x)

    t_current = 0
    t_vals = []
    X_vals = []

    while t_current < t_end:
        # Spiking correction (project if constraint violated)
        violated = constraint_violation(x0) > 0
        if np.any(violated):
            x0 -= k1 * C.T @ violated.astype(float)

        sol = solve_ivp(snn_dynamics, [t_current, t_end], x0, max_step=0.1)
        t_vals.append(sol.t)
        X_vals.append(sol.y.T)
        t_current = sol.t[-1]
        x0 = sol.y[:, -1]

        if np.all(constraint_violation(x0) <= 0):
            break

    t_all = np.concatenate(t_vals)
    X_all = np.vstack(X_vals)
    return t_all, X_all
