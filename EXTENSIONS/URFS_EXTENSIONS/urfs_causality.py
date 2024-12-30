import numpy as np
import json

# --- Subsystem-Specific Functions ---
def logistic_map(x, r=3.9):
    F = r * x * (1 - x)
    B = -r * x**2
    return F, B

def kalman_filter_contributions(x, A=1.0, B=0.5, K=0.3, Z=0.1):
    F = A * x + B
    B = K * (Z - A * x)
    return F, B

def wave_equation_contributions(U, c=1.0, omega=0.05):
    if isinstance(U, (float, int)):
        F = c**2 * U + omega * np.sin(U)
        B = -U
    elif isinstance(U, np.ndarray):
        F = c**2 * np.gradient(U, edge_order=2) + omega * np.sin(U)
        B = -np.gradient(U, edge_order=2)
    else:
        raise ValueError("Unsupported input type for wave equation contributions.")
    return F, B

def fourier_to_kalman(frequency_data):
    time_domain_data = np.fft.ifft(frequency_data).real
    return time_domain_data

def bayesian_to_gradient_descent(posterior):
    if isinstance(posterior, (float, int)):
        return posterior * 0.01
    raise ValueError("Posterior should be a float or int.")

# --- Core Recursive Feedback and Weight Update ---
def subsystem_dynamics(X, W_f, W_b, F, B, interaction_effects, scale_factor=1.0):
     return scale_factor * (W_f * F + W_b * B + interaction_effects) / (W_f + W_b + 1e-8)

def weight_evolution(X, R_t, W_f, W_b, eta=0.001):
     """Adapt weights based on differences between X and R_t."""
     dW_f = eta * np.linalg.norm(X - R_t) # minimize diffs
     dW_b = eta * np.linalg.norm(X - R_t)  # minimize diffs
     return np.array([W_f - dW_f, W_b + dW_b])

# --- Parameters ---
n_subsystems = 3
n_iterations = 200
history_length = 50
causality_violation_strength = 0.5

# --- Initial Conditions ---
wave_state_length = 10
X_t_wave = np.random.rand(wave_state_length) * 0.1
X_t = [np.random.rand() * 0.1 if i != 2 else X_t_wave for i in range(n_subsystems)]
X_star_wave = np.zeros(wave_state_length)
X_star = [0.0 if i != 2 else X_star_wave for i in range(n_subsystems)]
state_histories = [[] for _ in range(n_subsystems)]
variances = np.full(n_subsystems, 0.001)
weights = [np.array([1, 0.001]) for _ in range(n_subsystems)] # Initialize w_f and w_b
interaction_matrix = np.random.rand(n_subsystems, n_subsystems) * 0.01
contribution_functions = [logistic_map, kalman_filter_contributions, wave_equation_contributions]

# --- Metrics ---
results = []
metrics = {
    "n_subsystems": n_subsystems,
    "n_iterations": n_iterations,
    "history_length": history_length,
}
lyapunov_values = []
rho_values = [] # store rho values
tolerance = 1e-2

def compute_lyapunov(X_t, X_star, weights):
    return sum(weights[i][0] * np.linalg.norm(X_t[i] - X_star[i])**2 for i in range(n_subsystems))

def compute_rho(weights):
  return  [w[1] / (w[0] + w[1] + 1e-8) for w in weights] # add small epsilon

# --- Main Loop ---
for t in range(n_iterations):
    # Store current state
    X_t_prev =  [x.copy() if isinstance(x, np.ndarray) else x for x in X_t]

    # Get the raw contributions from each subsystem
    contributions = [
        contribution_functions[i](X_t[i] if i != 2 else X_t_wave)
        for i in range(n_subsystems)
    ]

    # Simulate interconnections
    if isinstance(X_t_wave, np.ndarray):
        kalman_observation = fourier_to_kalman(contributions[2][0])
        contributions[1] = kalman_filter_contributions(X_t[1], Z=kalman_observation)
    gradient_learning_rate = bayesian_to_gradient_descent(contributions[0][0])
    contributions[0] = logistic_map(X_t[0], r=gradient_learning_rate)

    # Extract F and B values
    F_values = [np.mean(F) if isinstance(F, np.ndarray) else F for F, B in contributions]
    B_values = [np.mean(B) if isinstance(B, np.ndarray) else B for F, B in contributions]

    # Introduce causal violations with a backward term
    B_values_with_causality_violation = [
        B_values[i] + causality_violation_strength * (X_t_prev[i] - X_t[i]) if i != 2 else B_values[i]  + causality_violation_strength * np.sum(X_t_prev[i] - X_t[i])
         for i in range(n_subsystems)
    ]

    # Compute inter-subsystem contributions
    interaction_contributions = [
      np.sum([interaction_matrix[i, j] * F_values[j] for j in range(n_subsystems)])
      for i in range(n_subsystems)
    ]

    if isinstance(X_t_wave, np.ndarray):
        X_t_wave_next = subsystem_dynamics(
            X_t, weights[2][0], weights[2][1], F_values[2], B_values_with_causality_violation[2], interaction_contributions[2], scale_factor=1.0
        )
        X_t_wave = X_t_wave_next

    # Update all subsystems based on the recursive equation
    X_t_next = [
         subsystem_dynamics(X_t[i], weights[i][0], weights[i][1], F_values[i], B_values_with_causality_violation[i], interaction_contributions[i], scale_factor=1.0)
        if i != 2 else X_t_wave
         for i in range(n_subsystems)
    ]

    # Update state histories and calculate variance
    for i in range(n_subsystems):
      if isinstance(X_t_next[i], np.ndarray):
          state_histories[i].append(X_t_next[i].copy())
      else:
          state_histories[i].append(X_t_next[i])

      if len(state_histories[i]) > history_length:
        state_histories[i].pop(0)

    variances = [np.var(np.array(state_histories[i]), axis=0) if len(state_histories[i]) > 0 else np.array(0.001) for i in range(n_subsystems) ]

    # Update weights
    weights = [
        weight_evolution(X_t_next[i], X_t[i], weights[i][0], weights[i][1])
        for i in range(n_subsystems)
    ]

    V_t = compute_lyapunov(X_t_next, X_star, weights)
    lyapunov_values.append(V_t)
    rho_t = compute_rho(weights)
    rho_values.append(rho_t)
    results.append({
        "iteration": t,
        "lyapunov": V_t,
        "weights": [w.tolist() for w in weights],
        "rho": rho_t,
    })

    X_t = X_t_next

    if t > 10:
        if lyapunov_values[t] > 0 and abs(lyapunov_values[t] - lyapunov_values[t-1]) / lyapunov_values[t] < tolerance:
            print(f"Converged at iteration: {t}")
            break

# Save results to JSON
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

metrics["lyapunov_values"] = lyapunov_values
metrics["rho_values"] = rho_values
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("Simulation completed. Results and metrics saved.")
