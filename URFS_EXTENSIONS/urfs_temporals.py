import numpy as np
import json
from scipy.sparse import random

# --- Subsystem-Specific Functions ---
def logistic_map(x, r=3.9):
    F = r * x * (1 - x)
    B = -r * x**2
    return F, B

def kalman_filter_contributions(x, A=1.0, B=0.5, K=0.3, Z=0.1):
    F = A * x + B
    B = K * (Z - A * x)
    return F, B

def wave_equation_contributions(U, c=1.0, omega=0.05, max_derivative=10, damping=0.0001):
    if isinstance(U, (float, int)):
        F = c**2 * U + omega * np.sin(U) - damping*U
        B = -U
    elif isinstance(U, np.ndarray):
        derivative = np.clip(np.gradient(U, edge_order=2), -max_derivative, max_derivative)
        F = c**2 * derivative + omega * np.sin(U) - damping*U
        B = -derivative
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

# --- Weight Evolution Functions (Domain-Specific) ---
def weight_evolution_logistic(X, R_t, W_f, W_b, eta_f=0.001, eta_b=0.001, min_variance=1e-8, max_val=1e10):
    variance = np.maximum((X - 0.9)**2, min_variance)
    dW_f = eta_f * variance
    dW_b = eta_b * variance
    return np.array([np.clip(W_f - dW_f, -max_val, max_val), np.clip(W_b + dW_b, -max_val, max_val)])

def weight_evolution_kalman(X, R_t, W_f, W_b, eta_f=0.001, eta_b=0.001, min_variance=1e-8, max_val=1e10):
  variance = np.maximum(np.linalg.norm(X)**2, min_variance)
  dW_f = eta_f * variance
  dW_b = eta_b * variance
  return np.array([np.clip(W_f - dW_f, -max_val, max_val), np.clip(W_b + dW_b, -max_val, max_val)])

def weight_evolution_wave(X, R_t, U_prev, W_f, W_b, eta_f=0.001, eta_b=0.001, min_variance=1e-8, max_val=1e10):
    variance = np.maximum(np.linalg.norm(U_prev) / (np.linalg.norm(X) + 1e-8) , min_variance) # the ratio of energies
    dW_f = eta_f * variance
    dW_b = eta_b * variance
    return np.array([np.clip(W_f - dW_f, -max_val, max_val), np.clip(W_b + dW_b, -max_val, max_val)])

# --- Core Recursive Feedback and Weight Update ---
def subsystem_dynamics(X, W_f, W_b, F, B, interaction_effects, scale_factor=1.0, max_val=1e10, damping=0.0001, X_prev=None, interaction_damping=0.0001):
     W_f = np.clip(W_f, 1e-8, max_val)
     W_b = np.clip(W_b, 1e-8, max_val)
     F = np.clip(F, -max_val, max_val) # Clip F
     B = np.clip(B, -max_val, max_val) # Clip B
     interaction_effects = np.clip(interaction_effects, -max_val, max_val)
     if X_prev is not None:
        if isinstance(X, np.ndarray) and isinstance(X_prev, np.ndarray):
            diff = X - X_prev
            value = scale_factor * (W_f * F + W_b * B + interaction_effects - damping*diff - interaction_damping * interaction_effects ) / (W_f + W_b + 1e-8)
        elif isinstance(X, np.ndarray):
            if isinstance(X_prev, (int,float)):
                diff = X - X_prev
                value = scale_factor * (W_f * F + W_b * B + interaction_effects - damping*diff - interaction_damping * interaction_effects) / (W_f + W_b + 1e-8)
            else:
                diff = X - np.array(X_prev, dtype=object)[2] if isinstance(X_prev, list) and len(X_prev) > 2 else X - np.array(X_prev)
                value = scale_factor * (W_f * F + W_b * B + interaction_effects - damping * diff - interaction_damping * interaction_effects) / (W_f + W_b + 1e-8)
        elif isinstance(X_prev, np.ndarray):
            if isinstance(X,(int,float)):
                diff =  np.array(X) - X_prev
                value = scale_factor * (W_f * F + W_b * B + interaction_effects - damping * diff - interaction_damping * interaction_effects ) / (W_f + W_b + 1e-8)
            else:
              diff = np.array(X, dtype=object)[2]  if isinstance(X, list) and len(X) > 2 else np.array(X) - X_prev
              value = scale_factor * (W_f * F + W_b * B + interaction_effects - damping * diff - interaction_damping * interaction_effects) / (W_f + W_b + 1e-8)
        else:
          value = scale_factor * (W_f * F + W_b * B + interaction_effects - damping*(float(X) - float(X_prev)) - interaction_damping * interaction_effects) / (W_f + W_b + 1e-8)
     else:
          value = scale_factor * (W_f * F + W_b * B + interaction_effects) / (W_f + W_b + 1e-8)
     return np.clip(value, -max_val, max_val)

def reverse_subsystem_dynamics(R_t, X, W_f, W_b, scale_factor=1.0, epsilon=1e-8):
    """Evolve the stabilized result R_t backward in time."""
    W_f = float(W_f)
    W_b = float(W_b)

    if isinstance(R_t, np.ndarray) and isinstance(X, np.ndarray):
        return (W_b * R_t - W_f * X) / (W_b + epsilon)
    elif isinstance(R_t, np.ndarray):
        if isinstance(X, (int, float)):
            return (W_b * R_t - W_f * np.array(X)) / (W_b + epsilon)
        else: #  X is not an array but a list or something else
            return (W_b * R_t - W_f * np.array(X, dtype=object)[2] if isinstance(X, list) and len(X) > 2 else W_b * R_t - W_f * np.array(X)) / (W_b + epsilon)
    elif isinstance(X, np.ndarray):
        if isinstance(R_t, (int, float)):
            return (W_b * np.array(R_t) - W_f * X) / (W_b + epsilon)
        else:
             return (W_b * np.array(R_t, dtype=object)[2]  if isinstance(R_t, list) and len(R_t)>2 else W_b * np.array(R_t) - W_f * X) / (W_b + epsilon)
    else:
      return (W_b * R_t - W_f * X) / (W_b + epsilon)

def compute_lyapunov_backward(X_t, X_t_prev, weights, max_val=1e10):
    if X_t_prev is None:
       return 0.0
    return sum(np.clip(np.linalg.norm(X_t[i] - X_t_prev[i])**2, 0, max_val)  for i in range(n_subsystems))

def wave_equation_contributions(U, c=1.0, omega=0.05, max_derivative=10, damping=0.0001):
    if isinstance(U, (float, int)):
        F = c**2 * U + omega * np.sin(U) - damping*U
        B = -U
    elif isinstance(U, np.ndarray):
        derivative = np.clip(np.gradient(U, edge_order=2), -max_derivative, max_derivative)
        F = c**2 * derivative + omega * np.sin(U) - damping*U
        B = -derivative
    else:
        raise ValueError("Unsupported input type for wave equation contributions.")
    return F, B
# --- Parameters ---
n_subsystems = 3
n_iterations = 200
history_length = 50
causality_violation_strength = 0.05

# --- Initial Conditions ---
wave_state_length = 10
X_t_wave = np.random.rand(wave_state_length) * 0.1
X_t = [np.random.rand() * 0.1 if i != 2 else X_t_wave for i in range(n_subsystems)]
X_star_wave = np.zeros(wave_state_length)
X_star = [0.0 if i != 2 else X_star_wave for i in range(n_subsystems)]
state_histories = [[] for _ in range(n_subsystems)]
variances = np.full(n_subsystems, 0.001)
weights = [np.array([1, 0.001]) for _ in range(n_subsystems)] # Initialize w_f and w_b
interaction_matrix = random(n_subsystems, n_subsystems, density=0.3).toarray() * 0.01 # Sparsity
contribution_functions = [logistic_map, kalman_filter_contributions, wave_equation_contributions]
weight_functions = [weight_evolution_logistic, weight_evolution_kalman, weight_evolution_wave]

# --- Metrics ---
results = []
metrics = {
    "n_subsystems": n_subsystems,
    "n_iterations": n_iterations,
    "history_length": history_length,
}
lyapunov_values = []
rho_values = []
tolerance = 1e-2

def compute_lyapunov_forward(X_t, X_star, weights, max_val = 1e10):
    return np.clip(sum(weights[i][0] * np.linalg.norm(X_t[i] - X_star[i])**2 for i in range(n_subsystems)) , 0, max_val)

def compute_lyapunov_backward(X_t, X_t_prev, weights, max_val=1e10):
    if X_t_prev is None:
       return 0.0
    return sum(weights[i][0] * np.clip(np.linalg.norm(X_t[i] - X_t_prev[i])**2, 0, max_val)  for i in range(n_subsystems))

def compute_rho(weights):
  return  [w[1] / (w[0] + w[1] + 1e-8) for w in weights] # add small epsilon

# --- Main Loop ---
X_t_prev = None
R_history = [] # store states as the results # initialized here
for t in range(n_iterations):

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
        B_values[i] + causality_violation_strength * (X_t[i] - X_t_prev[i]) if i != 2 and X_t_prev is not None else B_values[i] + causality_violation_strength * np.sum(X_t[i] - X_t_prev[i]) if isinstance(X_t_prev, list) else B_values[i]
         for i in range(n_subsystems)
    ]

    # Compute inter-subsystem contributions
    interaction_contributions = [
      np.sum([interaction_matrix[i, j] * F_values[j] for j in range(n_subsystems)])
      for i in range(n_subsystems)
    ]

    if isinstance(X_t_wave, np.ndarray):
        wave_F, wave_B = contributions[2]
        X_t_wave_next = subsystem_dynamics(
            X_t, weights[2][0], weights[2][1], wave_F, wave_B, interaction_contributions[2], scale_factor=1.0, X_prev = X_t[2], interaction_damping=0.0001
        )
        X_t_wave = X_t_wave_next

    # Update all subsystems based on the recursive equation
    X_t_next = [
         subsystem_dynamics(X_t[i], weights[i][0], weights[i][1], F_values[i], B_values_with_causality_violation[i], interaction_contributions[i], scale_factor=1.0, X_prev = X_t_prev[i] if X_t_prev is not None else None, interaction_damping=0.0001)
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

    # Update weights with a diversity term
    weights = [
        weight_functions[i](X_t_next[i] if i!=2 else X_t_wave, X_t[i] if i != 2 else X_t_wave, X_t_prev[2] if i==2 and X_t_prev is not None else 0, weights[i][0], weights[i][1])
        for i in range(n_subsystems)
    ]
    V_t = compute_lyapunov_forward(X_t_next, X_star, weights, max_val=1e10)
    lyapunov_values.append(V_t)
    rho_t = compute_rho(weights)
    rho_values.append(rho_t)

    results.append({
        "iteration": t,
        "lyapunov": V_t,
        "weights": [w.tolist() for w in weights],
        "rho": rho_t,
    })

    X_t_prev = X_t
    X_t = X_t_next
    R_history.append(X_t)

# --- Main Loop (Backward Pass) ---
X_t_reverse = R_history[-1] # start from the end
lyapunov_reverse_values = []
lyapunov_reverse_differences = []
for t in range(n_iterations):
  max_val = 1e10
  t_reverse = len(R_history) - 1 - t
  if t_reverse < 1:
    break  # Stop if we are at the start of the history
  X_t_prev_reverse = [x.copy() if isinstance(x, np.ndarray) else x for x in X_t_reverse]

  X_t_reverse_next = [
      reverse_subsystem_dynamics(X_t_reverse[i], R_history[t_reverse-1][i] if i!=2 else R_history[t_reverse-1], float(weights[i][0]), float(weights[i][1]))
      for i in range(n_subsystems)
  ]
  V_t_reverse = compute_lyapunov_backward(X_t_reverse, X_t_prev_reverse, weights, max_val=max_val)
  lyapunov_reverse_values.append(V_t_reverse)
  if t > 0:
      lyapunov_reverse_differences.append(np.clip(abs(lyapunov_reverse_values[t]-lyapunov_reverse_values[t-1]), -max_val, max_val))

  results.append({
        "iteration": t,
        "lyapunov_backward": V_t_reverse,
        "weights": [w.tolist() for w in weights],
        "rho": rho_t,
    })
  X_t_reverse = X_t_reverse_next
  if t > 10:
    if V_t_reverse > 0 and  np.mean(lyapunov_reverse_differences[-10:]) / lyapunov_reverse_values[t-1] < tolerance:
          print(f"Converged at iteration: {t}")
          break


# Save results to JSON
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

metrics["lyapunov_values"] = lyapunov_values
metrics["lyapunov_reverse_values"] = lyapunov_reverse_values
metrics["rho_values"] = rho_values

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Simulation completed. Results and metrics saved.")
