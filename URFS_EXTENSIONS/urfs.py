import numpy as np
import json

# Subsystem-specific contribution functions
def logistic_map(x, r=3.9):
    F = r * x * (1 - x)
    B = -r * x**2
    return F, B

def kalman_filter_contributions(x, A=1.0, B=0.5, K=0.3, Z=0.1):
    F = A * x + B
    B = K * (Z - A * x)
    return F, B

def wave_equation_contributions(U, c=1.0):
    if isinstance(U, (float, int)):
        F = c**2 * U
        B = -U
    elif isinstance(U, np.ndarray):
        F = c**2 * np.gradient(U, edge_order=2)
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

def subsystem_dynamics(X, W_f, W_b, F, B, interaction_effects, scale_factor=1.0):
    return scale_factor * (W_f * F + W_b * B + interaction_effects) / (W_f + W_b)

def weight_evolution(variances, smoothing, prev_weights):
    prev_weights = np.array(prev_weights)
    variances = np.array(variances)
    return smoothing * prev_weights + (1 - smoothing) / (1 + variances)

# Parameters
n_subsystems = 3
n_iterations = 100

# Initial conditions for subsystems
wave_state_length = 10
X_t_wave = np.random.rand(wave_state_length) * 0.1
X_t = [np.random.rand() * 0.1 if i != 2 else X_t_wave for i in range(n_subsystems)]
X_star_wave = np.zeros(wave_state_length)  # Ensure X_star[2] matches X_t[2]
X_star = [np.random.rand() * 0.1 if i != 2 else X_star_wave for i in range(n_subsystems)]
variances = np.full(n_subsystems, 0.001)
weights = weight_evolution(variances, smoothing=1.0, prev_weights=np.ones(n_subsystems))
interaction_matrix = np.random.rand(n_subsystems, n_subsystems) * 0.01
contribution_functions = [logistic_map, kalman_filter_contributions, wave_equation_contributions]

results = []
metrics = {
    "n_subsystems": n_subsystems,
    "n_iterations": n_iterations,
}
lyapunov_values = []

def compute_lyapunov(X_t, X_star, weights):
    return sum(weights[i] * np.linalg.norm(X_t[i] - X_star[i])**2 for i in range(n_subsystems))

# Main loop
for t in range(n_iterations):
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

    # Compute interaction effects
    external_contributions = np.array(F_values)
    interaction_effects = np.dot(interaction_matrix, external_contributions)

    if isinstance(X_t_wave, np.ndarray):
        X_t_wave_next = subsystem_dynamics(
            X_t_wave, weights[2], weights[2], F_values[2], B_values[2], interaction_effects[2], scale_factor=1.0
        )
        X_t_wave = X_t_wave_next

    # Ensure wave subsystem remains consistent
    X_t = [
        subsystem_dynamics(X_t[i], weights[i], weights[i], F_values[i], B_values[i], interaction_effects[i], scale_factor=1.0)
        if i != 2 else X_t_wave  # Use X_t_wave explicitly for the wave subsystem
        for i in range(n_subsystems)
    ]

    # Compute variances for weight evolution
    variances = []
    for i in range(n_subsystems):
        if isinstance(X_t[i], np.ndarray) and isinstance(X_star[i], np.ndarray):
            # Compute variance for arrays by comparing their norms
            variance = float(np.linalg.norm(X_t[i] - X_star[i])**2)
        elif isinstance(X_t[i], (float, int)) and isinstance(X_star[i], (float, int)):
            # Compute squared deviation for scalars
            variance = float((X_t[i] - X_star[i])**2)
        else:
            # Fix type mismatch by resetting to arrays if needed
            if isinstance(X_t[i], (float, int)):
                X_t[i] = np.array([X_t[i]])
            if isinstance(X_star[i], (float, int)):
                X_star[i] = np.array([X_star[i]])
            variance = float(np.linalg.norm(X_t[i] - X_star[i])**2)
        
        variances.append(variance)

    # Update weights
    weights = weight_evolution(variances, smoothing=0.95, prev_weights=weights)

    V_t = compute_lyapunov(X_t, X_star, weights)
    lyapunov_values.append(V_t)
    results.append({
        "iteration": t,
        "lyapunov": V_t,
        "weights": weights.tolist(),
    })

# Save results to JSON
with open("results.json", "w") as f:
    json.dump(results, f)

metrics["lyapunov_values"] = lyapunov_values
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Simulation completed. Results and metrics saved.")

