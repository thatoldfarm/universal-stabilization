import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import minimize

# Function to calculate time-dependent weighted average
def dynamic_parameter(w_f, w_b, X, X_prime):
    return (w_f * X + w_b * X_prime) / (w_f + w_b)

# Function to calculate ARFS energy with dynamic parameters
def calculate_dynamic_energy(A_t, R_t, F_t, S_t, K):
    return K * A_t * R_t * F_t * S_t

# Function to evolve dynamic parameters over time
def evolve_parameters(time_steps, w_f, w_b, X, X_prime):
    dynamic_params = []
    for t in range(time_steps):
        dynamic_param = dynamic_parameter(w_f, w_b, X[t], X_prime[t])
        dynamic_params.append(dynamic_param)
    return np.array(dynamic_params)

# Function to optimize energy
def energy_to_minimize(params, K):
    A, R, F, S = params
    # Ensure parameters stay within valid bounds
    if not (0 <= A <= 1 and 0 <= R <= 1 and F > 0 and 0 <= S <= 1):
        return -np.inf
    return -calculate_dynamic_energy(A, R, F, S, K)  # Negate for maximization

# Perform optimization
def optimize_dynamic_energy(K):
    initial_guess = [0.5, 0.8, 10, 0.7]
    bounds = [(0.5, 1.0), (0.8, 1.0), (5, 15), (0.6, 1.0)]  # Bounds for A, R, F, S
    result = minimize(energy_to_minimize, initial_guess, args=(K,), bounds=bounds)
    return result.x, -result.fun  # Optimal parameters and maximum energy

# Analyze parameter evolution
def analyze_dynamic_behavior(K, time_steps):
    # Example weights and data
    w_f, w_b = 0.7, 0.3
    X = np.linspace(0.5, 1.0, time_steps)  # Forward data
    X_prime = np.linspace(0.4, 0.9, time_steps)  # Backward data

    # Evolve parameters
    R_t = evolve_parameters(time_steps, w_f, w_b, X, X_prime)
    A_t = evolve_parameters(time_steps, w_f, w_b, X, X_prime * 0.9)
    F_t = evolve_parameters(time_steps, w_f, w_b, X * 0.8, X_prime)
    S_t = evolve_parameters(time_steps, w_f, w_b, X * 0.7, X_prime * 1.1)

    # Calculate dynamic energy over time
    energies = [calculate_dynamic_energy(A, R, F, S, K) for A, R, F, S in zip(A_t, R_t, F_t, S_t)]
    return A_t, R_t, F_t, S_t, energies

# Plot dynamic behavior
def plot_dynamic_behavior(A_t, R_t, F_t, S_t, energies, time_steps, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    time = np.arange(time_steps)

    # Plot parameters over time
    plt.figure()
    plt.plot(time, A_t, label='Alignment (A_t)')
    plt.plot(time, R_t, label='Resonance (R_t)')
    plt.plot(time, F_t, label='Frequency (F_t)')
    plt.plot(time, S_t, label='Stabilization (S_t)')
    plt.title('Dynamic Parameters Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Parameter Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/Dynamic_Parameters.png")
    plt.close()

    # Plot energy over time
    plt.figure()
    plt.plot(time, energies, label='Dynamic Energy')
    plt.title('Dynamic Energy Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/Dynamic_Energy.png")
    plt.close()

# Main function
if __name__ == "__main__":
    # Configuration
    K = 1000  # Proportionality constant
    time_steps = 50  # Number of time steps to simulate

    # Analyze dynamic behavior
    A_t, R_t, F_t, S_t, energies = analyze_dynamic_behavior(K, time_steps)

    # Output directory
    output_dir = "output_dynamic_behavior"
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON
    results = {
        "dynamic_parameters": {
            "A_t": A_t.tolist(),
            "R_t": R_t.tolist(),
            "F_t": F_t.tolist(),
            "S_t": S_t.tolist()
        },
        "energies": energies
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generate dynamic behavior plots
    plot_dynamic_behavior(A_t, R_t, F_t, S_t, energies, time_steps, output_dir)

    print(f"Dynamic behavior analysis and visualizations saved in '{output_dir}' folder.")

