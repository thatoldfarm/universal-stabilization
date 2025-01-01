import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import minimize

# Function to calculate time-dependent weighted average with non-linear weight adjustments
def dynamic_parameter(w_f, w_b, X, X_prime):
    return (w_f * X + w_b * X_prime) / (w_f + w_b)

# Non-linear weight adjustments
# Example: Sinusoidal variation of weights over time
def nonlinear_weights(t, max_weight=1.0):
    w_f = max_weight * (0.5 + 0.5 * np.sin(2 * np.pi * t / 50))  # Periodic forward weight
    w_b = max_weight - w_f  # Complementary backward weight
    return w_f, w_b

# Add stochastic variation to inputs
# Adds Gaussian noise to input values
def add_stochastic_variation(X, noise_std=0.05):
    return X + np.random.normal(0, noise_std, len(X))

# Function to calculate ARFS energy with dynamic parameters
# Includes parameter interdependencies (e.g., A_t influences R_t and F_t influences S_t)
def calculate_dynamic_energy(A_t, R_t, F_t, S_t, K):
    # Example dependency: R_t influenced by A_t, S_t influenced by F_t
    R_t = R_t * (1 + 0.1 * A_t)
    S_t = S_t * (1 + 0.05 * F_t)
    return K * A_t * R_t * F_t * S_t

# Function to evolve dynamic parameters with non-linear weights and stochastic variation
def evolve_parameters(time_steps, X, X_prime, max_weight):
    dynamic_params = []
    for t in range(time_steps):
        w_f, w_b = nonlinear_weights(t, max_weight)
        dynamic_param = dynamic_parameter(w_f, w_b, X[t], X_prime[t])
        dynamic_params.append(dynamic_param)
    return np.array(dynamic_params)

# Analyze parameter evolution
# Includes stochastic variation and interdependencies
def analyze_dynamic_behavior(K, time_steps):
    # Example inputs
    X = np.linspace(0.5, 1.0, time_steps)
    X_prime = np.linspace(0.4, 0.9, time_steps)

    # Add stochastic variation to inputs
    X = add_stochastic_variation(X)
    X_prime = add_stochastic_variation(X_prime)

    # Evolve parameters
    A_t = evolve_parameters(time_steps, X, X_prime * 0.9, max_weight=1.0)
    R_t = evolve_parameters(time_steps, X, X_prime, max_weight=1.0)
    F_t = evolve_parameters(time_steps, X * 0.8, X_prime, max_weight=1.0)
    S_t = evolve_parameters(time_steps, X * 0.7, X_prime * 1.1, max_weight=1.0)

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
    plt.title('Advanced Dynamic Parameters Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Parameter Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/Advanced_Dynamic_Parameters.png")
    plt.close()

    # Plot energy over time
    plt.figure()
    plt.plot(time, energies, label='Advanced Dynamic Energy')
    plt.title('Advanced Dynamic Energy Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/Advanced_Dynamic_Energy.png")
    plt.close()

# Main function
if __name__ == "__main__":
    # Configuration
    K = 1000  # Proportionality constant
    time_steps = 50  # Number of time steps to simulate

    # Analyze dynamic behavior
    A_t, R_t, F_t, S_t, energies = analyze_dynamic_behavior(K, time_steps)

    # Output directory
    output_dir = "output_advanced_dynamics"
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

    print(f"Advanced dynamic behavior analysis and visualizations saved in '{output_dir}' folder.")

