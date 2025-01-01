import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import minimize

# Function to calculate ARFS energy
def calculate_energy(A, R, F, S, K):
    return K * A * R * F * S

# Function to optimize energy
def energy_to_minimize(params, K):
    A, R, F, S = params
    # Ensure parameters stay within valid bounds
    if not (0 <= A <= 1 and 0 <= R <= 1 and F > 0 and 0 <= S <= 1):
        return -np.inf
    return -calculate_energy(A, R, F, S, K)  # Negate for maximization

# Perform optimization
def optimize_energy(K):
    initial_guess = [0.5, 0.8, 10, 0.7]
    bounds = [(0.5, 1.0), (0.8, 1.0), (5, 15), (0.6, 1.0)]  # Bounds for A, R, F, S
    result = minimize(energy_to_minimize, initial_guess, args=(K,), bounds=bounds)
    return result.x, -result.fun  # Optimal parameters and maximum energy

# Analyze trade-offs
def analyze_tradeoffs(K, F_values):
    S_values = np.linspace(0.6, 1.0, len(F_values))  # Stabilization increases as Frequency increases
    energies = [calculate_energy(1, 1, F, S, K) for F, S in zip(F_values, S_values)]
    return F_values, S_values, energies

# Plot trade-offs
def plot_tradeoffs(F_values, S_values, energies, output_dir):
    plt.figure()
    plt.scatter(S_values, F_values, c=energies, cmap='viridis', label='Energy Levels')
    plt.colorbar(label='Energy')
    plt.title('Trade-offs: Stabilization vs Frequency')
    plt.xlabel('Stabilization (S)')
    plt.ylabel('Frequency (F)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/Tradeoffs_S_vs_F.png")
    plt.close()

# Main function
if __name__ == "__main__":
    # Configuration
    K = 1000  # Proportionality constant

    # Optimize energy
    optimal_params, max_energy = optimize_energy(K)
    print(f"Optimal Parameters: A={optimal_params[0]}, R={optimal_params[1]}, F={optimal_params[2]}, S={optimal_params[3]}")
    print(f"Maximum Energy: {max_energy}")

    # Analyze trade-offs
    F_values = np.linspace(5, 15, 10)
    S_values, _, energies = analyze_tradeoffs(K, F_values)

    # Output directory
    output_dir = "output_optimization_tradeoffs"
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON
    results = {
        "optimal_params": {
            "A": optimal_params[0],
            "R": optimal_params[1],
            "F": optimal_params[2],
            "S": optimal_params[3],
        },
        "max_energy": max_energy,
        "tradeoffs": {
            "F_values": F_values.tolist(),
            "S_values": S_values.tolist(),
            "energies": energies,
        },
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generate trade-off plot
    plot_tradeoffs(F_values, S_values, energies, output_dir)
    print(f"Results and visualizations saved in '{output_dir}' folder.")

