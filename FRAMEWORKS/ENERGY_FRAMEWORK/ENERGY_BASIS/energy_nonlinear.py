import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import minimize

# Function to calculate ARFS energy with non-linear trade-offs
def calculate_energy_with_tradeoffs(A, R, F, S, K):
    # Introduce diminishing returns for S and F
    effective_F = F * np.exp(-0.1 * S)  # Frequency decreases with high stabilization
    effective_S = S * np.exp(-0.1 * F)  # Stabilization decreases with high frequency
    return K * A * R * effective_F * effective_S

# Function to optimize energy
def energy_to_minimize(params, K):
    A, R, F, S = params
    # Ensure parameters stay within valid bounds
    if not (0 <= A <= 1 and 0 <= R <= 1 and F > 0 and 0 <= S <= 1):
        return -np.inf
    return -calculate_energy_with_tradeoffs(A, R, F, S, K)  # Negate for maximization

# Perform optimization
def optimize_energy(K):
    initial_guess = [0.5, 0.8, 10, 0.7]
    bounds = [(0.5, 1.0), (0.8, 1.0), (5, 15), (0.6, 1.0)]  # Bounds for A, R, F, S
    result = minimize(energy_to_minimize, initial_guess, args=(K,), bounds=bounds)
    return result.x, -result.fun  # Optimal parameters and maximum energy

# Analyze broader parameter ranges
def analyze_broader_ranges(K):
    A_values = np.linspace(0.5, 1.0, 10)
    R_values = np.linspace(0.8, 1.0, 10)
    F_values = np.linspace(5, 20, 10)
    S_values = np.linspace(0.5, 1.5, 10)  # Expanded range for S
    energies = []

    for A in A_values:
        for R in R_values:
            for F in F_values:
                for S in S_values:
                    energy = calculate_energy_with_tradeoffs(A, R, F, S, K)
                    energies.append((A, R, F, S, energy))

    return energies

# Plot broader range analysis
def plot_broader_ranges(energies, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    A, R, F, S, energy = zip(*energies)
    plt.figure()
    plt.scatter(S, F, c=energy, cmap='viridis', label='Energy Levels')
    plt.colorbar(label='Energy')
    plt.title('Broader Analysis: Stabilization vs Frequency with Energy Levels')
    plt.xlabel('Stabilization (S)')
    plt.ylabel('Frequency (F)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/Broader_Analysis_S_vs_F.png")
    plt.close()

# Main function
if __name__ == "__main__":
    # Configuration
    K = 1000  # Proportionality constant

    # Optimize energy with trade-offs
    optimal_params, max_energy = optimize_energy(K)
    print(f"Optimal Parameters: A={optimal_params[0]}, R={optimal_params[1]}, F={optimal_params[2]}, S={optimal_params[3]}")
    print(f"Maximum Energy: {max_energy}")

    # Analyze broader ranges
    energies = analyze_broader_ranges(K)

    # Output directory
    output_dir = "output_non_linear_tradeoffs"
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
        "broader_ranges": [
            {"A": A, "R": R, "F": F, "S": S, "energy": energy} for A, R, F, S, energy in energies
        ]
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generate broader range plot
    plot_broader_ranges(energies, output_dir)
    print(f"Results and visualizations saved in '{output_dir}' folder.")

