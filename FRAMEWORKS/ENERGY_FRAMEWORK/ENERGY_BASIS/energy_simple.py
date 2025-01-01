import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Function to calculate ARFS energy
def calculate_energy(A, R, F, S, K):
    return K * A * R * F * S

# Function to visualize results
def plot_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Energy vs Parameters
    for param in ['A', 'R', 'F', 'S']:
        values = results[param]
        energy = results['E']
        plt.figure()
        plt.plot(values, energy, marker='o')
        plt.title(f'Energy vs {param}')
        plt.xlabel(param)
        plt.ylabel('Energy')
        plt.grid(True)
        plt.savefig(f'{output_dir}/Energy_vs_{param}.png')
        plt.close()

# Function to ensure JSON serialization compatibility
def make_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    elif isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}  # Recursively handle dictionaries
    elif isinstance(data, list):
        return [make_serializable(value) for value in data]  # Recursively handle lists
    else:
        return data  # Return the item if it's already serializable

# Main function
if __name__ == "__main__":
    # Configuration
    config = {
        "A": np.linspace(0.5, 1.0, 10),  # Alignment
        "R": np.linspace(0.8, 1.0, 10),  # Resonance
        "F": np.linspace(5, 15, 10),     # Frequency (Hz)
        "S": np.linspace(0.6, 1.0, 10),  # Stabilization
        "K": 1000                        # Constant
    }

    results = {
        "A": [],
        "R": [],
        "F": [],
        "S": [],
        "E": []
    }

    # Calculate energy for all combinations
    for A in config['A']:
        for R in config['R']:
            for F in config['F']:
                for S in config['S']:
                    energy = calculate_energy(A, R, F, S, config['K'])
                    results['A'].append(A)
                    results['R'].append(R)
                    results['F'].append(F)
                    results['S'].append(S)
                    results['E'].append(energy)

    # Convert numpy arrays and lists for JSON serialization
    config_serializable = make_serializable(config)
    results_serializable = make_serializable(results)

    # Save results to JSON
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({"config": config_serializable, "results": results_serializable}, f, indent=4)

    # Generate visualizations
    plot_results(results, output_dir)

    print(f"Results and visualizations saved in '{output_dir}' folder.")

