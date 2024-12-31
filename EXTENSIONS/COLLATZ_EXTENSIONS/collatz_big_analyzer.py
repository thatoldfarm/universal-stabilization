import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ARFS-based Collatz function with cycle detection
def arfs_collatz(n, steps, wf, wb, alpha, beta, decay_factor=0.95):
    sequence = [n]
    energy_log = []
    seen = set()  # To detect cycles
    cycle_detected = None

    for t in range(steps):
        current = sequence[-1]

        # Apply Collatz rules
        if current % 2 == 0:
            next_val = current // 2
        else:
            next_val = 3 * current + 1

        # ARFS stabilization
        R = current
        X = next_val
        stabilization = (wf * X - wb * R) / (wf + wb + 1e-10)
        R_stabilized = R + decay_factor**t * stabilization

        # Handle overflow by scaling
        if abs(R_stabilized) > 1e9:
            R_stabilized = R_stabilized % 1e9

        # Calculate energy
        energy = alpha * np.log1p(abs(R_stabilized))**2 + beta * np.log1p(abs(X))**2
        energy_log.append((t, energy))

        # Update sequence and detect cycles
        stabilized_value = max(1, int(R_stabilized))
        sequence.append(stabilized_value)
        if stabilized_value in seen:
            cycle_detected = stabilized_value
            break
        seen.add(stabilized_value)

    return sequence, energy_log, cycle_detected

# Heatmap generation
def generate_heatmap(start_numbers, steps, wf, wb, alpha, beta, decay_factor):
    heatmap_data = []
    for n in start_numbers:
        _, energy_log, _ = arfs_collatz(n, steps, wf, wb, alpha, beta, decay_factor)
        energies = [e for _, e in energy_log]
        # Replace None with np.nan to handle missing values
        padded_energies = energies + [np.nan] * (steps - len(energies))
        heatmap_data.append(padded_energies)

    heatmap_data = np.array(heatmap_data)  # Convert to numpy array for Seaborn
    plt.figure(figsize=(12, 8))  # Adjust figure size for better clarity
    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        cbar=True,
        xticklabels=50,  # Adjust frequency of x-axis labels
        yticklabels=start_numbers
    )
    plt.title(f"Energy Heatmap (wf={wf}, wb={wb}, decay={decay_factor})")
    plt.xlabel("Steps")
    plt.ylabel("Starting Number")

    # Save the heatmap as a PNG
    filename = f"heatmap_wf_{wf}_wb_{wb}_decay_{decay_factor}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved as {filename}")

    # Display the heatmap
    plt.show()

# Experiment automation
def run_advanced_experiments(start_numbers, steps, wf_values, wb_values, alpha_values, beta_values, decay_factors):
    results = {}
    for start_number in start_numbers:
        for wf in wf_values:
            for wb in wb_values:
                for alpha in alpha_values:
                    for beta in beta_values:
                        for decay_factor in decay_factors:
                            key = f"start_{start_number}_wf_{wf}_wb_{wb}_alpha_{alpha}_beta_{beta}_decay_{decay_factor}"
                            sequence, energy_log, cycle_detected = arfs_collatz(start_number, steps, wf, wb, alpha, beta, decay_factor)
                            results[key] = {
                                "sequence": sequence,
                                "energy_log": energy_log,
                                "cycle_detected": cycle_detected
                            }

    with open("arfs_collatz_refined_experiments.json", "w") as f:
        json.dump(results, f, indent=4)
    return results

# Parameters
start_numbers = range(1, 21)  # Analyze numbers from 1 to 20
steps = 4
wf_values = [0.5, 0.8, 0.9]
wb_values = [0.2, 0.5, 0.7]
alpha_values = [0.3, 0.5, 0.7]
beta_values = [0.3, 0.5, 0.7]
decay_factors = [0.90, 0.95, 0.99]

# Run experiments and analyze
results = run_advanced_experiments(start_numbers, steps, wf_values, wb_values, alpha_values, beta_values, decay_factors)

# Example heatmap for visualization
generate_heatmap(start_numbers, steps, 0.5, 0.2, 0.3, 0.3, 0.95)

