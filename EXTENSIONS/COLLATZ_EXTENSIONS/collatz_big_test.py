import numpy as np
import matplotlib.pyplot as plt
import json

# Define ARFS-based Collatz sequence function with decay factor
def arfs_collatz(n, steps, wf, wb, alpha, beta, decay_factor=0.95):
    """
    Simulate Collatz sequence with ARFS stabilization and decay factor.
    n: Starting number for Collatz sequence
    steps: Number of steps to simulate
    wf, wb: Forward and backward weights
    alpha, beta: Energy coefficients
    decay_factor: Reduces stabilization effect over time
    """
    sequence = [n]  # Store Collatz sequence
    energy_log = []  # Store energy dynamics

    for t in range(steps):
        current = sequence[-1]

        # Apply Collatz rules
        if current % 2 == 0:
            next_val = current // 2
        else:
            next_val = 3 * current + 1

        # ARFS stabilization with decay factor
        R = current
        X = next_val
        stabilization = (wf * X - wb * R) / (wf + wb + 1e-10)
        R_stabilized = R + decay_factor**t * stabilization  # Apply decay factor

        # Handle overflow by scaling down large numbers
        if abs(R_stabilized) > 1e9:
            R_stabilized = R_stabilized % 1e9  # Keep values within manageable bounds

        # Calculate energy with log-scaling to avoid overflow
        energy = alpha * np.log1p(abs(R_stabilized))**2 + beta * np.log1p(abs(X))**2
        energy_log.append((t, energy))

        # Append stabilized value to sequence
        sequence.append(max(1, int(R_stabilized)))  # Ensure values are >= 1

        # Stop if we reach 1 or detect a repeating pattern
        if len(sequence) > 2 and sequence[-1] == sequence[-2]:
            break

    return sequence, energy_log

# Function to automate experiments
def run_experiments(start_numbers, steps, wf_values, wb_values, alpha_values, beta_values, decay_factors):
    results = {}

    for start_number in start_numbers:
        for wf in wf_values:
            for wb in wb_values:
                for alpha in alpha_values:
                    for beta in beta_values:
                        for decay_factor in decay_factors:
                            key = f"start_{start_number}_wf_{wf}_wb_{wb}_alpha_{alpha}_beta_{beta}_decay_{decay_factor}"
                            sequence, energy_log = arfs_collatz(start_number, steps, wf, wb, alpha, beta, decay_factor)
                            results[key] = {
                                "sequence": sequence,
                                "energy_log": energy_log
                            }

    # Save results to JSON
    with open("arfs_collatz_experiments.json", "w") as f:
        json.dump(results, f, indent=4)

    return results

# Parameters for experiments
start_numbers = [18, 19, 20, 25]  # Range of starting numbers
steps = 10  # Maximum steps to simulate
wf_values = [0.5, 0.8, 0.9]  # Forward weights
wb_values = [0.2, 0.5, 0.7]  # Backward weights
alpha_values = [0.3, 0.5, 0.7]  # Energy coefficients for R
beta_values = [0.3, 0.5, 0.7]  # Energy coefficients for X
decay_factors = [0.90, 0.95, 0.99]  # Decay factors

# Run experiments
results = run_experiments(start_numbers, steps, wf_values, wb_values, alpha_values, beta_values, decay_factors)

# Example analysis for a specific experiment
example_key = list(results.keys())[0]
sequence = results[example_key]["sequence"]
energy_log = results[example_key]["energy_log"]

# Extract energy data for visualization
time = [t for t, _ in energy_log]
energy = [e for _, e in energy_log]

# Plot Collatz sequence for example
plt.figure(figsize=(10, 6))
plt.plot(range(len(sequence)), sequence, label="Collatz Sequence", color="blue")
plt.xlabel("Steps")
plt.ylabel("Value")
plt.title(f"ARFS-Enhanced Collatz Sequence\n({example_key})")
plt.grid()
plt.legend()
plt.savefig("arfs_collatz_sequence_example.png")
plt.show()

# Plot energy dynamics for example
plt.figure(figsize=(10, 6))
plt.plot(time, energy, label="Energy Dynamics", color="green")
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title(f"Energy Stabilization in ARFS-Enhanced Collatz\n({example_key})")
plt.grid()
plt.legend()
plt.savefig("arfs_collatz_energy_example.png")
plt.show()

