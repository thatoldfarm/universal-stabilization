import numpy as np
import matplotlib.pyplot as plt
import json

# Define ARFS-based Collatz sequence function
def arfs_collatz(n, steps, wf, wb, alpha, beta):
    """
    Simulate Collatz sequence with ARFS stabilization.
    n: Starting number for Collatz sequence
    steps: Number of steps to simulate
    wf, wb: Forward and backward weights
    alpha, beta: Energy coefficients
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

        # ARFS stabilization
        R = next_val
        X = current
        stabilization = (wf * X - wb * R) / (wf + wb + 1e-10)
        R_stabilized = R + stabilization

        # Calculate energy
        energy = alpha * R_stabilized**2 + beta * X**2
        energy_log.append((t, energy))

        # Append stabilized value to sequence
        sequence.append(max(1, int(R_stabilized)))  # Ensure values are >= 1

        # Stop if we reach 1
        if sequence[-1] == 1:
            break

    return sequence, energy_log

# Parameters
start_number = 19  # Example starting number
steps = 500  # Maximum steps to simulate
wf = 0.8  # Forward weight
wb = 0.2  # Backward weight
alpha = 0.5  # Energy coefficient for R
beta = 0.3  # Energy coefficient for X

# Run ARFS-enhanced Collatz simulation
sequence, energy_log = arfs_collatz(start_number, steps, wf, wb, alpha, beta)

# Extract energy data for visualization
time = [t for t, _ in energy_log]
energy = [e for _, e in energy_log]

# Save results to JSON
results = {
    "start_number": start_number,
    "sequence": sequence,
    "energy_log": energy_log
}

with open("arfs_collatz_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Plot Collatz sequence
plt.figure(figsize=(10, 6))
plt.plot(range(len(sequence)), sequence, label="Collatz Sequence", color="blue")
plt.xlabel("Steps")
plt.ylabel("Value")
plt.title(f"ARFS-Enhanced Collatz Sequence (Start: {start_number})")
plt.grid()
plt.legend()
plt.savefig("arfs_collatz_sequence.png")
plt.show()

# Plot energy dynamics
plt.figure(figsize=(10, 6))
plt.plot(time, energy, label="Energy Dynamics", color="green")
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title("Energy Stabilization in ARFS-Enhanced Collatz")
plt.grid()
plt.legend()
plt.savefig("arfs_collatz_energy.png")
plt.show()

