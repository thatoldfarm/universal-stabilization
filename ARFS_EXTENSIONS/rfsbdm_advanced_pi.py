import numpy as np
import json

# Function to load digits of Pi from a file
def load_pi_digits(file_path, max_digits=None):
    """
    Load digits of Pi from a text file.
    :param file_path: Path to the text file containing Pi digits.
    :param max_digits: Maximum number of digits to load.
    :return: List of integers representing the digits of Pi.
    """
    with open(file_path, "r") as file:
        content = file.read().strip()
        digits = [int(char) for char in content if char.isdigit()]
        return digits[:max_digits] if max_digits else digits

# Load the first million digits of Pi
file_path = "pi_digits.txt"
pi_digits = load_pi_digits(file_path, max_digits=1000000)  # Adjust max_digits if needed

# Prepare forward and backward sequences
forward_sequence = pi_digits[:500000]  # Use the first 500,000 digits Adjust to 1000000 for a million
backward_sequence = forward_sequence[::-1]  # Reverse for backward sequence

# Advanced Recursive Feedback System
def advanced_recursive_feedback(
    forward,
    backward,
    steps=20,
    periodic_modulation=True,
    invariance_transformation=None,
    optimize_energy=True,
    entropy_maximization=True,
    inter_domain_scaling=True
):
    weights_f = np.ones_like(forward, dtype=float)
    weights_b = np.ones_like(backward, dtype=float)
    results = []
    deltas = []

    prev_result = np.zeros_like(forward, dtype=float)

    for step in range(steps):
        # Periodic modulation
        if periodic_modulation:
            modulation_factor = np.abs(np.sin(np.pi * step)) + 1e-6  # Avoid zero
            weights_f *= modulation_factor
            weights_b *= 1 / modulation_factor

        # Apply invariance transformation if provided
        if invariance_transformation:
            forward = invariance_transformation(forward)
            backward = invariance_transformation(backward)

        # Recursive transformation
        current_result = (weights_f * forward + weights_b * backward) / (weights_f + weights_b)
        results.append(current_result.tolist())

        # Compute the difference (Delta_t)
        delta = np.linalg.norm(current_result - prev_result)
        deltas.append(delta)
        prev_result = current_result

        # Energy optimization
        if optimize_energy:
            weights_f = 1 / (1 + np.var(current_result))
            weights_b = 1 - weights_f

        # Entropy maximization
        if entropy_maximization:
            entropy = -np.sum(current_result * np.log(current_result + 1e-6))
            weights_f *= np.abs(entropy)
            weights_b *= 1 / (np.abs(entropy) + 1e-6)

        # Inter-domain scaling
        if inter_domain_scaling:
            scaling_factor = np.mean(np.linalg.norm(current_result)) if current_result.ndim > 1 else np.mean(current_result)
            weights_f *= scaling_factor
            weights_b *= 1 / (scaling_factor + 1e-6)

    return results, deltas

# Run the system
results, deltas = advanced_recursive_feedback(
    forward=forward_sequence,
    backward=backward_sequence,
    steps=50,
    periodic_modulation=True,
    invariance_transformation=lambda x: x / np.max(x),  # Normalize
    optimize_energy=True,
    entropy_maximization=True,
    inter_domain_scaling=True
)

# Save results to a JSON file
output_data = {
    "dataset_size": len(forward_sequence),
    "final_result": results[-1],
    "deltas": deltas,
}
with open("large_pi_output.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print("Processing complete. Results saved to large_pi_output.json.")

