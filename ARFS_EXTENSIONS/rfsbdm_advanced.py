import numpy as np
import matplotlib.pyplot as plt
import json

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
    """
    Advanced Recursive Feedback System with extended functionalities.
    :param forward: Forward input sequence (1D, 2D, or higher).
    :param backward: Backward input sequence (1D, 2D, or higher).
    :param steps: Number of recursive steps.
    :param periodic_modulation: Apply periodic modulation to weights.
    :param invariance_transformation: Function to apply invariance transformations.
    :param optimize_energy: Minimize variance or other energy-like quantities.
    :param entropy_maximization: Balance stabilization with information retention.
    :param inter_domain_scaling: Dynamically adapt to higher dimensions.
    :return: Stabilized results and differences (Delta_t).
    """
    weights_f = np.ones_like(forward, dtype=float)  # Initial weights for forward
    weights_b = np.ones_like(backward, dtype=float)  # Initial weights for backward
    results = []
    deltas = []

    prev_result = np.zeros_like(forward)  # Initialize previous result

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
        delta = np.linalg.norm(current_result - prev_result)  # L2 norm for stability
        deltas.append(delta)
        prev_result = current_result

        # Energy optimization (variance minimization)
        if optimize_energy:
            weights_f = 1 / (1 + np.var(current_result))
            weights_b = 1 - weights_f

        # Entropy maximization (balance stabilization with diversity)
        if entropy_maximization:
            entropy = -np.sum(current_result * np.log(current_result + 1e-6))
            weights_f *= np.abs(entropy)
            weights_b *= 1 / (np.abs(entropy) + 1e-6)

        # Inter-domain scaling
        if inter_domain_scaling:
            scaling_factor = np.mean(np.linalg.norm(current_result, axis=-1)) if current_result.ndim > 1 else np.mean(current_result)
            weights_f *= scaling_factor
            weights_b *= 1 / (scaling_factor + 1e-6)

    return results, deltas

# Example data
pi_digits = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])  # Digits of Pi
negative_pi_digits = pi_digits[::-1]  # Reverse for backward sequence

# Run the advanced recursive feedback system
results, deltas = advanced_recursive_feedback(
    forward=pi_digits,
    backward=negative_pi_digits,
    steps=20,
    periodic_modulation=True,
    invariance_transformation=lambda x: x / np.max(x),  # Example: Normalize
    optimize_energy=True,
    entropy_maximization=True,
    inter_domain_scaling=True
)

# Plot the geometric decay of Delta_t
plt.figure(figsize=(10, 6))
plt.plot(range(len(deltas)), deltas, marker="o", label="Delta_t")
plt.title("Geometric Decay of Delta_t (Advanced Recursive Feedback System)")
plt.xlabel("Step (t)")
plt.ylabel("Delta_t")
plt.yscale("log")  # Log scale to emphasize decay
plt.legend()
plt.grid()
plt.show()

# Prepare data for JSON output
output_data = {
    "final_result": results[-1],
    "deltas": deltas
}

# Write output to JSON file
with open("output.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

# Print confirmation
print("Results have been written to output.json")

