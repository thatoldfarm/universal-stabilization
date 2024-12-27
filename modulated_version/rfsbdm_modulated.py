import numpy as np
import matplotlib.pyplot as plt

# Constants for the datasets
PI_DIGITS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]
SQRT5_DIGITS = [2, 2, 3, 6, 0, 6, 7, 9, 2, 2, 8, 7, 4, 4, 0, 8, 9, 6, 4, 6]
ONE_SEVENTH_DIGITS = [1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4]
RANDOM_DIGITS = np.random.randint(0, 10, 20).tolist()

# Datasets for testing
datasets = {
    "Pi Digits": PI_DIGITS,
    "Sqrt5 Digits": SQRT5_DIGITS,
    "1/7 Digits": ONE_SEVENTH_DIGITS,
    "Random Digits": RANDOM_DIGITS
}

# Recursive Feedback System with π-modulated weights
def recursive_feedback_with_pi(data, steps=20, pi_modulation=True, modulation_type="periodic"):
    n = len(data)
    forward_weights = np.ones(n)
    backward_weights = np.ones(n)
    stabilized_outputs = []
    deltas = []

    prev_result = np.zeros(n, dtype=float)
    data = np.array(data, dtype=float)
    reversed_data = data[::-1]

    for step in range(steps):
        # Apply modulation to weights
        if pi_modulation:
            if modulation_type == "periodic":
                forward_weights *= abs(np.sin(np.pi * step))
                backward_weights *= abs(np.cos(np.pi * step))
            elif modulation_type == "scaling":
                forward_weights *= np.pi
                backward_weights *= 22 / 7
            elif modulation_type == "proportional":
                forward_weights /= (1 + np.pi)
                backward_weights /= (1 + 22 / 7)

        # Compute recursive feedback
        current_result = (
            forward_weights * data + backward_weights * reversed_data
        ) / (forward_weights + backward_weights)
        stabilized_outputs.append(current_result.copy())

        # Compute delta for convergence analysis
        delta = np.linalg.norm(current_result - prev_result)
        deltas.append(delta)
        prev_result = current_result.copy()

        # Update weights based on current results
        forward_weights = np.mean(current_result)
        backward_weights = np.max(current_result)

    return stabilized_outputs, deltas

# Test the system across datasets and modulation types
results = {}
modulation_types = ["periodic", "scaling", "proportional"]

for name, data in datasets.items():
    results[name] = {}
    for modulation in modulation_types:
        outputs, deltas = recursive_feedback_with_pi(data, modulation_type=modulation)
        results[name][modulation] = {
            "final_output": outputs[-1],
            "deltas": deltas
        }

# Plot convergence analysis for all datasets
plt.figure(figsize=(12, 8))
for name, dataset_results in results.items():
    for modulation, result in dataset_results.items():
        plt.plot(result["deltas"], label=f"{name} ({modulation})")
plt.yscale("log")
plt.title("Convergence Analysis of Recursive Feedback System")
plt.xlabel("Iteration Step")
plt.ylabel("Delta (Change)")
plt.legend()
plt.grid()
plt.show()

# Output final stabilized results for all datasets and modulation types
for name, dataset_results in results.items():
    print(f"Results for {name}:")
    for modulation, result in dataset_results.items():
        print(f"  {modulation.capitalize()} Modulation: Final Output: {result['final_output']}")

