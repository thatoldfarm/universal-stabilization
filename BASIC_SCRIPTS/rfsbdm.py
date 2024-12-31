import numpy as np
import matplotlib.pyplot as plt

# Recursive feedback system implementation
def recursive_feedback_system(forward, backward, steps=20):
    """
    Recursive feedback system for bidirectional data.
    :param forward: Forward input sequence (1D or 2D array).
    :param backward: Backward input sequence (1D or 2D array).
    :param steps: Number of recursive steps.
    :return: Stabilized results and differences (Delta_t).
    """
    weights_f = 1.0  # Initial weight for forward
    weights_b = 1.0  # Initial weight for backward
    results = []
    deltas = []

    prev_result = np.zeros_like(forward)  # Initialize previous result

    for step in range(steps):
        # Recursive transformation
        current_result = (forward * weights_f + backward * weights_b) / (weights_f + weights_b)
        results.append(current_result)

        # Compute the difference (Delta_t)
        delta = np.linalg.norm(current_result - prev_result)  # L2 norm for stability
        deltas.append(delta)
        prev_result = current_result

        # Update weights dynamically based on results
        weights_f = np.mean(np.linalg.norm(current_result, axis=-1))  # Mean of vector norms if multi-dimensional
        weights_b = np.max(np.linalg.norm(current_result, axis=-1))  # Max of vector norms if multi-dimensional

    return results, deltas

# Example: Scalar data (digits of Pi and reversed)
pi_digits = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])  # First 10 digits of Pi for simplicity
negative_pi_digits = pi_digits[::-1]  # Reverse for backward sequence

# Run the system on scalar data
scalar_results, scalar_deltas = recursive_feedback_system(pi_digits, negative_pi_digits, steps=20)

# Plot the geometric decay of Delta_t for scalar data
plt.figure(figsize=(10, 6))
plt.plot(range(len(scalar_deltas)), scalar_deltas, marker="o", label="Scalar Data")
plt.title("Geometric Decay of Delta_t (Scalar Data)")
plt.xlabel("Step (t)")
plt.ylabel("Delta_t")
plt.yscale("log")  # Log scale to emphasize decay
plt.legend()
plt.grid()
plt.show()

# Example: Vector data (2D random points)
forward_vectors = np.random.rand(10, 2)  # 10 random 2D points
backward_vectors = forward_vectors[::-1]  # Reverse for backward sequence

# Run the system on vector data
vector_results, vector_deltas = recursive_feedback_system(forward_vectors, backward_vectors, steps=20)

# Plot the geometric decay of Delta_t for vector data
plt.figure(figsize=(10, 6))
plt.plot(range(len(vector_deltas)), vector_deltas, marker="o", label="Vector Data")
plt.title("Geometric Decay of Delta_t (Vector Data)")
plt.xlabel("Step (t)")
plt.ylabel("Delta_t")
plt.yscale("log")  # Log scale to emphasize decay
plt.legend()
plt.grid()
plt.show()

# Display the stabilized results for both cases
final_scalar_result = scalar_results[-1]
final_vector_result = vector_results[-1]

print("Final Stabilized Scalar Result:", final_scalar_result)
print("Final Stabilized Vector Result:\n", final_vector_result)

