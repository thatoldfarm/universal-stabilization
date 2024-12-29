import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Strange Attractor Finder in Four-Leaf Clover Recursive Feedback System
class StrangeAttractorFeedback:
    def __init__(self, learning_rate=0.05):
        self.weights = np.ones(8) * 0.5  # Initialize weights equally
        self.learning_rate = learning_rate  # Rate at which weights evolve

    def compute_feedback(self, R, gravity, time):
        # Clamp gravity and time inputs to avoid extreme values
        gravity = min(max(gravity, 0.01), 10.0)
        time = min(max(time, 0.01), 10.0)

        # Compute feedback for each node
        R_new = np.zeros(4)
        try:
            R_new[0] = (self.weights[0] + self.weights[1]) / (
                (self.weights[0] * R[1] + self.weights[1] * R[3] + self.weights[0] * gravity) /
                (self.weights[0] + self.weights[1] + self.weights[0] + 1e-10)
            )
            R_new[1] = (self.weights[2] + self.weights[3]) / (
                (self.weights[2] * R[2] + self.weights[3] * R[0] + self.weights[2] * gravity) /
                (self.weights[2] + self.weights[3] + self.weights[2] + 1e-10)
            )
            R_new[2] = (self.weights[4] + self.weights[5]) / (
                (self.weights[4] * R[3] + self.weights[5] * R[1] + self.weights[4] * time) /
                (self.weights[4] + self.weights[5] + self.weights[4] + 1e-10)
            )
            R_new[3] = (self.weights[6] + self.weights[7]) / (
                (self.weights[6] * R[0] + self.weights[7] * R[2] + self.weights[6] * time) /
                (self.weights[6] + self.weights[7] + self.weights[6] + 1e-10)
            )
        except OverflowError:
            print("Overflow encountered in feedback computation.")
        return np.clip(R_new, -10, 10)  # Clip outputs to avoid extreme values

    def update_weights(self, R):
        # Adjust weights based on feedback outputs
        self.weights[:4] += self.learning_rate * R
        self.weights[4:] += self.learning_rate * (1 - R)
        # Ensure weights remain positive
        self.weights = np.maximum(self.weights, 0.01)

# Calculate Fractal Dimension using Correlation Dimension Method
def calculate_fractal_dimension(points):
    if len(points) < 2:
        return 0  # Not enough points to calculate fractal dimension

    # Normalize points to avoid extreme distances
    points = (points - np.min(points, axis=0)) / (np.ptp(points, axis=0) + 1e-10)

    distances = pdist(points)
    distances = distances[distances > 1e-10]  # Remove near-zero distances

    if len(distances) == 0:
        return 0  # No valid distances to calculate fractal dimension

    distances = np.sort(distances)

    counts = []
    radii = np.logspace(np.log10(max(distances.min(), 1e-10)), np.log10(distances.max()), num=50)
    for r in radii:
        counts.append(np.sum(distances < r))

    counts = np.array(counts)

    # Ensure valid log-log data
    valid = counts > 0
    if valid.sum() < 2:
        return 0  # Not enough valid data points for regression

    log_radii = np.log(radii[valid])
    log_counts = np.log(counts[valid])

    # Debugging logs for diagnostics
    print("Radii (log):", log_radii)
    print("Counts (log):", log_counts)

    # Perform linear regression on log-log plot to estimate slope
    try:
        slope, _ = np.polyfit(log_radii, log_counts, 1)
    except np.linalg.LinAlgError:
        return 0  # Regression failed due to singular matrix or other issues

    return slope

if __name__ == "__main__":
    # Initialize the system
    system = StrangeAttractorFeedback(learning_rate=0.05)

    # Initial outputs
    R = np.array([1.0, 0.9, 0.8, 0.7])

    # Nonlinear Inputs
    iterations = 2000  # Increased iterations for richer data
    results = []

    # Input functions
    gravity_values = [0.5 * np.sin(0.01 * t) + 0.5 for t in range(iterations)]  # Oscillating gravity
    time_values = [0.2 * np.exp(0.001 * t) for t in range(iterations)]  # Exponentially growing time

    attractor_points = []  # Collect R values for fractal dimension calculation

    for t in range(iterations):
        gravity = gravity_values[t]
        time = time_values[t]

        # Compute new feedback outputs
        R_new = system.compute_feedback(R, gravity, time)

        # Update weights
        system.update_weights(R_new)

        # Record results
        results.append({
            "iteration": t,
            "R": R.tolist(),
            "R_new": R_new.tolist(),
            "weights": system.weights.tolist(),
            "gravity": gravity,
            "time": time
        })

        # Collect points for fractal dimension calculation
        attractor_points.append(R_new.copy())

        # Update R for the next iteration
        R = R_new

    # Save results to a JSON file
    with open("quad_strange_results_with_fractal.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot strange attractor patterns
    R1_values = [point[0] for point in attractor_points]
    R2_values = [point[1] for point in attractor_points]

    plt.figure(figsize=(10, 6))
    plt.plot(R1_values, R2_values, '.', markersize=0.5, alpha=0.7)
    plt.title("Strange Attractor in Four-Leaf Clover Recursive Feedback System")
    plt.xlabel("R1")
    plt.ylabel("R2")
    plt.grid()
    plt.savefig("quad_strange_attractor_with_fractal.png")
    plt.show()

    # Calculate and print fractal dimension
    attractor_points_array = np.array(attractor_points)
    fractal_dimension = calculate_fractal_dimension(attractor_points_array)
    print(f"Estimated Fractal Dimension: {fractal_dimension}")

    print("Simulation complete. Results saved to 'quad_strange_results_with_fractal.json'. Plot saved as 'quad_strange_attractor_with_fractal.png'.")

