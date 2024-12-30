import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Define Jacob's Ladder feedback system
class JacobsLadder:
    def __init__(self, learning_rate=0.05):
        self.weights = np.ones(16) * 0.5  # Initialize weights
        self.learning_rate = learning_rate

    def compute_feedback(self, R, forces):
        """Compute feedback for each node based on the expanded system."""
        feedback = np.zeros(8)
        F = forces  # Forces [G, T, E, S, Q, pi, phi, Lambda]

        feedback[0] = (self.weights[0] + self.weights[1]) / (
            (self.weights[0] * R[1] + self.weights[1] * R[3] + self.weights[0] * F[0]) /
            (self.weights[0] + self.weights[1] + 1e-10)
        )
        feedback[1] = (self.weights[2] + self.weights[3]) / (
            (self.weights[2] * R[2] + self.weights[3] * R[0] + self.weights[2] * F[1]) /
            (self.weights[2] + self.weights[3] + 1e-10)
        )
        feedback[2] = (self.weights[4] + self.weights[5]) / (
            (self.weights[4] * R[3] - self.weights[5] * R[1] + self.weights[4] * F[2]) /
            (self.weights[4] + self.weights[5] + 1e-10)
        )
        feedback[3] = (self.weights[6] + self.weights[7]) / (
            (self.weights[6] * R[0] + self.weights[7] * R[2] + self.weights[6] * F[3]) /
            (self.weights[6] + self.weights[7] + 1e-10)
        )
        feedback[4] = (self.weights[8] + self.weights[9]) / (
            (self.weights[8] * R[1] + self.weights[9] * R[3] + self.weights[8] * F[4]) /
            (self.weights[8] + self.weights[9] + 1e-10)
        )
        feedback[5] = (self.weights[10] + self.weights[11]) / (
            (self.weights[10] * R[2] + self.weights[11] * R[4] + self.weights[10] * F[5]) /
            (self.weights[10] + self.weights[11] + 1e-10)
        )
        feedback[6] = (self.weights[12] + self.weights[13]) / (
            (self.weights[12] * R[3] + self.weights[13] * R[5] + self.weights[12] * F[6]) /
            (self.weights[12] + self.weights[13] + 1e-10)
        )
        feedback[7] = (self.weights[14] + self.weights[15]) / (
            (self.weights[14] * R[4] + self.weights[15] * R[6] + self.weights[14] * F[7]) /
            (self.weights[14] + self.weights[15] + 1e-10)
        )

        return np.clip(feedback, -10, 10)  # Clip outputs to avoid extreme values

    def update_weights(self, R):
        """Update weights based on feedback."""
        self.weights[:8] += self.learning_rate * R
        self.weights[8:] += self.learning_rate * (1 - R)
        self.weights = np.maximum(self.weights, 0.01)  # Keep weights positive

# Initialize the system
ladder = JacobsLadder(learning_rate=0.05)

# Initial outputs
R = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
iterations = 2000
results = []

# Dynamic inputs for forces
forces = [
    [0.5 * np.sin(0.01 * t) + 0.5 for t in range(iterations)],  # Gravity (G)
    [0.2 * np.exp(0.001 * t) for t in range(iterations)],       # Time (T)
    [np.random.uniform(0.1, 1) for _ in range(iterations)],     # Electromagnetism (E)
    [np.random.uniform(0.1, 1) for _ in range(iterations)],     # Entropy (S)
    [np.random.uniform(0.1, 1) for _ in range(iterations)],     # Quantum Fluctuations (Q)
    [np.pi] * iterations,                                      # Pi (pi)
    [1.618] * iterations,                                      # Phi (phi)
    [0.7] * iterations                                         # Dark Energy (Lambda)
]

forces = np.array(forces).T  # Transpose for easier iteration

# Collect results and simulate
for t in range(iterations):
    F = forces[t]  # Current forces

    # Compute feedback
    R_new = ladder.compute_feedback(R, F)

    # Update weights
    ladder.update_weights(R_new)

    # Record results
    results.append({
        "iteration": t,
        "R": R.tolist(),
        "R_new": R_new.tolist(),
        "weights": ladder.weights.tolist(),
        "forces": F.tolist()
    })

    R = R_new  # Update R for next iteration

# Visualization of dynamics
plt.figure(figsize=(10, 6))
plt.plot(
    [r["R_new"][0] for r in results],  # Extract the first component of R_new
    [r["R_new"][1] for r in results],  # Extract the second component of R_new
    '.', markersize=0.5, alpha=0.7
)
plt.title("Jacob's Ladder Attractor Dynamics")
plt.xlabel("R1")
plt.ylabel("R2")
plt.grid()
plt.show()

# Save results
import json
with open("jacobs_ladder_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Simulation complete. Results saved to 'jacobs_ladder_results.json'.")

