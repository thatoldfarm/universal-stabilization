import numpy as np
import json

# Gravity-Time Recursive Feedback System
class GravityTimeFeedback:
    def __init__(self, learning_rate=0.05):
        self.weights = np.ones(8) * 0.5  # Initialize weights equally
        self.learning_rate = learning_rate  # Rate at which weights evolve

    def compute_feedback(self, R, gravity, time):
        # Compute feedback for each node
        R_new = np.zeros(4)
        R_new[0] = (self.weights[0] + self.weights[1]) / (
            (self.weights[0] * R[1] + self.weights[1] * R[3] + self.weights[0] * gravity) /
            (self.weights[0] + self.weights[1] + self.weights[0])
        )
        R_new[1] = (self.weights[2] + self.weights[3]) / (
            (self.weights[2] * R[2] + self.weights[3] * R[0] + self.weights[2] * gravity) /
            (self.weights[2] + self.weights[3] + self.weights[2])
        )
        R_new[2] = (self.weights[4] + self.weights[5]) / (
            (self.weights[4] * R[3] + self.weights[5] * R[1] + self.weights[4] * time) /
            (self.weights[4] + self.weights[5] + self.weights[4])
        )
        R_new[3] = (self.weights[6] + self.weights[7]) / (
            (self.weights[6] * R[0] + self.weights[7] * R[2] + self.weights[6] * time) /
            (self.weights[6] + self.weights[7] + self.weights[6])
        )
        return R_new

    def update_weights(self, R):
        # Adjust weights based on feedback outputs
        self.weights[:4] += self.learning_rate * R
        self.weights[4:] += self.learning_rate * (1 - R)
        # Ensure weights remain positive
        self.weights = np.maximum(self.weights, 0.01)

if __name__ == "__main__":
    # Initialize the system
    system = GravityTimeFeedback(learning_rate=0.05)

    # Initial outputs
    R = np.array([1.0, 0.9, 0.8, 0.7])

    # Nonlinear Inputs
    iterations = 20
    results = []

    for t in range(iterations):
        # Oscillating gravity and exponential time inputs
        gravity = 0.5 * np.sin(0.2 * t) + 0.5  # Oscillates between 0 and 1
        time = 0.2 * np.exp(0.1 * t)  # Exponential growth

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

        # Update R for the next iteration
        R = R_new

    # Save results to a JSON file
    with open("gravity_time_feedback_nonlinear_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Simulation complete with nonlinear inputs. Results saved to 'gravity_time_feedback_nonlinear_results.json'.")

