import numpy as np
import json

# Double-Coupled Recursive Feedback System
class DoubleCoupledFeedback:
    def __init__(self, w_f, w_b, learning_rate=0.1):
        self.w_f = w_f  # Forward weight
        self.w_b = w_b  # Backward weight
        self.learning_rate = learning_rate  # Rate at which weights evolve

    def compute_feedback(self, X, X_prime):
        numerator = self.w_f + self.w_b
        denominator = (self.w_f * X + self.w_b * X_prime) / (self.w_f + self.w_b)
        return numerator / denominator

    def update_weights(self, feedback_output):
        # Adjust weights based on feedback output
        self.w_f += self.learning_rate * feedback_output
        self.w_b += self.learning_rate * (1 - feedback_output)
        # Ensure weights remain positive
        self.w_f = max(self.w_f, 0.01)
        self.w_b = max(self.w_b, 0.01)

# Example usage
if __name__ == "__main__":
    # Initialize the system
    feedback_system = DoubleCoupledFeedback(w_f=0.9, w_b=0.1, learning_rate=0.05)

    # Inputs
    X_values = np.linspace(5, 15, 10)
    X_prime_values = np.linspace(5, 15, 10)
    results = []

    # Test with evolving weights
    for X, X_prime in zip(X_values, X_prime_values):
        R = feedback_system.compute_feedback(X, X_prime)
        feedback_system.update_weights(R)  # Update weights based on feedback
        results.append({
            "X": X,
            "X_prime": X_prime,
            "Feedback_Output": R,
            "Updated_w_f": feedback_system.w_f,
            "Updated_w_b": feedback_system.w_b
        })

    print("Feedback Outputs with Evolving Weights:", results)

    # Save results to a JSON file
    with open("double_coupled_feedback_with_weights_results.json", "w") as f:
        json.dump(results, f, indent=4)

