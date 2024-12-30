import numpy as np
import json

# Double-Coupled Recursive Feedback System
class DoubleCoupledFeedback:
    def __init__(self, w_f, w_b):
        self.w_f = w_f  # Forward weight
        self.w_b = w_b  # Backward weight

    def compute_feedback(self, X, X_prime):
        numerator = self.w_f + self.w_b
        denominator = (self.w_f * X + self.w_b * X_prime) / (self.w_f + self.w_b)
        return numerator / denominator

# Example usage
if __name__ == "__main__":
    # Initialize the system
    feedback_system = DoubleCoupledFeedback(w_f=0.9, w_b=0.1)

    # Inputs
    X = 10  # Forward input
    X_prime = 8  # Backward input

    # Compute the double-coupled feedback
    R = feedback_system.compute_feedback(X, X_prime)

    # Output the result
    print("Double-Coupled Recursive Feedback Output:", R)

    # Test with multiple values
    X_values = np.linspace(5, 15, 10)
    X_prime_values = np.linspace(5, 15, 10)
    results = []

    for X, X_prime in zip(X_values, X_prime_values):
        R = feedback_system.compute_feedback(X, X_prime)
        results.append({"X": X, "X_prime": X_prime, "Feedback_Output": R})

    print("Feedback Outputs for Range of Inputs:", results)

    # Save results to a JSON file
    with open("double_coupled_feedback_results.json", "w") as f:
        json.dump(results, f, indent=4)

