import numpy as np
import matplotlib.pyplot as plt
import json

# Gravitational Feedback System Class
class GravitationalFeedbackSystem:
    def __init__(self, w_f, w_b):
        self.w_f = w_f  # Forward weight
        self.w_b = w_b  # Backward weight

    def gravitational_term(self, m1, m2, r):
        if r == 0:  # Prevent division by zero
            r = 1e-6
        return (m1 * m2) / r**2

    def compute_recursive_feedback(self, X, X_prime, masses, separations):
        gravitational_influences = [
            self.gravitational_term(m1, m2, r)
            for (m1, m2), r in zip(masses, separations)
        ]
        total_gravitational_effect = sum(gravitational_influences)
        numerator = self.w_f * X + self.w_b * X_prime + total_gravitational_effect
        denominator = self.w_f + self.w_b + len(gravitational_influences)
        return numerator / denominator

# Initialize the system
feedback_system = GravitationalFeedbackSystem(w_f=0.9, w_b=0.1)

# Define parameter ranges for analysis
X_values = np.linspace(5, 15, 50)  # Forward input range
X_prime_values = np.linspace(5, 15, 50)  # Backward input range
masses_list = [[(2, 3), (4, 5), (1, 2)], [(1, 1), (2, 2), (3, 3)]]  # Different mass pairs
separations_list = [[2, 3, 1.5], [1, 2, 3]]  # Different separations

# Store results for visualization and JSON output
results = []
for masses, separations in zip(masses_list, separations_list):
    feedback_outputs = []
    for X, X_prime in zip(X_values, X_prime_values):
        R = feedback_system.compute_recursive_feedback(X, X_prime, masses, separations)
        feedback_outputs.append(R)
    results.append({
        "masses": masses,
        "separations": separations,
        "feedback_outputs": feedback_outputs
    })

# Save results to a JSON file
with open("gravity_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Plot the results
plt.figure(figsize=(12, 6))
for i, result in enumerate(results):
    feedback_outputs = result["feedback_outputs"]
    masses = result["masses"]
    separations = result["separations"]
    plt.plot(X_values, feedback_outputs, label=f"Masses: {masses}, Separations: {separations}")
plt.xlabel("Forward Input (X)")
plt.ylabel("Recursive Feedback Output (R)")
plt.title("Impact of Gravitational Influence on Recursive Feedback System")
plt.legend()
plt.grid(True)
plt.show()

