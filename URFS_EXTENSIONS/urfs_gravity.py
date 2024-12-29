import numpy as np

# Simulate gravitational influence in a recursive feedback system
class GravitationalFeedbackSystem:
    def __init__(self, w_f, w_b):
        self.w_f = w_f  # Forward weight
        self.w_b = w_b  # Backward weight

    def gravitational_term(self, m1, m2, r):
        if r == 0:  # Prevent division by zero
            r = 1e-6
        return (m1 * m2) / r**2

    def compute_recursive_feedback(self, X, X_prime, masses, separations):
        # Calculate gravitational influences
        gravitational_influences = [
            self.gravitational_term(m1, m2, r)
            for (m1, m2), r in zip(masses, separations)
        ]
        
        # Aggregate gravitational effects
        total_gravitational_effect = sum(gravitational_influences)

        # Recursive feedback equation with gravitational term
        numerator = self.w_f * X + self.w_b * X_prime + total_gravitational_effect
        denominator = self.w_f + self.w_b + len(gravitational_influences)
        return numerator / denominator

# Example Usage
if __name__ == "__main__":
    # Initialize the system
    feedback_system = GravitationalFeedbackSystem(w_f=0.9, w_b=0.1)

    # Inputs
    X = 10  # Forward input
    X_prime = 8  # Backward input

    # Masses and separations
    masses = [(2, 3), (4, 5), (1, 2)]  # Pairs of masses (m1, m2)
    separations = [2, 3, 1.5]  # Distances (r)

    # Compute the recursive feedback with gravitational influence
    R = feedback_system.compute_recursive_feedback(X, X_prime, masses, separations)

    # Display results
    print("Gravitational Influences:", [
        feedback_system.gravitational_term(m1, m2, r)
        for (m1, m2), r in zip(masses, separations)
    ])
    print("Recursive Feedback Output:", R)

