import numpy as np

# Main Recursive Feedback System
class RecursiveFeedbackSystem:
    def __init__(self, alpha_branches, w_f, w_b):
        self.alpha_branches = alpha_branches  # Contribution of branches
        self.w_f = w_f  # Forward weight
        self.w_b = w_b  # Backward weight

    def compute_main(self, X, X_prime, branch_outputs):
        total_alpha = sum(self.alpha_branches)
        numerator = self.w_f * X + self.w_b * X_prime + np.dot(self.alpha_branches, branch_outputs)
        denominator = self.w_f + self.w_b + total_alpha
        return numerator / denominator

# Branch Recursive Feedback System
class Branch:
    def __init__(self, w_f_branch, w_b_branch):
        self.w_f_branch = w_f_branch  # Forward weight for branch
        self.w_b_branch = w_b_branch  # Backward weight for branch

    def compute_branch(self, X_branch, X_prime_branch):
        numerator = self.w_f_branch * X_branch + self.w_b_branch * X_prime_branch
        denominator = self.w_f_branch + self.w_b_branch
        return numerator / denominator

# Example Usage
if __name__ == "__main__":
    # Main system inputs
    X_main = 10  # Forward input for the main system
    X_prime_main = 8  # Backward input for the main system

    # Branch inputs
    branch_inputs = [(6, 4), (5, 7), (9, 3)]  # List of (X, X_prime) for branches

    # Branch weights
    branch_weights = [(0.7, 0.3), (0.6, 0.4), (0.8, 0.2)]  # (w_f_branch, w_b_branch) for branches

    # Initialize branches
    branches = [Branch(w_f, w_b) for w_f, w_b in branch_weights]

    # Compute branch outputs
    branch_outputs = []
    for i, (X_branch, X_prime_branch) in enumerate(branch_inputs):
        branch_output = branches[i].compute_branch(X_branch, X_prime_branch)
        branch_outputs.append(branch_output)

    # Initialize main system with branch contributions
    alpha_branches = [0.5, 0.3, 0.2]  # Contribution weights for branches
    main_system = RecursiveFeedbackSystem(alpha_branches, w_f=0.9, w_b=0.1)

    # Compute main system output
    R_main = main_system.compute_main(X_main, X_prime_main, branch_outputs)

    # Display results
    print("Branch Outputs:", branch_outputs)
    print("Main System Output:", R_main)

