import numpy as np
import matplotlib.pyplot as plt

class RecursiveFeedbackSystem:
    def __init__(self, forward_data, backward_data, w_f_init=1.0, w_b_init=1.0):
        """
        Initialize the recursive feedback system.
        
        Parameters:
        forward_data: array-like, forward sequence
        backward_data: array-like, backward sequence
        w_f_init: float, initial forward weight
        w_b_init: float, initial backward weight
        """
        self.forward_data = np.array(forward_data)
        self.backward_data = np.array(backward_data)
        self.w_f = w_f_init
        self.w_b = w_b_init
        self.history = []
        
    def recursive_transform(self):
        """Perform one step of recursive transformation."""
        denominator = self.w_f + self.w_b
        R_t = (self.w_f * self.forward_data + self.w_b * self.backward_data) / denominator
        self.history.append(R_t)
        return R_t
    
    def update_weights(self, R_t):
        """Update weights based on summary statistics of R_t."""
        # Example weight update rules using variance
        variance = np.var(R_t)
        self.w_f = 1.0 / (1.0 + variance)  # Decreases weight when variance is high
        self.w_b = 1.0 - self.w_f  # Ensures weights sum to 1
    
    def run_simulation(self, n_steps):
        """Run the recursive feedback system for n steps."""
        convergence_metrics = []
        
        for _ in range(n_steps):
            R_t = self.recursive_transform()
            self.update_weights(R_t)
            
            # Calculate convergence metric
            if len(self.history) > 1:
                delta = np.max(np.abs(self.history[-1] - self.history[-2]))
                convergence_metrics.append(delta)
        
        return np.array(self.history), np.array(convergence_metrics)
    
    def plot_results(self, history, convergence_metrics):
        """Plot the evolution of the system and convergence metrics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot system evolution
        for i in range(len(self.forward_data)):
            ax1.plot([h[i] for h in history], label=f'Element {i}')
        ax1.set_title('Evolution of Recursive Transformation')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        
        # Plot convergence metrics
        ax2.plot(convergence_metrics)
        ax2.set_title('Convergence Metrics')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Delta')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Example with simple numerical sequence
    forward_seq = [1.0, 2.0, 3.0, 4.0]
    backward_seq = [4.0, 3.0, 2.0, 1.0]
    
    system = RecursiveFeedbackSystem(forward_seq, backward_seq)
    history, convergence = system.run_simulation(n_steps=20)
    system.plot_results(history, convergence)
    plt.show()
