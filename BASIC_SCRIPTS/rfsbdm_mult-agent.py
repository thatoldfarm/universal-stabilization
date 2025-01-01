import numpy as np
import json


def extended_recursive_feedback(
    forward_inputs, backward_inputs, forward_weights, backward_weights, steps=20
):
    """
    Extended recursive feedback system for multiple agents or subsystems.

    Parameters:
    - forward_inputs: List of arrays, each representing forward inputs from an agent.
    - backward_inputs: List of arrays, each representing backward inputs from an agent.
    - forward_weights: Initial weights for forward contributions (one per agent).
    - backward_weights: Initial weights for backward contributions (one per agent).
    - steps: Number of recursive steps for stabilization.

    Returns:
    - stabilized_results: Final stabilized output after all iterations.
    - deltas: Convergence metric (Delta_t) over iterations.
    """
    n_agents = len(forward_inputs)
    assert n_agents == len(backward_inputs), "Mismatch in forward and backward agents"
    assert n_agents == len(forward_weights) == len(backward_weights), "Mismatch in weights"

    # Initialize arrays to track results and deltas
    stabilized_results = []
    deltas = []

    # Initialize previous results as zeros
    prev_result = np.zeros_like(forward_inputs[0])

    for step in range(steps):
        # Calculate weighted contributions for forward and backward inputs
        weighted_forward = sum(wf * fwd for wf, fwd in zip(forward_weights, forward_inputs))
        weighted_backward = sum(wb * bwd for wb, bwd in zip(backward_weights, backward_inputs))
        total_weights = sum(forward_weights) + sum(backward_weights)

        # Compute the stabilized result
        current_result = (weighted_forward + weighted_backward) / total_weights
        stabilized_results.append(current_result.tolist())

        # Calculate convergence metric (Delta_t)
        delta = np.linalg.norm(current_result - prev_result)
        deltas.append(delta)
        prev_result = current_result

        # Update weights dynamically (example: based on variance of current result)
        forward_weights = [1 / (1 + np.var(current_result)) for _ in forward_weights]
        backward_weights = [1 - wf for wf in forward_weights]

    return stabilized_results, deltas


# Example Usage
if __name__ == "__main__":
    # Simulate data for 3 agents
    np.random.seed(42)
    n_agents = 3
    data_length = 10

    forward_inputs = [np.random.rand(data_length) for _ in range(n_agents)]
    backward_inputs = [np.random.rand(data_length) for _ in range(n_agents)]

    # Initialize weights for each agent
    forward_weights = [1.0] * n_agents
    backward_weights = [1.0] * n_agents

    # Run the extended recursive feedback system
    stabilized_results, deltas = extended_recursive_feedback(
        forward_inputs, backward_inputs, forward_weights, backward_weights, steps=20
    )

    # Output results to a JSON file
    output_data = {
        "final_stabilized_result": stabilized_results[-1],
        "all_stabilized_results": stabilized_results,
        "convergence_deltas": deltas,
    }

    output_filename = "extended_recursive_feedback_results.json"
    with open(output_filename, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Results saved to {output_filename}")

    # Plot convergence metrics
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(deltas)), deltas, marker="o", label="Delta_t (Convergence)")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Delta_t")
    plt.title("Convergence Over Iterations")
    plt.legend()
    plt.grid()
    plt.show()

