import numpy as np
import math

def convergence_rate(delta_t_list):
    """Calculates the geometric decay rate of delta_t."""
    if len(delta_t_list) < 2:
        return 1 #No decay
    ratios = [delta_t_list[i+1] / (delta_t_list[i] + 1e-9)  for i in range(len(delta_t_list)-1)]
    if not ratios:
        return 1
    return np.mean(ratios)

def delta_variance(delta_t_list):
    """Measures the variance of delta_t."""
    return np.var(delta_t_list)

def final_delta(delta_t_list):
    """Measures the value of delta_t at the final state."""
    if not delta_t_list:
        return 0
    return delta_t_list[-1]

def average_entropy(state_t_list):
    """Calculates the average entropy of the system's states."""
    if not state_t_list:
        return 0

    # Normalize each state vector before calculating entropy
    entropies = []
    for state in state_t_list:
        state = state / np.sum(state)  # Normalize the state
        entropies.append(-np.sum(state * np.log(state + 1e-9)))
    return np.mean(entropies)

def final_entropy(state_t):
    """Calculates entropy of the system's final state."""
    if state_t is None or state_t.size == 0:  # Check if the array is None or empty
        return 0
    # Normalize the state to ensure it sums to 1
    state_t = state_t / np.sum(state_t)
    return -np.sum(state_t * np.log(state_t + 1e-9))

def count_distinct_states(state_t):
    """Counts the number of unique values in the final state."""
    return len(np.unique(state_t))

def response_time_to_perturbation(state_history, threshold=0.01):
    """
    Measure the response time (number of steps) for the system to return to equilibrium
    after perturbation.
    """
    if not state_history:
        return 0

    original_delta = np.diff(state_history, prepend=0)
    perturbed_state_history = state_history.copy()
    perturbed_state_history[-1] += 1

    perturbed_delta = np.diff(perturbed_state_history, prepend=0)
    for time, delta in enumerate(perturbed_delta):
        if np.all(abs(delta) < threshold):  # Ensures the entire array satisfies the condition
            return time
    return len(perturbed_delta)  # If no response has been made, return total number of iterations

def change_in_equilibrium_state(non_perturbed_state, perturbed_state):
     """Measures the absolute change in the equilibrium state after a perturbation."""
     return np.linalg.norm(non_perturbed_state - perturbed_state)

def equilibrium_score(stability_score, diversity_score, adaptability_score):
    """Combines metrics to produce an overall equilibrium score."""
    return 0.33 * stability_score + 0.33 * diversity_score + 0.33 * adaptability_score

