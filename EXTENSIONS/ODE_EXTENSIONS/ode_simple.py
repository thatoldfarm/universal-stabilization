import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json

# Define the coupled ODEs for the recursive feedback system
def feedback_system(t, y, wf, wb, gamma):
    """
    y[0] -> State variable R(t)
    y[1] -> Auxiliary variable X(t)
    wf, wb -> Forward and backward weights
    gamma -> Damping coefficient
    """
    R, X = y
    X_prime = -gamma * X + np.sin(t)  # Example dynamic for X(t)

    # Recursive feedback core equation
    dR_dt = (wf * X + wb * X_prime) / (wf + wb + 1e-10) - gamma * R

    return [dR_dt, X_prime]

# Parameters
wf = 0.8  # Forward weight
wb = 0.2  # Backward weight
gamma = 0.1  # Damping coefficient
initial_state = [1.0, 0.0]  # Initial conditions for R(t) and X(t)
time_span = (0, 50)  # Time range for simulation
time_eval = np.linspace(time_span[0], time_span[1], 1000)  # Time points for evaluation

# Solve the ODE system
solution = solve_ivp(feedback_system, time_span, initial_state, t_eval=time_eval, args=(wf, wb, gamma))

# Extract results
time = solution.t
R = solution.y[0]  # State variable R(t)
X = solution.y[1]  # Auxiliary variable X(t)

# Save results to JSON
results = {
    "time": time.tolist(),
    "R": R.tolist(),
    "X": X.tolist()
}

with open("recursive_feedback_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(time, R, label="State R(t)", color="blue")
plt.plot(time, X, label="Auxiliary X(t)", color="orange")
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Recursive Feedback System Dynamics")
plt.legend()
plt.grid()
plt.savefig("recursive_feedback_dynamics.png")
plt.show()

# Frequency domain analysis
from scipy.fft import fft, fftfreq

# Fourier Transform of R(t)
R_fft = fft(R)
frequencies = fftfreq(len(R), d=(time[1] - time[0]))

# Plot frequency domain
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(R_fft)[:len(frequencies)//2], label="Magnitude of R(t) in Frequency Domain")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency Domain Analysis of Recursive Feedback System")
plt.grid()
plt.legend()
plt.savefig("recursive_feedback_frequency.png")
plt.show()

