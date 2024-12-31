import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json

# Define dynamic diodes with different types
def forward_diode(t, tau):
    """ Forward-only diode """
    return 1  # Always allow forward flow

def reverse_diode(t, tau):
    """ Reverse-only diode """
    return -1  # Always allow reverse flow

def blocking_diode(t, tau, stability_metric=None):
    """ Blocking diode """
    if stability_metric is not None and stability_metric > 0.5:
        return 0  # Block feedback if instability is high
    return 1  # Default to forward flow

def time_gated_diode(t, tau):
    """ Time-gated diode """
    return 1 if (t % tau) < (tau / 2) else -1

# Define the coupled ODEs for the ARFS with multiple diode types
def feedback_system(t, y, wf, wb, gamma, tau, diode_type):
    """
    y[0] -> State variable R(t)
    y[1] -> Auxiliary variable X(t)
    wf, wb -> Forward and backward weights
    gamma -> Damping coefficient
    tau -> Time period for diode switching
    diode_type -> Type of diode to use
    """
    R, X = y
    X_prime = -gamma * X + np.sin(t)  # Example dynamic for X(t)

    # Determine diode state based on type
    if diode_type == "forward":
        D_t = forward_diode(t, tau)
    elif diode_type == "reverse":
        D_t = reverse_diode(t, tau)
    elif diode_type == "blocking":
        stability_metric = abs(wf * X - wb * X_prime) / (wf + wb + 1e-10)
        D_t = blocking_diode(t, tau, stability_metric)
    elif diode_type == "time_gated":
        D_t = time_gated_diode(t, tau)
    else:
        raise ValueError("Invalid diode type")

    # ARFS core equation with diode control
    dR_dt = (D_t * wf * X + (-D_t) * wb * X_prime) / (abs(D_t) * (wf + wb) + 1e-10) - gamma * R

    return [dR_dt, X_prime]

# Parameters
wf = 0.8  # Forward weight
wb = 0.2  # Backward weight
gamma = 0.1  # Damping coefficient
tau = 10  # Switching period for the diode
diodes = ["forward", "reverse", "blocking", "time_gated"]
initial_state = [1.0, 0.0]  # Initial conditions for R(t) and X(t)
time_span = (0, 50)  # Time range for simulation
time_eval = np.linspace(time_span[0], time_span[1], 1000)  # Time points for evaluation

# Store results for all diode types
all_results = {}

for diode in diodes:
    # Solve the ODE system
    solution = solve_ivp(feedback_system, time_span, initial_state, t_eval=time_eval, args=(wf, wb, gamma, tau, diode))

    # Extract results
    time = solution.t
    R = solution.y[0]  # State variable R(t)
    X = solution.y[1]  # Auxiliary variable X(t)

    # Save results
    all_results[diode] = {
        "time": time.tolist(),
        "R": R.tolist(),
        "X": X.tolist()
    }

    # Visualization of R(t) and X(t) for each diode
    plt.figure(figsize=(10, 6))
    plt.plot(time, R, label=f"State R(t) - {diode} diode", color="blue")
    plt.plot(time, X, label=f"Auxiliary X(t) - {diode} diode", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title(f"ARFS with {diode.capitalize()} Diode: Time Domain")
    plt.legend()
    plt.grid()
    plt.savefig(f"arfs_{diode}_diode_time.png")
    plt.show()

# Save all results to JSON
with open("arfs_all_diode_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Frequency domain analysis for each diode type
for diode, results in all_results.items():
    R = np.array(results["R"])
    time = np.array(results["time"])

    # Fourier Transform of R(t)
    R_fft = np.fft.fft(R)
    frequencies = np.fft.fftfreq(len(R), d=(time[1] - time[0]))

    # Plot frequency domain
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[:len(frequencies)//2], np.abs(R_fft)[:len(frequencies)//2], label=f"Magnitude of R(t) - {diode} diode")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title(f"Frequency Domain Analysis of ARFS with {diode.capitalize()} Diode")
    plt.grid()
    plt.legend()
    plt.savefig(f"arfs_{diode}_diode_frequency.png")
    plt.show()

