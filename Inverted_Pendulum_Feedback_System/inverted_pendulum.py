import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json

# Define the inverted pendulum system with recursive feedback control
def inverted_pendulum(t, y, wf, wb, gamma, g, L, m):
    """
    y[0] -> Angle theta (radians)
    y[1] -> Angular velocity omega (radians/second)
    wf, wb -> Forward and backward weights
    gamma -> Damping coefficient
    g -> Gravitational acceleration
    L -> Length of the pendulum
    m -> Mass of the pendulum
    """
    theta, omega = y

    # Torque (tau) calculated using recursive feedback
    tau = (wf * theta + wb * omega) / (wf + wb + 1e-10) - gamma * omega

    # Equations of motion
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta) - gamma * omega + tau / (m * L**2)

    return [dtheta_dt, domega_dt]

# Parameters
wf = 0.8  # Forward weight
wb = 0.2  # Backward weight
gamma = 0.1  # Damping coefficient
g = 9.81  # Gravitational acceleration (m/s^2)
L = 1.0  # Length of the pendulum (meters)
m = 1.0  # Mass of the pendulum (kg)
initial_state = [0.1, 0.0]  # Initial angle (radians) and angular velocity (rad/s)
time_span = (0, 10)  # Time range for simulation (seconds)
time_eval = np.linspace(time_span[0], time_span[1], 1000)  # Time points for evaluation

# Solve the ODE system
solution = solve_ivp(
    inverted_pendulum,
    time_span,
    initial_state,
    t_eval=time_eval,
    args=(wf, wb, gamma, g, L, m)
)

# Extract results
time = solution.t
theta = solution.y[0]  # Angle (theta)
omega = solution.y[1]  # Angular velocity (omega)

# Save results to a JSON file
results = {
    "time": time.tolist(),
    "theta": theta.tolist(),
    "omega": omega.tolist()
}
with open("inverted_pendulum_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

# Visualization: Angle over time
plt.figure(figsize=(10, 6))
plt.plot(time, theta, label="Angle \u03b8(t) (radians)", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.title("Inverted Pendulum Angle Dynamics")
plt.legend()
plt.grid()
plt.savefig("inverted_pendulum_angle.png")
plt.show()

# Visualization: Angular velocity over time
plt.figure(figsize=(10, 6))
plt.plot(time, omega, label="Angular Velocity \u03c9(t) (rad/s)", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Inverted Pendulum Angular Velocity Dynamics")
plt.legend()
plt.grid()
plt.savefig("inverted_pendulum_angular_velocity.png")
plt.show()

# Frequency domain analysis
from scipy.fft import fft, fftfreq

# Fourier Transform of theta(t)
theta_fft = fft(theta)
frequencies = fftfreq(len(theta), d=(time[1] - time[0]))

# Plot frequency domain
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(theta_fft)[:len(frequencies)//2],
         label="Magnitude of \u03b8(t) in Frequency Domain", color="green")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain Analysis of Inverted Pendulum")
plt.grid()
plt.legend()
plt.savefig("inverted_pendulum_frequency.png")
plt.show()

