# README: Recursive Feedback System with Coupled ODEs

## Overview
This project explores a recursive feedback system modeled by coupled ordinary differential equations (ODEs). The system demonstrates stabilization, symmetry, and convergence across dynamic states, highlighting its potential applications in physics, control theory, and complex systems analysis.

The core principle behind the system is to combine forward and backward influences through a recursive feedback mechanism. By coupling state variables with their derivatives and incorporating damping effects, the system achieves stabilization over time.

## Key Concepts

### Recursive Feedback Equation
The state variable \( R(t) \) is influenced by an auxiliary variable \( X(t) \) and its derivative \( X'(t) \), with weights assigned to forward (\( w_f \)) and backward (\( w_b \)) influences:

\[
\frac{dR}{dt} = \frac{w_f \cdot X(t) + w_b \cdot X'(t)}{w_f + w_b + \epsilon} - \gamma \cdot R(t)
\]

Where:
- \( R(t) \): Primary state variable
- \( X(t) \): Auxiliary variable
- \( X'(t) \): Derivative of the auxiliary variable
- \( w_f \): Forward weight
- \( w_b \): Backward weight
- \( \gamma \): Damping coefficient
- \( \epsilon \): Small constant to avoid division by zero

The auxiliary variable \( X(t) \) evolves independently:

\[
X'(t) = -\gamma \cdot X + \sin(t)
\]

### Goals of the System
1. **Stabilization**: Achieve a balanced state over time despite initial oscillations.
2. **Self-Similarity**: Represent universal dynamics applicable to various fields.
3. **Frequency Domain Insights**: Analyze dominant oscillatory components using Fourier transforms.

## Implementation

### Script Breakdown
The system is implemented in Python using the `scipy.integrate.solve_ivp` function to solve the coupled ODEs. Here's an overview of the key components:

#### 1. Feedback System ODE
The `feedback_system` function defines the coupled ODEs for \( R(t) \) and \( X(t) \):
```python
# Define the coupled ODEs for the recursive feedback system
def feedback_system(t, y, wf, wb, gamma):
    R, X = y
    X_prime = -gamma * X + np.sin(t)  # Auxiliary variable dynamics

    # Recursive feedback core equation
    dR_dt = (wf * X + wb * X_prime) / (wf + wb + 1e-10) - gamma * R

    return [dR_dt, X_prime]
```

#### 2. Parameters and Initial Conditions
Key parameters for the simulation include:
- `wf = 0.8`: Forward weight
- `wb = 0.2`: Backward weight
- `gamma = 0.1`: Damping coefficient
- `initial_state = [1.0, 0.0]`: Initial values for \( R(t) \) and \( X(t) \)
- `time_span = (0, 50)`: Time range for the simulation

#### 3. Solving the ODEs
The `solve_ivp` function integrates the system over the specified time span:
```python
solution = solve_ivp(feedback_system, time_span, initial_state, t_eval=time_eval, args=(wf, wb, gamma))
```
The solution contains time points and the corresponding values of \( R(t) \) and \( X(t) \).

#### 4. Visualization
Two plots are generated to illustrate the system's dynamics:

1. **Time Domain Dynamics**: Shows the evolution of \( R(t) \) and \( X(t) \) over time.
2. **Frequency Domain Analysis**: Uses Fourier transforms to analyze the dominant frequencies in \( R(t) \).

#### 5. Data Export
Results are written to a JSON file for further analysis or integration:
```python
# Save results to JSON
import json
results = {
    "time": time.tolist(),
    "R": R.tolist(),
    "X": X.tolist()
}
with open("recursive_feedback_results.json", "w") as f:
    json.dump(results, f, indent=4)
```

## Results

### Dynamics
The time-domain analysis reveals:
- Initial oscillations in \( R(t) \) and \( X(t) \).
- Gradual stabilization as the damping term balances the feedback loop.

### Frequency Domain
The Fourier transform highlights:
- Dominant low-frequency components in \( R(t) \).
- Rapid decay of higher-frequency components due to damping.

## Applications
The recursive feedback system has broad implications across multiple fields:
1. **Physics**: Modeling oscillatory systems with damping.
2. **Control Theory**: Stabilizing feedback loops.
3. **Economics**: Balancing dynamic variables in complex markets.
4. **Artificial Intelligence**: Reinforcement learning with recursive feedback mechanisms.

## How to Run
1. Install dependencies:
```bash
pip install numpy scipy matplotlib
```

2. Run the script:
```bash
python recursive_feedback_ode.py
```

3. View the generated plots (`recursive_feedback_dynamics.png` and `recursive_feedback_frequency.png`) and analyze the JSON results in `recursive_feedback_results.json`.

## Future Work
- Extend to partial differential equations (PDEs) for spatially dependent systems.
- Incorporate more complex feedback mechanisms.
- Explore real-world applications in engineering and computational biology.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

---

This README provides a comprehensive overview of the recursive feedback system, its theoretical foundation, and its implementation. Let me know if you'd like to refine or expand it further!


