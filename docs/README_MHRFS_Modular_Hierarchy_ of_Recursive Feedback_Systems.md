# Modular Hierarchy of Recursive Feedback Systems (MHRFS) Engine

## Overview
The MHRFS Engine implements a **Modular Hierarchy of Recursive Feedback Systems** with a **Meta-Layer** to integrate and stabilize outputs from independent systems. Each system (e.g., Energy, Gravity) uses a core universal equation for achieving stabilization and symmetry within its respective domain, and their outputs are integrated in a higher-level Meta-Layer.

This implementation is a demonstration of recursive feedback systems applied to a variety of domains, leveraging shared mathematical principles for achieving boundedness, convergence, and symmetry across seemingly disparate fields.

---

## Features
- **Independent Systems**: Implements modular recursive feedback systems for Energy, Gravity, and other domains.
- **Meta-Layer Integration**: Aggregates stabilized outputs from individual systems into a cohesive state.
- **Visualization**: Plots the history of each system and the Meta-Layer's dynamics.
- **Stabilization**: Prevents numerical overflows and achieves bounded outputs.

---

## Key Components

### Recursive Feedback System
The base class for each subsystem. Subclasses (e.g., `EnergySystem`, `GravitySystem`) inherit and define domain-specific behaviors.

#### Attributes
- `state`: Current state of the system.
- `phase`: Current phase of the system.
- `history`: List storing the historical states of the system.

#### Methods
- `update(t)`: Abstract method to define system-specific updates.
- `stabilize()`: Returns the mean value of the system's historical states as its stabilized output.

---

### Meta-Layer
The Meta-Layer integrates the stabilized outputs of all subsystems into a higher-level state.

#### Attributes
- `systems`: A list of subsystems (e.g., `EnergySystem`, `GravitySystem`).
- `meta_state`: Current state of the Meta-Layer.
- `meta_history`: List storing the historical states of the Meta-Layer.

#### Methods
- `integrate()`: Computes the weighted average of the stabilized outputs of all subsystems.
- `run(iterations)`: Runs all subsystems for a specified number of iterations and updates the Meta-Layer.
- `visualize()`: Plots the dynamics of each subsystem and the Meta-Layer.

---

## Mathematical Formulas

### Energy System
The state of the `EnergySystem` evolves according to:

\[
R_E(t) = \tanh(0.5 \cdot R_E(t-1)^2 + 9.8 \cdot R_E(t-1))
\]

Where:
- \( R_E(t) \): State of the Energy system at time \( t \).
- \( \tanh \): Ensures boundedness.
- **Stabilization**: The output is clipped to the range \([-10^6, 10^6]\).

### Gravity System
The state of the `GravitySystem` evolves according to:

\[
R_G(t) = \frac{G}{R_G(t-1) + \epsilon}
\]

Where:
- \( R_G(t) \): State of the Gravity system at time \( t \).
- \( G = 6.67430 \times 10^{-11} \): Gravitational constant.
- \( \epsilon \): Small value to avoid division by zero.

### Meta-Layer Integration
The Meta-Layer integrates stabilized outputs from individual systems using:

\[
R_M(t) = \frac{\sum_{i=1}^N w_i \cdot S_i(t)}{\sum_{i=1}^N w_i}
\]

Where:
- \( R_M(t) \): State of the Meta-Layer at time \( t \).
- \( w_i \): Weight for subsystem \( i \).
- \( S_i(t) \): Stabilized output of subsystem \( i \).

---

## How It Works
1. **Initialize Systems**: Each system starts with a random initial state and evolves according to its update equation.
2. **Run Simulation**: For each time step:
   - Update each system based on its recursive feedback.
   - Integrate the stabilized outputs into the Meta-Layer.
3. **Visualize Results**: Plot the dynamics of each system and the Meta-Layer.

---

## Visualization
The generated visualization includes:
- **Individual System Dynamics**: Tracks the evolution of each system's state over time.
- **Meta-Layer Dynamics**: Shows the aggregated state of the Meta-Layer over time.

---

## File Outputs
1. **JSON Results**: 
   - Stores the historical states of each system and the Meta-Layer.
   - File: `mhrfs_results.json`.
2. **Visualization**: 
   - A line plot showing system and Meta-Layer dynamics.
   - File: `mhrfs_dynamics.png`.

---

## Usage
### Running the Script
```bash
python mhrfs_engine.py
```

### Adding New Systems
To add a new system:
1. Define a subclass of `RecursiveFeedbackSystem`.
2. Implement the `update(t)` method with the system's specific equation.
3. Add an instance of the system to the `systems` list in the main script.

---

## Example Output
The Meta-Layer stabilizes as subsystems independently achieve convergence, producing a cohesive view of the system's dynamics.

---

## Applications
- **Physics**: Unifies the dynamics of fundamental forces.
- **Economics**: Models market equilibria using recursive feedback.
- **Artificial Intelligence**: Stabilizes learning systems with modular recursive layers.
- **Complex Systems**: Provides insights into the interplay of interdependent subsystems.

---

## Next Steps
- Expand the library of subsystems (e.g., Thermodynamics, Electromagnetism).
- Explore applications in real-world domains like climate modeling or financial markets.
- Integrate stochastic elements for simulating real-world noise.

---

## Acknowledgments
This work leverages recursive feedback principles to unify disparate domains, contributing to the broader understanding of stabilization in complex systems.

