# Inverted Pendulum Dynamics with Recursive Feedback

This project implements a computational model to simulate the dynamics of an inverted pendulum using a recursive feedback system. The script leverages differential equations to describe the system's motion and stability. Key outputs include the angle dynamics, angular velocity, and a frequency domain analysis, stored in JSON format and visualized as PNG plots.

---

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [Recursive Feedback](#recursive-feedback)
  - [System Dynamics](#system-dynamics)
- [Formulas Used](#formulas-used)
  - [Angle Dynamics](#angle-dynamics)
  - [Angular Velocity](#angular-velocity)
  - [Recursive Feedback Formula](#recursive-feedback-formula)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
  - [PNG Plots](#png-plots)
  - [JSON File](#json-file)
- [Applications](#applications)
- [License](#license)

---

## Introduction

The inverted pendulum is a classic problem in physics and control theory, often used to study stability, feedback, and dynamic systems. This script models the pendulum using a recursive feedback system to balance it effectively.

---

## Core Concepts

### Recursive Feedback
Recursive feedback ensures that the system continuously adjusts based on its previous state and the forces acting upon it. This approach stabilizes the pendulum by dynamically calculating the required torque or force.

### System Dynamics
The inverted pendulum dynamics are modeled using second-order ordinary differential equations (ODEs). These equations capture the interplay between angle, angular velocity, and external forces.

---

## Formulas Used

### Angle Dynamics
The angle of the pendulum, \( \theta(t) \), changes over time due to angular velocity:

\[
\frac{d\theta}{dt} = \omega(t)
\]

### Angular Velocity
The angular velocity, \( \omega(t) \), evolves due to the forces acting on the pendulum:

\[
\frac{d\omega}{dt} = \frac{-g}{L} \sin(\theta) - \gamma \omega + F_{input}
\]

Where:
- \( g \): Gravitational acceleration
- \( L \): Length of the pendulum
- \( \gamma \): Damping coefficient
- \( F_{input} \): External input force (e.g., control force)

### Recursive Feedback Formula
The recursive feedback formula combines forward and backward influences to stabilize the system:

\[
F_{feedback} = \frac{w_f \cdot \theta(t) + w_b \cdot \omega(t)}{w_f + w_b + \epsilon}
\]

Where:
- \( w_f \): Forward weight
- \( w_b \): Backward weight
- \( \epsilon \): Small constant to prevent division by zero

This feedback force is used to adjust the pendulum's dynamics iteratively.

---

## Implementation Details

The script integrates the system's differential equations over time using the `scipy.integrate.solve_ivp` library. It outputs:
- Time-series data for \( \theta(t) \) (angle) and \( \omega(t) \) (angular velocity)
- Frequency domain analysis via Fourier Transform

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thatoldfarm/universal-stabilization
   cd universal-stabilization
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```

---

## Usage

Run the script with:
```bash
python inverted_pendulum.py
```

---

## Outputs

### PNG Plots

1. **Angle Dynamics**: \( \theta(t) \) over time.
2. **Angular Velocity Dynamics**: \( \omega(t) \) over time.
3. **Frequency Domain Analysis**: Magnitude of \( \theta(t) \) in the frequency domain.

### JSON File
The script generates a `pendulum_results.json` file containing:
- Time-series data for \( \theta(t) \) and \( \omega(t) \).
- Key parameters used in the simulation.

---

## Applications

This implementation has applications in:
- Robotics and control systems
- Physics and dynamics research
- Machine learning for dynamic systems

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

