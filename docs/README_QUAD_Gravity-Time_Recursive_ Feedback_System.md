# Gravity-Time Recursive Feedback System

The **Gravity-Time Recursive Feedback System** expands the recursive feedback framework by introducing **two distinct inputs**: gravity and time. These inputs influence different branches of a four-leaf clover-shaped recursive system, demonstrating the interplay between physical and temporal dynamics.

## **Overview**
This system models the recursive interactions of four nodes (or equations) interconnected in a circular structure, where:
- **Gravity (\(X_1\))** influences nodes 1 and 2.
- **Time (\(X_2\))** influences nodes 3 and 4.

Each node computes a feedback output based on its neighbors, dynamically evolving weights, and the respective input.

## **Core Equations**
The recursive feedback for each node is calculated as follows:

### Node 1:
\[
R_t^{(1)} = \frac{w_{1,t} + w_{2,t}}{\frac{w_{1,t} \cdot R_t^{(2)} + w_{2,t} \cdot R_t^{(4)} + w_{1,t} \cdot X_1}{w_{1,t} + w_{2,t} + w_{1,t}}}
\]

### Node 2:
\[
R_t^{(2)} = \frac{w_{3,t} + w_{4,t}}{\frac{w_{3,t} \cdot R_t^{(3)} + w_{4,t} \cdot R_t^{(1)} + w_{3,t} \cdot X_1}{w_{3,t} + w_{4,t} + w_{3,t}}}
\]

### Node 3:
\[
R_t^{(3)} = \frac{w_{5,t} + w_{6,t}}{\frac{w_{5,t} \cdot R_t^{(4)} + w_{6,t} \cdot R_t^{(2)} + w_{5,t} \cdot X_2}{w_{5,t} + w_{6,t} + w_{5,t}}}
\]

### Node 4:
\[
R_t^{(4)} = \frac{w_{7,t} + w_{8,t}}{\frac{w_{7,t} \cdot R_t^{(1)} + w_{8,t} \cdot R_t^{(3)} + w_{7,t} \cdot X_2}{w_{7,t} + w_{8,t} + w_{7,t}}}
\]

## **Weight Evolution**
Weights are dynamically updated after each iteration based on the feedback outputs:

\[
w_{i,t+1} = w_{i,t} + \eta \cdot R_t(i)
\]
\[
w_{j,t+1} = w_{j,t} + \eta \cdot (1 - R_t(i))
\]

Where \(\eta\) is the learning rate controlling how quickly the weights evolve.

## **Inputs**
1. **Gravity (\(X_1\))**:
   - Simulates gravitational effects, such as:
     \[
     X_1 = \frac{m_1 \cdot m_2}{r^2}
     \]
     where \(m_1\) and \(m_2\) are masses, and \(r\) is the distance.
   - Linearly increases over time in this implementation.

2. **Time (\(X_2\))**:
   - Simulates temporal effects, such as time dilation:
     \[
     X_2 = \alpha \cdot t
     \]
     where \(t\) is time, and \(\alpha\) is a scaling factor.
   - Also increases linearly over iterations.

## **How It Works**
1. Each node computes its feedback output using neighboring nodes and its associated input (gravity or time).
2. Weights dynamically adjust to balance forward and backward contributions.
3. The recursive coupling ensures that changes in gravity or time propagate through the system, affecting all nodes.

## **Implementation**
The following Python script simulates the system over multiple iterations and saves results to a JSON file.

### Code
```python
import numpy as np
import json

# Gravity-Time Recursive Feedback System
class GravityTimeFeedback:
    def __init__(self, learning_rate=0.05):
        self.weights = np.ones(8) * 0.5  # Initialize weights equally
        self.learning_rate = learning_rate  # Rate at which weights evolve

    def compute_feedback(self, R, gravity, time):
        # Compute feedback for each node
        R_new = np.zeros(4)
        R_new[0] = (self.weights[0] + self.weights[1]) / (
            (self.weights[0] * R[1] + self.weights[1] * R[3] + self.weights[0] * gravity) /
            (self.weights[0] + self.weights[1] + self.weights[0])
        )
        R_new[1] = (self.weights[2] + self.weights[3]) / (
            (self.weights[2] * R[2] + self.weights[3] * R[0] + self.weights[2] * gravity) /
            (self.weights[2] + self.weights[3] + self.weights[2])
        )
        R_new[2] = (self.weights[4] + self.weights[5]) / (
            (self.weights[4] * R[3] + self.weights[5] * R[1] + self.weights[4] * time) /
            (self.weights[4] + self.weights[5] + self.weights[4])
        )
        R_new[3] = (self.weights[6] + self.weights[7]) / (
            (self.weights[6] * R[0] + self.weights[7] * R[2] + self.weights[6] * time) /
            (self.weights[6] + self.weights[7] + self.weights[6])
        )
        return R_new

    def update_weights(self, R):
        # Adjust weights based on feedback outputs
        self.weights[:4] += self.learning_rate * R
        self.weights[4:] += self.learning_rate * (1 - R)
        # Ensure weights remain positive
        self.weights = np.maximum(self.weights, 0.01)

if __name__ == "__main__":
    # Initialize the system
    system = GravityTimeFeedback(learning_rate=0.05)

    # Initial outputs
    R = np.array([1.0, 0.9, 0.8, 0.7])

    # Inputs
    gravity = 0.5  # Simulated gravitational input (e.g., G = m1 * m2 / r^2)
    time = 0.2  # Simulated time input (e.g., X2 = alpha * t)

    # Iterative simulation
    iterations = 20
    results = []

    for t in range(iterations):
        # Compute new feedback outputs
        R_new = system.compute_feedback(R, gravity, time)

        # Update weights
        system.update_weights(R_new)

        # Record results
        results.append({
            "iteration": t,
            "R": R.tolist(),
            "R_new": R_new.tolist(),
            "weights": system.weights.tolist(),
            "gravity": gravity,
            "time": time
        })

        # Update R for the next iteration
        R = R_new

        # Optionally update gravity and time (dynamic inputs)
        gravity += 0.01  # Simulating a changing gravitational input
        time += 0.01  # Simulating a linear increase in time input

    # Save results to a JSON file
    with open("gravity_time_feedback_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Simulation complete. Results saved to 'gravity_time_feedback_results.json'.")
```

## **Results**
1. The feedback outputs dynamically adapt to changes in gravity and time inputs.
2. Weights evolve to balance the contributions from neighbors and inputs, maintaining overall system stability.
3. The recursive coupling propagates gravitational and temporal influences throughout the system, creating a unified dynamic.

## **Applications**
1. **Physics**: Model interactions between gravity and time, such as time dilation in strong gravitational fields.
2. **AI and Optimization**: Use the recursive feedback mechanism for balancing multiple competing influences.
3. **Complex Systems**: Explore emergent dynamics, such as chaotic behavior or strange attractors.

## **Conclusion**
The Gravity-Time Recursive Feedback System offers a versatile framework for modeling the interplay of physical and temporal dynamics. Its ability to stabilize and adapt to changing inputs makes it a powerful tool for studying complex systems.

