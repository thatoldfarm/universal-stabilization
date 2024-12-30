import numpy as np
import matplotlib.pyplot as plt
import json

# Define the Universal Laws Modeling System
class UniversalLawsModel:
    def __init__(self, num_nodes=10, iterations=1000):
        self.num_nodes = num_nodes
        self.iterations = iterations
        self.weights = np.random.rand(17)  # Weights for 17 components
        self.states = np.random.rand(num_nodes)  # Initial states
        self.data = []

    def entropy(self, R):
        epsilon = 1e-10
        R_clipped = np.clip(R, epsilon, None)  # Ensure R is always >= epsilon
        return -R_clipped * np.log(R_clipped)

    def noise(self):
        return np.random.normal(0, 0.1, self.num_nodes)

    def coupling(self, states):
        coupling_matrix = np.random.rand(self.num_nodes, self.num_nodes)
        return np.dot(coupling_matrix, states)

    def growth(self, R):
        k = 0.01
        return np.exp(k * R)

    def energy(self, R):
        kinetic = 0.5 * R**2
        potential = 9.8 * R
        return kinetic + potential

    def momentum(self, R):
        mass = 1.0
        velocity = R  # Simplistic assumption
        return mass * velocity

    def equilibrium(self, R):
        return R / (np.sum(R) + 1e-10)

    def damping(self, R):
        damping_coefficient = 0.1
        return -damping_coefficient * R

    def wave(self, t):
        A = 1.0
        frequency = 0.05
        return A * np.sin(2 * np.pi * frequency * t)

    def information_flow(self, states):
        flow_matrix = np.random.rand(self.num_nodes, self.num_nodes)
        return np.dot(flow_matrix, states)

    def temperature(self, states):
        return np.mean(states)

    def feedback(self, R):
        return R * (1 - R)

    def potential_field(self, R):
        return -9.8 / (R + 1e-10)

    def scaling(self, R):
        return R**2

    def update_state(self, t):
        X = self.states
        X_prime = np.roll(self.states, 1)  # Example historical shift
        F = np.random.rand(self.num_nodes)  # Random external force

        components = np.array([
            np.mean(X),                      # Current state
            np.mean(X_prime),                # Historical state
            np.mean(F),                      # External force
            np.mean(self.entropy(X)),        # Entropy
            np.mean(self.noise()),           # Noise
            np.mean(self.coupling(X)),       # Coupling
            np.mean(self.growth(X)),         # Growth
            np.mean(self.energy(X)),         # Energy
            np.mean(self.momentum(X)),       # Momentum
            np.mean(self.equilibrium(X)),    # Equilibrium
            np.mean(self.damping(X)),        # Damping
            self.wave(t),                    # Wave (single scalar already)
            np.mean(self.information_flow(X)),  # Information flow
            self.temperature(X),             # Temperature (single scalar)
            np.mean(self.feedback(X)),       # Feedback
            np.mean(self.potential_field(X)),# Potential field
            np.mean(self.scaling(X)),        # Scaling
        ])

        weighted_sum = np.dot(self.weights, components)
        total_weight = np.sum(self.weights)

        new_state = weighted_sum / (total_weight + 1e-10)
        return np.full(self.num_nodes, new_state)  # Return a uniform updated state

    def run_simulation(self):
        for t in range(self.iterations):
            new_state = self.update_state(t)
            self.states = new_state

            self.data.append({
                "iteration": t,
                "states": self.states.tolist()
            })

    def save_results(self, filename):
        with open(filename, "w") as f:
            json.dump(self.data, f, indent=4)

    def visualize(self):
        # Standard Trends Visualization
        plt.figure(figsize=(10, 6))
        for i in range(self.num_nodes):
            plt.plot(
                [d["iteration"] for d in self.data],
                [d["states"][i] for d in self.data],
                label=f"Node {i + 1}"
            )

        plt.xlabel("Iteration")
        plt.ylabel("State")
        plt.title("Universal Laws System Dynamics")
        plt.legend()
        plt.grid()
        plt.savefig("universal_laws_dynamics.png")

        # Attractor Visualization (Inspired by Ladder.py)
        plt.figure(figsize=(10, 6))
        plt.plot(
            [d["states"][0] for d in self.data],  # First node state
            [d["states"][1] for d in self.data],  # Second node state
            '.', markersize=0.5, alpha=0.7
        )
        plt.title("Universal Laws Attractor Dynamics")
        plt.xlabel("State of Node 1")
        plt.ylabel("State of Node 2")
        plt.grid()
        plt.savefig("universal_laws_attractor_dynamics.png")

# Initialize and run the system
model = UniversalLawsModel(num_nodes=10, iterations=1000)
model.run_simulation()
model.save_results("universal_laws_results.json")
model.visualize()

