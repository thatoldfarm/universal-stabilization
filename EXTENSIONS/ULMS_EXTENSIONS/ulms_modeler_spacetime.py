import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D

# Define the Universal Laws Modeling System
class UniversalLawsModel:
    def __init__(self, num_nodes=10, iterations=1000):
        self.num_nodes = num_nodes
        self.iterations = iterations
        self.weights = np.random.rand(18)  # Weights for 18 components (added spacetime curvature)
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
        R_clipped = np.clip(R, -100, 100)  # Clip input to avoid very large values
        return np.clip(np.exp(k * R_clipped), -1e10, 1e10)  # Clip output as a fallback

    def energy(self, R):
        kinetic = 0.5 * np.clip(R**2, -1e10, 1e10)
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
        return np.clip(R * (1 - R), -1e10, 1e10)

    def potential_field(self, R):
        return -9.8 / (R + 1e-10)

    def scaling(self, R):
        return np.clip(R**2, -1e10, 1e10)

    def spacetime_curvature(self, mass, energy, distance):
        G = 6.67430e-11  # Gravitational constant
        c = 3e8  # Speed of light
        curvature = 2 * G * mass / (distance * c**2)
        curvature += energy / (distance + 1e-10)  # Simplified addition of energy influence
        return np.clip(curvature, -1e10, 1e10)  # Clip for stability

    def update_state(self, t):
        X = self.states
        X_prime = np.roll(self.states, 1)  # Example historical shift
        F = np.random.rand(self.num_nodes)  # Random external force

        # Example parameters for spacetime curvature
        mass = 1.0  # Mass influencing curvature
        energy = np.mean(self.energy(X))  # Average energy in the system
        distance = np.mean(X) + 1e-10  # Average state as a proxy for distance
        curvature = self.spacetime_curvature(mass, energy, distance)

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
            curvature,                       # Spacetime curvature
        ])

        weighted_sum = np.dot(self.weights, components)
        total_weight = np.sum(self.weights)

        new_state = np.clip(weighted_sum / (total_weight + 1e-10), -1e10, 1e10)  # Clip to avoid overflow
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

        # Additional Visualizations Inspired by Ladder
        states_array = np.array([d["states"] for d in self.data])

        # 2D Scatter Plot for Node 0 vs Node 1
        plt.figure(figsize=(10, 6))
        plt.scatter(states_array[:, 0], states_array[:, 1], s=0.5, alpha=0.6, c=np.linspace(0, 1, len(states_array)), cmap='viridis')
        plt.title("2D Attractor: Node 0 vs Node 1")
        plt.xlabel("Node 0")
        plt.ylabel("Node 1")
        plt.colorbar(label="Iteration Progress")
        plt.grid()
        plt.savefig("attractor_node0_node1.png")
        plt.close()

        # 3D Scatter Plot for Node 0, Node 1, Node 2
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(states_array[:, 0], states_array[:, 1], states_array[:, 2],
                        c=np.linspace(0, 1, len(states_array)), s=0.5, cmap='coolwarm', alpha=0.6)
        ax.set_title("3D Attractor Dynamics")
        ax.set_xlabel("Node 0")
        ax.set_ylabel("Node 1")
        ax.set_zlabel("Node 2")
        plt.colorbar(sc, label="Iteration Progress")
        plt.savefig("3d_attractor_dynamics.png")
        plt.close()

# Initialize and run the system
model = UniversalLawsModel(num_nodes=10, iterations=1000)
model.run_simulation()
model.save_results("universal_laws_results.json")
model.visualize()

