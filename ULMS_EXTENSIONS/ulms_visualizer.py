import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from JSON file
def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Extract states and iterations
def process_data(data):
    iterations = [entry['iteration'] for entry in data]
    states = np.array([entry['states'] for entry in data])
    return iterations, states

# Visualize 2D Attractor: Node 0 vs Node 1
def visualize_2d_attractor(states):
    plt.figure(figsize=(10, 6))
    plt.scatter(states[:, 0], states[:, 1], s=0.5, alpha=0.6, c=np.linspace(0, 1, len(states)), cmap='viridis')
    plt.title("2D Attractor: Node 0 vs Node 1")
    plt.xlabel("Node 0")
    plt.ylabel("Node 1")
    plt.colorbar(label="Iteration Progress")
    plt.grid()
    plt.savefig("attractor_node0_node1.png")
    plt.close()

# Visualize 3D Attractor: Node 0, Node 1, Node 2
def visualize_3d_attractor(states):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(states[:, 0], states[:, 1], states[:, 2],
                    c=np.linspace(0, 1, len(states)), s=0.5, cmap='coolwarm', alpha=0.6)
    ax.set_title("3D Attractor Dynamics")
    ax.set_xlabel("Node 0")
    ax.set_ylabel("Node 1")
    ax.set_zlabel("Node 2")
    plt.colorbar(sc, label="Iteration Progress")
    plt.savefig("3d_attractor_dynamics.png")
    plt.close()

# Visualize System Dynamics for all nodes
def visualize_system_dynamics(iterations, states):
    plt.figure(figsize=(10, 6))
    for i in range(states.shape[1]):
        plt.plot(iterations, states[:, i], label=f"Node {i + 1}")

    plt.xlabel("Iteration")
    plt.ylabel("State")
    plt.title("Universal Laws System Dynamics")
    plt.legend()
    plt.grid()
    plt.savefig("universal_laws_dynamics.png")
    plt.close()

# Visualize Universal Laws Attractor Dynamics (Node 1 vs Node 2)
def visualize_attractor_dynamics(states):
    plt.figure(figsize=(10, 6))
    plt.plot(states[:, 0], states[:, 1], '.', markersize=0.5, alpha=0.7)
    plt.title("Universal Laws Attractor Dynamics")
    plt.xlabel("State of Node 1")
    plt.ylabel("State of Node 2")
    plt.grid()
    plt.savefig("universal_laws_attractor_dynamics.png")
    plt.close()

# Main function to execute visualizations
def main():
    filename = "universal_laws_results.json"
    data = load_data(filename)
    iterations, states = process_data(data)

    visualize_system_dynamics(iterations, states)
    visualize_attractor_dynamics(states)
    visualize_2d_attractor(states)
    visualize_3d_attractor(states)

if __name__ == "__main__":
    main()

