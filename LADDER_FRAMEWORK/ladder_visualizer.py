import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load JSON data
def load_results(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Plot R_new[0] vs. R_new[1]
def plot_2d_r0_r1(results, output_file):
    R_new_values = np.array([entry["R_new"] for entry in results])
    plt.figure(figsize=(10, 6))
    plt.scatter(R_new_values[:, 0], R_new_values[:, 1], s=0.5, alpha=0.6, c=np.linspace(0, 1, len(R_new_values)), cmap='viridis')
    plt.title("Jacob's Ladder: R_new[0] vs. R_new[1] (Attractor Patterns)")
    plt.xlabel("R_new[0]")
    plt.ylabel("R_new[1]")
    plt.colorbar(label="Iteration Progress")
    plt.grid()
    plt.savefig(output_file)
    plt.close()

# Plot R_new[2] vs. R_new[3]
def plot_2d_r2_r3(results, output_file):
    R_new_values = np.array([entry["R_new"] for entry in results])
    plt.figure(figsize=(10, 6))
    plt.scatter(R_new_values[:, 2], R_new_values[:, 3], s=0.5, alpha=0.6, c=np.linspace(0, 1, len(R_new_values)), cmap='plasma')
    plt.title("Jacob's Ladder: R_new[2] vs. R_new[3] (Attractor Patterns)")
    plt.xlabel("R_new[2]")
    plt.ylabel("R_new[3]")
    plt.colorbar(label="Iteration Progress")
    plt.grid()
    plt.savefig(output_file)
    plt.close()

# Plot 3D graph of R_new[0], R_new[1], R_new[2]
def plot_3d_r0_r1_r2(results, output_file):
    R_new_values = np.array([entry["R_new"] for entry in results])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(R_new_values[:, 0], R_new_values[:, 1], R_new_values[:, 2],
                    c=np.linspace(0, 1, len(R_new_values)), s=0.5, cmap='coolwarm', alpha=0.6)
    ax.set_title("Jacob's Ladder: 3D Patterns in R_new")
    ax.set_xlabel("R_new[0]")
    ax.set_ylabel("R_new[1]")
    ax.set_zlabel("R_new[2]")
    plt.colorbar(sc, label="Iteration Progress")
    plt.savefig(output_file)
    plt.close()

# Main function to process JSON and generate plots
def main():
    json_file = "jacobs_ladder_results.json"  # Path to the JSON file
    results = load_results(json_file)

    # Generate plots
    plot_2d_r0_r1(results, "jacobs_ladder_r0_r1.png")
    plot_2d_r2_r3(results, "jacobs_ladder_r2_r3.png")
    plot_3d_r0_r1_r2(results, "jacobs_ladder_r0_r1_r2.png")

    print("Graphs generated and saved as PNG files.")

if __name__ == "__main__":
    main()

