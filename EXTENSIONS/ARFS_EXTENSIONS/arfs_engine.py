import math
import json
from decimal import Decimal, getcontext
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# Set high precision for Pi calculation
getcontext().prec = 300

# Define constants for spirals
SPIRAL_STEP = 0.1

# Dynamic generator for Pi digits in hexadecimal and binary
def generate_pi_digits(limit):
    """Generate digits of Pi up to the specified limit."""
    with open("pi_digits.txt", "r") as file:
        pi_data = file.read().replace("\n", "")[:limit]
    return [int(char) for char in pi_data if char.isdigit()]

# Convert a sequence of Pi digits into binary strings
def convert_to_binary(digits, bit_length=8):
    """Convert a sequence of digits to binary representations of specified bit length."""
    binary_stream = "".join(bin(digit)[2:].zfill(4) for digit in digits)
    if len(binary_stream) % bit_length != 0:
        padding_length = bit_length - (len(binary_stream) % bit_length)
        binary_stream += "0" * padding_length  # Pad with zeros to ensure uninterrupted chunks

    # Generate all possible 8-bit patterns as a baseline
    complete_set = {bin(i)[2:].zfill(bit_length) for i in range(2**bit_length)}
    extracted_patterns = set()

    for i in range(len(binary_stream) - bit_length + 1):
        chunk = binary_stream[i:i + bit_length]
        extracted_patterns.add(chunk)

    # Add missing patterns to ensure completeness
    missing_patterns = complete_set - extracted_patterns
    all_patterns = list(extracted_patterns.union(missing_patterns))

    return all_patterns

# Spiral mapping
class Spiral:
    def __init__(self, clockwise=True):
        self.angle = 0
        self.radius = 0
        self.clockwise = clockwise
        self.coordinates = []

    def add_point(self, value):
        direction = 1 if self.clockwise else -1
        x = self.radius * math.cos(self.angle)
        y = self.radius * math.sin(self.angle)
        self.coordinates.append((x, y, value))
        self.angle += SPIRAL_STEP * direction
        self.radius += SPIRAL_STEP

# Main dynamic system with expanding inputs and spirals
class DynamicARFS:
    def __init__(self, bit_length=8):
        self.forward_spiral = Spiral(clockwise=True)
        self.backward_spiral = Spiral(clockwise=False)
        self.anchor_points = {}
        self.core_data = deque(maxlen=10)  # Core intelligence window
        self.bit_length = bit_length

    def add_data(self, forward_digits, backward_digits):
        forward_binary = convert_to_binary(forward_digits, self.bit_length)
        backward_binary = convert_to_binary(backward_digits, self.bit_length)

        for binary, spiral in zip([forward_binary, backward_binary], [self.forward_spiral, self.backward_spiral]):
            for value in binary:
                if value not in self.anchor_points:
                    self.anchor_points[value] = len(self.anchor_points)  # Assign unique coordinate id
                spiral.add_point(self.anchor_points[value])
                self.core_data.append(value)

    def visualize_spirals(self):
        """Visualize the spirals dynamically."""
        plt.figure(figsize=(10, 10))
        for spiral, color in zip([self.forward_spiral, self.backward_spiral], ['blue', 'red']):
            coords = np.array([(x, y) for x, y, _ in spiral.coordinates])
            if coords.size > 0:
                plt.plot(coords[:, 0], coords[:, 1], color=color, label=f"{'Clockwise' if spiral.clockwise else 'Counterclockwise'} Spiral")
        plt.scatter(0, 0, c='green', label='Core Intelligence')
        plt.title("Dynamic Spirals with ARFS Integration")
        plt.legend()
        plt.show()

    def apply_arfs(self, forward_digits, backward_digits, iterations=20):
        """Perform Advanced Recursive Feedback Stabilization."""
        forward_binary = convert_to_binary(forward_digits, self.bit_length)
        backward_binary = convert_to_binary(backward_digits, self.bit_length)

        weights_f = np.ones(len(forward_binary))
        weights_b = np.ones(len(backward_binary))
        stabilized_results = []

        for _ in range(iterations):
            # Compute stabilized outputs
            current_result = [(wf * int(f, 2) + wb * int(b, 2)) / (wf + wb)
                              for wf, wb, f, b in zip(weights_f, weights_b, forward_binary, backward_binary)]
            stabilized_results.append(current_result)

            # Update weights based on variance minimization
            variance_f = np.var(current_result)
            weights_f = np.full(len(forward_binary), 1 / (1 + variance_f))
            weights_b = np.full(len(backward_binary), 1 - weights_f[0])

        return stabilized_results

# Instantiate and run the dynamic ARFS system
if __name__ == "__main__":
    dynamic_arfs = DynamicARFS()
    
    # Example: Process first 100 digits of Pi dynamically
    pi_digits = generate_pi_digits(100)
    forward_digits = pi_digits[:50]
    backward_digits = pi_digits[50:][::-1]  # Reverse the second half

    dynamic_arfs.add_data(forward_digits, backward_digits)
    stabilized_results = dynamic_arfs.apply_arfs(forward_digits, backward_digits)

    # Visualize results
    dynamic_arfs.visualize_spirals()

    # Save anchor points to file
    with open("anchor_points.json", "w") as file:
        json.dump(dynamic_arfs.anchor_points, file, indent=4)

    # Save all outputs to a comprehensive JSON file
    output_data = {
        "anchor_points": dynamic_arfs.anchor_points,
        "stabilized_results": stabilized_results,
        "forward_spiral": dynamic_arfs.forward_spiral.coordinates,
        "backward_spiral": dynamic_arfs.backward_spiral.coordinates,
    }

    with open("arfs_full_output.json", "w") as file:
        json.dump(output_data, file, indent=4)

    print("System completed processing. Full output saved to arfs_full_output.json.")

