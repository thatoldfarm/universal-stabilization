from aden import AdaptiveDynamicEquilibriumNetwork
from feedback import VarianceMinimization, EntropyMaximization
import numpy as np
import os

# Ensure the groupings folder exists
os.makedirs("outputs", exist_ok=True)

if __name__ == "__main__":
    feedback_mechanisms = [VarianceMinimization(), EntropyMaximization()]  # Choose Feedback Mechanisms
    aden = AdaptiveDynamicEquilibriumNetwork(feedback_mechanisms)
    raw_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]  # Example Data
    aden.run(raw_data, steps=20)
    aden.save_results()
