from core import dynamic_equilibrium_transform, HardPoint
import numpy as np
from feedback import VarianceMinimization, EntropyMaximization
from structure import Stack, Heap, Funnels, NeutralZone
from physics import EinsteinFieldEquation
from analysis import convergence_rate, delta_variance, final_delta, average_entropy, final_entropy, count_distinct_states, response_time_to_perturbation, change_in_equilibrium_state, equilibrium_score
from utils import create_spiral_coordinates, load_data_from_json, save_data_to_json
from datetime import datetime
import random

class AdaptiveDynamicEquilibriumNetwork:
    def __init__(self, feedback_mechanisms, input_mapping_method="spiral"):
        """Initializes ADEN with specified feedback mechanisms."""
        self.feedback_mechanisms = feedback_mechanisms  # A list of FeedbackMechanisms
        self.input_mapping_method = input_mapping_method
        self.data_structures = {
            "stack": Stack(),
            "heap": Heap(),
            "funnels": Funnels(),
            "neutral_zone": NeutralZone()
        }
        self.einstein_field_equation = EinsteinFieldEquation()  # for additional analysis
        self.state_history = []
        self.hard_points = []  # Array of HardPoints
        self.current_weights = (1,1)
        self.metrics = {}
    def map_input_data(self, raw_data):
        """Maps raw input data to a defined structure of HardPoints."""
        if self.input_mapping_method == "spiral":
            for index, item in enumerate(raw_data):
                x, y = create_spiral_coordinates(index)
                hard_point = HardPoint({
                    "offset": index,
                    "coordinates": (x, y),
                    "raw_value": item,
                    "timestamp": datetime.utcnow().isoformat()
                })
                self.hard_points.append(hard_point)


    def run_transformation(self):
        """Performs the recursive transformation on the data."""
        forward_data = np.array([data.properties['raw_value'] for data in self.hard_points], dtype=float)
        backward_data = np.array([data.properties['raw_value'] for data in reversed(self.hard_points)],dtype=float)
        w_f, w_b = self.current_weights

        current_result = dynamic_equilibrium_transform(forward_data, backward_data, w_f, w_b)
        self.state_history.append(current_result)  # track change of result

        for feedback_mechanism in self.feedback_mechanisms:
            w_f, w_b = feedback_mechanism.update_weights(current_result, w_f, w_b)

        self.current_weights = (w_f, w_b)

        for hard_point in self.hard_points:
            hard_point.properties["weights"] = self.current_weights
    def run_analysis(self):
        """Analyzes the results of the transformation."""
        self.delta_t_list = [np.linalg.norm(self.state_history[t + 1] - self.state_history[t]) for t in range(len(self.state_history) - 1)]
        stability_score = (1- convergence_rate(self.delta_t_list)) * (1 - delta_variance(self.delta_t_list)) * (1 - final_delta(self.delta_t_list))
        diversity_score = (average_entropy(self.state_history) * count_distinct_states(self.state_history[-1])) / (len(self.hard_points) + 1e-6 ) # Added a miniscule amount to avoid zero
        adaptability_score = (1/(response_time_to_perturbation(self.state_history)+ 1e-6)) + (1 - change_in_equilibrium_state(self.state_history[-1], self.state_history[-1] + random.randint(0, 1)))  # Added a miniscule amount to avoid zero
        self.metrics = {
           "convergence_rate": convergence_rate(self.delta_t_list),
           "delta_variance": delta_variance(self.delta_t_list),
            "final_delta": final_delta(self.delta_t_list),
            "average_entropy": average_entropy(self.state_history),
            "final_entropy": final_entropy(self.state_history[-1]),
             "distinct_states": count_distinct_states(self.state_history[-1]),
             "response_time": response_time_to_perturbation(self.state_history),
             "change_in_equilibrium": change_in_equilibrium_state(self.state_history[-1], self.state_history[-1] + random.randint(0, 1)),
            "equilibrium_score": equilibrium_score(stability_score, diversity_score, adaptability_score)
        }
    def run(self, input_data, steps=10):
        """Runs the ADEN system through all steps of the transformation."""
        self.map_input_data(input_data)  # Creates Hard Points
        for _ in range(steps):  # Runs the system through each step
            self.run_transformation()
        self.run_analysis()  # Calculates system metrics

    def save_results(self, file_path="outputs/aden_results.json"):
         """Saves all data to a JSON File."""
         data = {
             "metrics": self.metrics,
              "state_history": [item.tolist() for item in self.state_history], #To avoid errors related to numpy arrays
              "hard_points": [item.properties for item in self.hard_points]
         }
         save_data_to_json(data, file_path)

