import json
import math

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio

def load_data_from_json(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
         print(f"Error loading data from {file_path}: {e}")
         return None

def save_data_to_json(data, file_path):
     """Saves data to a JSON file."""
     try:
         with open(file_path, "w") as file:
              json.dump(data, file, indent=4)
         print(f"Data saved to {file_path}")
     except Exception as e:
         print(f"Error saving data to {file_path}: {e}")

def create_spiral_coordinates(offset):
    """Calculates 2D spiral coordinates."""
    if offset <= 0:
        return (0, 0)
    r = math.sqrt(offset)
    theta = 2 * math.pi * (offset / PHI)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y
