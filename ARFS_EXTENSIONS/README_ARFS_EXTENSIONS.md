**1. `arfs_engine.py` README:**

```markdown
# Dynamic ARFS with Expanding Spirals

## Description

This script implements a dynamic version of the Advanced Recursive Feedback System (ARFS) that integrates expanding spirals for data mapping. The system takes digits of Pi, transforms them into binary representations, and then maps these representations onto dynamically expanding spirals. It then applies a basic form of ARFS to these mapped binary values.

## Key Features

*   **Dynamic Pi Digit Generation:** Generates Pi digits from a file and converts them to integers.
*   **Binary Conversion:** Transforms integer digits into fixed length binary strings.
*   **Spiral Mapping:** Maps binary strings onto expanding spirals, using both clockwise and counter-clockwise rotations for forward and backward sequences.
*   **Dynamic Input Expansion:** The system continuously adds data, expanding the input space.
*   **Visualizations:** Uses Matplotlib to visualize the expanding spirals.
*   **ARFS Implementation:** Applies a basic form of the ARFS algorithm, using variance minimization to update weights.
*   **JSON Output:** Saves results, including anchor points, stabilized outputs, and coordinates, into JSON files for data persistance.

## Usage

1.  **Ensure You Have Pi Digits:** Place a `pi_digits.txt` file in the same directory as the script, containing Pi digits (without the "3.").

2.  **Run the Script:** `python3 arfs_engine.py`

3.  **View Results:** Output files (`anchor_points.json` and `arfs_full_output.json`) will be created in the same directory. The visualized spiral plot will also be displayed.

## Code Overview

*   **`generate_pi_digits(limit)`:** Loads digits of Pi from file.
*   **`convert_to_binary(digits, bit_length=8)`:** Converts a sequence of digits into binary strings.
*   **`Spiral` Class:** Defines the properties and methods for creating a spiral.
*   **`DynamicARFS` Class:** Implements the dynamic system and ARFS, including methods to add data, visualize spirals and apply the ARFS.

## Parameters

*   `limit`: Number of digits of Pi to generate.
*    `bit_length`: The length of the bit strings

## Notes

*   This script provides a basic form of the ARFS for use as a template.
*    Further modifications can be done to include new feedback mechanisms, and other analysis.
```

**2. `rfsbdm_advanced.py` README:**

```markdown
# Advanced Recursive Feedback System (ARFS)

## Description

This script implements an advanced version of the Recursive Feedback System (ARFS) with a variety of options and configurations. The system takes forward and backward input sequences and stabilizes them using a weighted averaging approach, while adapting the weights based on feedback and other methods.

## Key Features

*   **Dynamic Input:** Accepts forward and backward input sequences (both 1D and higher dimensions).
*   **Periodic Modulation:** An option to apply periodic modulation to weights using sine and cosine functions.
*   **Invariance Transformation:** An option to apply a transformation function to the inputs.
*   **Energy Optimization:** An option to minimize variance in the results, promoting convergence.
*   **Entropy Maximization:** An option to maximize the diversity of outputs by using an entropy based feedback function.
*   **Inter-Domain Scaling:** Dynamically adjusts to higher dimensions.
*   **Geometric Decay Visualization:** Creates a log scale graph of the decay of deltas over time.
*   **JSON output:** Saves results to a JSON file for further analysis.

## Usage

1.  **Run the Script:** `python3 rfsbdm_advanced.py`

2.  **View Output:**
    *   A Matplotlib graph showing the decay of `Delta_t` will be displayed.
    *  The `output.json` file will contain the final stabilized result and all deltas.

## Code Overview

*   **`advanced_recursive_feedback(...)`:** This function implements the core of ARFS with customizable options. It accepts arguments for the forward and backward sequences, iterations, periodic modulation, invariance transformation, energy optimization, entropy maximization and inter-domain scaling.

## Parameters

The `advanced_recursive_feedback` function takes the following parameters:

*   `forward`: The forward input sequence.
*   `backward`: The backward input sequence.
*   `steps`: The number of iterations or steps to perform.
*   `periodic_modulation`: A boolean to enable/disable periodic modulation.
*   `invariance_transformation`: An optional function for transforming inputs.
*   `optimize_energy`: A boolean to enable/disable variance minimization.
*   `entropy_maximization`: A boolean to enable/disable entropy maximization.
*    `inter_domain_scaling`: A boolean to enable/disable inter-domain scaling.

## Notes

*   The code is designed to work with scalar and vector data.
*    The system leverages NumPy for vector calculations.
*   The invariance transformation is optional.
```

**3. `rfsbdm_advanced_pi.py` README:**

```markdown
# Advanced Recursive Feedback System (ARFS) with Pi Digits

## Description

This script applies the Advanced Recursive Feedback System (ARFS) to a large dataset of Pi digits. It demonstrates how the system performs when given a large amount of data in sequential order, using the digits of pi. The system stabilizes two input sequences through a weighted recursive transform using multiple different feedback mechanisms.

## Key Features

*   **Large Pi Digits:** It loads a large number of Pi digits from file.
*   **Bidirectional Input:** Creates forward and backward sequences from the loaded digits.
*   **Configurable ARFS:** It features all of the same parameters as `rfsbdm_advanced.py`, including periodic modulation, invariance transformation, optimize energy (variance minimization), entropy maximization, and inter-domain scaling.
*   **JSON Output:** Saves the result, the differences between each step (deltas), and the dataset size into a JSON output file.

## Usage

1.  **Ensure You Have Pi Digits:** Place a `pi_digits.txt` file in the same directory as the script, containing Pi digits (without the "3.").

2.  **Run the Script:** `python3 rfsbdm_advanced_pi.py`

3.  **View Results:** The `large_pi_output.json` file will be created in the same directory.

## Code Overview

*   **`load_pi_digits(file_path, max_digits=None)`:** Loads Pi digits from a file.
*  **`advanced_recursive_feedback(...)`:** This function implements the core of ARFS with customizable options. It accepts arguments for the forward and backward sequences, iterations, periodic modulation, invariance transformation, energy optimization, entropy maximization and inter-domain scaling.

## Parameters

The `advanced_recursive_feedback` function takes the following parameters:

*   `forward`: The forward input sequence.
*   `backward`: The backward input sequence.
*   `steps`: The number of iterations or steps to perform.
*   `periodic_modulation`: A boolean to enable/disable periodic modulation.
*   `invariance_transformation`: An optional function for transforming inputs.
*   `optimize_energy`: A boolean to enable/disable variance minimization.
*   `entropy_maximization`: A boolean to enable/disable entropy maximization.
*    `inter_domain_scaling`: A boolean to enable/disable inter-domain scaling.
*    `file_path`: The path to the pi_digits file.
*    `max_digits`: An optional argument to limit how many digits are loaded.

## Notes

*   This script is designed to test and explore the ARFS using a large dataset.
*   The code leverages NumPy for efficient array computations.
```

