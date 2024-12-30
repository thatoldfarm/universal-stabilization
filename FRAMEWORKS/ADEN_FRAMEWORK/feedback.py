import numpy as np
from core import BaseFeedbackMechanism
import math
import torch


class VarianceMinimization(BaseFeedbackMechanism):
     """Adjusts weights to minimize the variance of the output."""
     def update_weights(self, current_result, forward_weight, backward_weight):
         """Update weights using variance minimization."""
         variance = np.var(current_result)
         w_f = 1 / (1 + variance)
         w_b = 1 - w_f
         return w_f, w_b

class EntropyMaximization(BaseFeedbackMechanism):
    """Adjusts weights to maximize the entropy of the output."""
    def update_weights(self, current_result, forward_weight, backward_weight):
         """Update weights using entropy maximization."""
         entropy = -np.sum(current_result * np.log(current_result + 1e-6))
         w_f = np.abs(entropy)
         w_b = 1 / (np.abs(entropy) + 1e-6)
         return w_f, w_b


class GradientDescent(BaseFeedbackMechanism):
     """Adjusts weights using gradient descent to minimize the output change."""
     def __init__(self, learning_rate=0.01):
          """Initialize Gradient Descent."""
          super().__init__()
          self.learning_rate = learning_rate

     def update_weights(self, current_result, forward_weight, backward_weight):
          """Update weights using gradient descent."""
          # Compute gradients (simplified example)
          delta = np.diff(current_result, prepend=0) # Estimate of change in result
          grad_wf = np.mean(np.array(delta))
          grad_wb = - np.mean(np.array(delta))

          w_f = forward_weight - self.learning_rate * grad_wf
          w_b = backward_weight - self.learning_rate * grad_wb
          return w_f, w_b

class MomentumBasedUpdate(BaseFeedbackMechanism):
   """Updates weights with momentum-based learning."""
   def __init__(self, learning_rate=0.01, momentum=0.9):
       """Initialize momentum based update"""
       super().__init__()
       self.learning_rate = learning_rate
       self.momentum = momentum
       self.v_f = 0.0
       self.v_b = 0.0

   def update_weights(self, current_result, forward_weight, backward_weight):
        """Update weights using gradient descent and momentum."""
        delta = np.diff(current_result, prepend=0)
        grad_wf = np.mean(np.array(delta))
        grad_wb = -np.mean(np.array(delta))
        self.v_f = self.momentum * self.v_f - self.learning_rate * grad_wf
        self.v_b = self.momentum * self.v_b - self.learning_rate * grad_wb
        w_f = forward_weight + self.v_f
        w_b = backward_weight + self.v_b
        return w_f, w_b

class InformationBottleneck(BaseFeedbackMechanism):
  """Adjusts weights based on the Information Bottleneck Principle"""
  def __init__(self, beta=0.1):
        """Initializes InformationBottleneck."""
        super().__init__()
        self.beta = beta

  def update_weights(self, current_result, forward_weight, backward_weight):
        """Update weights using a simplified Information Bottleneck approximation."""
        # Placeholder; this is a complex calculation
        w_f = abs(np.mean(current_result))
        w_b = 1/w_f if w_f > 0 else 1
        return w_f, w_b

class AdversarialFeedback(BaseFeedbackMechanism):
    """Implements an adversarial feedback loop."""
    def __init__(self, learning_rate=0.01):
         """Initialize adversarial feedback."""
         super().__init__()
         self.learning_rate = learning_rate

    def update_weights(self, current_result, forward_weight, backward_weight):
          """Update weights based on both minimizing and maximizing the change"""
          delta = np.diff(current_result, prepend=0)
          grad_wf = np.mean(np.array(delta))
          grad_wb = -np.mean(np.array(delta))

          w_f_min = forward_weight - self.learning_rate * grad_wf
          w_b_min = backward_weight - self.learning_rate * grad_wb

          w_f_max = forward_weight + self.learning_rate * grad_wf
          w_b_max = backward_weight + self.learning_rate * grad_wb

          return w_f_min, w_b_max # Returns w_f updated to minimize delta, and w_b updated to maximize delta

class AdaptiveCombination(BaseFeedbackMechanism):
    """Adjusts weights using an adaptive combination of variance and entropy."""
    def __init__(self, initial_alpha=0.5, alpha_learning_rate=0.01):
        """Initializes Adaptive Combination feedback mechanism."""
        super().__init__()
        self.alpha = initial_alpha
        self.alpha_learning_rate = alpha_learning_rate

    def update_weights(self, current_result, forward_weight, backward_weight):
        """Update weights using adaptive combination of variance and entropy."""
        variance = np.var(current_result)
        entropy = -np.sum(current_result * np.log(current_result + 1e-6))
        new_alpha = self.alpha + self.alpha_learning_rate * np.sign(entropy - variance)
        self.alpha = np.clip(new_alpha, 0, 1)
        w_f = self.alpha * (1 / (1 + variance)) + (1 - self.alpha) * np.abs(entropy)
        w_b = self.alpha * (1 - (1 / (1 + variance))) + (1 - self.alpha) * (1 / (np.abs(entropy) + 1e-6))
        return w_f, w_b


class KL_Divergence(BaseFeedbackMechanism):
    """Adjusts weights to minimize the KL divergence between input and output."""
    def __init__(self, learning_rate=0.01):
        """Initialize the KL Divergence feedback mechanism."""
        super().__init__()
        self.learning_rate = learning_rate

    def update_weights(self, current_result, forward_weight, backward_weight):
        """Updates weights using the KL Divergence method"""
        # Using a simplified method of tracking output change as an example
        delta = np.diff(current_result, prepend=0)
        grad_wf = np.mean(np.array(delta))
        grad_wb = -np.mean(np.array(delta))

        w_f = forward_weight - self.learning_rate * grad_wf
        w_b = backward_weight - self.learning_rate * grad_wb
        return w_f, w_b
