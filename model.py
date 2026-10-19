"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np
def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    
    return (np.exp(logits - np.max(logits, keepdims=True, axis=-1))) / (np.exp(logits - np.max(logits, keepdims=True, axis=-1))).sum(keepdims=True, axis=-1)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    

    if temperature <= 0:
        return logits

    return logits / temperature

