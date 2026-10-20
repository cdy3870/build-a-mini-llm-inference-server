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

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    # TODO: keep only the k largest logits along the last axis, set the rest to -inf
    
    if k >= logits.shape[-1]:
        return logits.copy()

    if logits.ndim != 1:    
        kth_largest = np.sort(logits, axis=-1)[..., -k].reshape(logits.shape[0], 1)
    else:
        kth_largest = np.sort(logits, axis=-1)[..., -k]

    mask = np.where(logits < kth_largest, -np.inf, logits)


    return mask

