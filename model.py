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

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    # TODO: keep smallest set of tokens whose cumulative prob >= p, mask the rest to -inf.
    probs = stable_softmax(logits)

    # ort ids and get corresponding vals
    idx = np.argsort(-probs, axis=-1, kind='stable')
    vals = np.take_along_axis(probs, idx, axis=-1)

    # Get cumulative prob mass
    cum_probs = np.cumsum(vals, axis=-1)

    # Decide if we keep the token if we met p before the token came along
    keep_sorted = (cum_probs - vals) < p

    # Start with zero mask
    mask = np.zeros_like(keep_sorted)

    # Fill zero mask using sorted ids map and sorted values to keep
    np.put_along_axis(mask, idx, keep_sorted, axis=-1)

    # Filter logits based on this
    return np.where(mask, logits, -np.inf)

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    # TODO: draw a single token id from the categorical distribution probs using rng
    
    id = rng.choice(len(probs), p=probs)

    return int(id)

# Step 6 - greedy_select
def greedy_select(logits):
    # TODO: return the index of the maximum logit (ties -> lowest index).
    
    indices = np.argwhere(logits == np.max(logits))

    return indices[0][0]

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # TODO: build a character-level vocab; specials get the lowest ids, then sorted unique chars.
    tokens = [s for s in special_tokens]
    total_set = set()
    for word in corpus:
        for char in word:
            total_set.add(char)

    for char in sorted(list(total_set)):
        tokens.append(char)

    token_to_id = {}
    for i in range(len(tokens)):
        token_to_id[tokens[i]] = i

    return {"token_to_id": token_to_id, "id_to_token": tokens}

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    # TODO: encode text into token ids using vocab, optionally prepending <bos>.
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]

    chars = [c for c in text]
    ids = []

    if add_bos:
        ids = [token_to_id["<bos>"]]


    for char in chars:
        if char not in token_to_id:
            if "<unk>" in token_to_id:
                ids.append(token_to_id["<unk>"])
        else:
            ids.append(token_to_id[char])

    return ids

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    # TODO: convert token ids back into a string using vocab['id_to_token'], optionally skipping specials.
    
    id_to_token = vocab["id_to_token"]

    string = ""

    for id in token_ids:
        token = id_to_token[id]
        if not skip_special:
            string += token
        else:
            if token[0] != "<" and token[-1] != ">":
                string += token

    return string

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    # TODO: return the (T, D) embedding rows for each token id in token_ids
    return embedding_matrix[token_ids, :]

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    # TODO: Apply y = x @ weight + bias, with bias optional and broadcasting over leading axes.
    
    if bias is not None:
        return x @ weight + bias.reshape(-1, weight.shape[1])

    return x @ weight

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    # TODO: allocate empty K and V buffers and a length counter for a single sequence
    
    K_buffer = np.zeros((max_seq_len, d_model), dtype=np.float32)
    V_buffer = np.zeros((max_seq_len, d_model), dtype=np.float32)

    return {"K": K_buffer, "V": V_buffer, "length": 0}

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    # TODO: write k_new and v_new into the cache starting at cache['length'] and bump length.
    
    L = cache["length"]
    t = k_new.shape[0]
    cache["K"][L:L+t] = k_new
    cache["V"] [L:L+t] = v_new
    cache["length"] += t

    return cache

# Step 14 - causal_attention
import numpy as np

def causal_attention(q, k, v, is_causal=True):
    # TODO: scaled dot-product attention with optional causal mask, returns (Tq, D)
    attention_scores = (q @ k.T) / np.sqrt(q.shape[-1]) # scale so scores don't get too high as vector does

    if is_causal:
        Tk = attention_scores.shape[1]
        Tq = attention_scores.shape[0]
        # k decides how many to the right of the diagonal gets masked
        # in normal attention tk = tq, but with kv cache q doesnt need to be
        # as high anymore
        mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=1 + (Tk - Tq))
        attention_scores[mask] = -np.inf
        
    attention = stable_softmax(attention_scores) @ v

    return attention

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    # TODO: embed tokens, project Q/K/V, fill the KV cache, run causal attention, return last-position logits.
    x = embed_tokens(token_ids, params["embedding"])
    k = linear_projection(x, params["Wk"])
    q = linear_projection(x, params["Wq"])
    v = linear_projection(x, params["Wv"])

    kv_cache = init_kv_cache(params["max_seq_len"], params["embedding"].shape[1])
    kv_cache = append_kv(kv_cache, k, v)

    T = len(token_ids)
    attention = causal_attention(q, kv_cache["K"][:T], kv_cache["V"][:T], is_causal=True)
    
    H = attention @ params["Wo"]

    logits = H[-1] @ params["W_out"]
    return logits, kv_cache

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    # TODO: advance generation by one token using the existing KV cache and return next-token logits
    
    x = embed_tokens([token_id], params["embedding"])

    k = linear_projection(x, params["Wk"])
    q = linear_projection(x, params["Wq"])
    v = linear_projection(x, params["Wv"])

    kv_cache = append_kv(cache, k, v)

    L = kv_cache["length"]
    attention = causal_attention(q, kv_cache["K"][:L], kv_cache["V"][:L], is_causal=True)
    
    H = attention @ params["Wo"]

    logits = H[-1] @ params["W_out"]
    return logits, kv_cache

# Step 17 - blocks_needed
import math
def blocks_needed(num_tokens, block_size):
    # TODO: return the number of fixed-size blocks needed to store num_tokens tokens.
    

    return math.ceil(num_tokens / block_size)

# Step 18 - init_block_allocator
def init_block_allocator(num_blocks, block_size, d_model):
    # TODO: build the paged KV allocator dict with K_blocks, V_blocks, free_list, seq_tables, and config.
    
    K_blocks = np.zeros((num_blocks, block_size, d_model))
    V_blocks = np.zeros((num_blocks, block_size, d_model))

    free_list = [i for i in range(num_blocks)]

    seq_tables = {}

    return {
        "K_blocks": K_blocks,
        "V_blocks": V_blocks,
        "free_list": free_list,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "d_model": d_model,
        "seq_tables": seq_tables
    }

