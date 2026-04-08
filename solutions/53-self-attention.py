import numpy as np

def softmax(vector):
    vector = np.asarray(vector)
    axes = vector.ndim - 1
    max_val = np.max(vector, axis=axes, keepdims=True)
    sum_exp = np.sum(np.exp(vector - max_val), axis=axes, keepdims=True)
    return np.exp(vector - max_val) / sum_exp

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    seq_len, d_k = Q.shape
    pre_softmax = (Q @ np.transpose(K))/np.sqrt(d_k)
    act = softmax(pre_softmax)
    output = act @ V
    return output

