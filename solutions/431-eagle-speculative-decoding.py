import numpy as np

def eagle_draft_forward(
    hidden_state: np.ndarray,
    token_id: int,
    embed_matrix: np.ndarray,
    fc_fuse_weight: np.ndarray,
    fc_fuse_bias: np.ndarray,
    draft_head_weight: np.ndarray,
    draft_head_bias: np.ndarray,
    lm_head_weight: np.ndarray,
    num_draft_tokens: int = 3
) -> list:
    """
    Generate draft tokens using an EAGLE-style draft model.
    
    Args:
        hidden_state: Last hidden state from target model, shape (d,)
        token_id: Last accepted token ID
        embed_matrix: Token embedding matrix, shape (vocab_size, d)
        fc_fuse_weight: Fusion layer weight, shape (d, 2*d)
        fc_fuse_bias: Fusion layer bias, shape (d,)
        draft_head_weight: Draft head weight, shape (d, d)
        draft_head_bias: Draft head bias, shape (d,)
        lm_head_weight: LM head projection, shape (vocab_size, d)
        num_draft_tokens: Number of draft tokens to generate
    
    Returns:
        List of draft token IDs
    """
    def relu(x: np.array):
        return np.where(x > 0, x, 0)

    token_ids = []
    for i in range(num_draft_tokens):
        current_token_embeddings = embed_matrix[token_id]
        concat = np.concatenate((hidden_state, current_token_embeddings))
        fused = relu(fc_fuse_weight @ concat + fc_fuse_bias).flatten()
        next_hidden = relu(draft_head_weight @ fused + draft_head_bias)
        logits = lm_head_weight @ next_hidden
        selected_token_id = np.argmax(logits)
        token_ids.append(int(selected_token_id))

        token_id = selected_token_id
        hidden_state = next_hidden
    return token_ids

