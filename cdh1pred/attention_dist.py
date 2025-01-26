"""
In parts from https://github.com/AliHaiderAhmad001/Self-Attention-with-Relative-Position-Representations/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePosition(nn.Module):
    """
    Relative Position Embeddings Module

    This module generates learnable relative position embeddings to enrich
    the self-attention mechanism with information about the relative distances
    between elements in input sequences.

    Args:
        d_a (int): Number of dimensions in the relative position embeddings.
        k (int): Clipping distance.

    Attributes:
        position_embeddings (nn.Parameter): Learnable parameter for relative position embeddings.
    """

    def __init__(self, d_a, k, add_cls=True):
        """
        Initialize the RelativePosition module.

        Args:
        - d_a (int): Number of dimensions in the relative position embeddings.
        - k (int): Clipping distance.
        - add_cls (bool): Whether to consider additional class token
        """
        super().__init__()
        self.d_a = d_a
        self.k = k
        self.add_cls = add_cls
        self.position_embeddings = nn.Parameter(torch.empty((k + 1, d_a)))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, coordinates):
        """
        Compute relative position embeddings.

        Args:
        - coordinates (torch.Tensor): Coordinates associated with the queries/keys.

        Returns:
        - embeddings (torch.Tensor): Relative position embeddings (length_query, length_key, embedding_dim).
        """
        # Generate relative position embeddings
        distance_matrix = torch.cdist(coordinates, coordinates)

        temp_1 = torch.argsort(torch.argsort(distance_matrix))
        temp_2 = torch.argsort(torch.argsort(distance_matrix, dim=-2), dim=-2)
        neighbors_matrix = torch.triu(temp_1, diagonal=1) + torch.tril(temp_2, diagonal=1)
        neighbors_matrix_clipped = torch.clamp(neighbors_matrix, 0, self.k)

        # zero-pad to deal with additional class-token
        if self.add_cls:
            length = coordinates.shape[-2]
            batch = coordinates.shape[0]
            neighbors_matrix_clipped = torch.cat((torch.zeros(batch, length, 1), neighbors_matrix_clipped),
                                                 dim=-1)
            neighbors_matrix_clipped = torch.cat((torch.zeros(batch, 1, length+1), neighbors_matrix_clipped),
                                                 dim=-2)

        embeddings = self.position_embeddings[neighbors_matrix_clipped.to(torch.long)]

        return embeddings


class RelationAwareAttentionHead(nn.Module):
    """
    Relation-aware attention head implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, hidden_size, head_dim):
        """
        Initializes the RelationAwareAttentionHead.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)

    def forward(self,
                query,
                key,
                value,
                k_bias_matrix,
                v_bias_matrix,
                mask=None):
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            k_bias_matrix (torch.Tensor): relation bias for keys.
            v_bias_matrix (torch.Tensor): relation bias for values.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query = self.query_weights(query)  # (b_s, n_t, head_dim)
        key = self.key_weights(key)  # (b_s, n_t, head_dim)
        value = self.value_weights(value)  # (b_s, n_t, head_dim)

        # Self-Attention scores
        attn_1 = torch.matmul(query, key.transpose(1, 2))  # Q*K^T:(b_s, n_t, n_t)

        # Relative Position Attention scores
        attn_2 = torch.squeeze(torch.matmul(query.unsqueeze(2), (k_bias_matrix.transpose(-2, -1))), -2)  # Q*K_shifting^T:(b_s, n_t, n_t)

        # Relation-aware Self-Attention scores
        att_scores = (attn_1 + attn_2) / self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights = F.softmax(att_scores, dim=-1)

        # Weighted sum of values
        values_1 = torch.matmul(att_weights, value)  # (b_s, n_t, head_dim)

        # Relative Position Representation for values
        values_2 = torch.squeeze(torch.matmul(att_weights.unsqueeze(2), v_bias_matrix), -2)  # (b_s, n_t, head_dim)

        # Relation-aware values
        n_value = values_1 + values_2

        return n_value


class RelationAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        k (int): Clipping distance for relative position embeddings.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        relative_position_k (RelativePosition): Instance of RelativePosition for query-key relative positions.
        relative_position_v (RelativePosition): Instance of RelativePosition for query-value relative positions.
        attention_heads (nn.ModuleList): List of RelationAwareAttentionHead layers.
        fc (nn.Linear): Fully connected layer for final projection.
    """

    def __init__(self, hidden_size, num_heads, k, add_cls=True):
        """
        Initializes the RelationAwareMultiHeadAttention layer.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            num_heads (int): Number of attention heads.
            k (int): Clipping distance for relative position embeddings.
            add_cls (bool): Whether to consider additional class token
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.relative_position_k = RelativePosition(self.head_dim, k, add_cls=add_cls)
        self.relative_position_v = RelativePosition(self.head_dim, k, add_cls=add_cls)
        self.attention_heads = nn.ModuleList([RelationAwareAttentionHead(self.hidden_size, self.head_dim)
                                              for _ in range(self.num_heads)])
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, coords, mask=None):
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            coords (torch.Tensor): Coordinates.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
        """

        k_bias_matrix = self.relative_position_k(coordinates=coords)
        v_bias_matrix = self.relative_position_v(coordinates=coords)

        attention_outputs = [attention_head(query, key, value, k_bias_matrix, v_bias_matrix, mask=mask)
                             for attention_head in self.attention_heads]

        hidden_state = torch.cat(attention_outputs, dim=-1)
        hidden_state = self.fc(hidden_state)

        return hidden_state
