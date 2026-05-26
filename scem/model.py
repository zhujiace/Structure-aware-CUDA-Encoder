from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .config import SCEMConfig
from .states import CudaASTGraphBatch


@dataclass
class SCEMBiasOutput:
    bias: torch.Tensor
    context: torch.Tensor
    memory: torch.Tensor
    fused: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class EdgeAwareGraphTransformerLayer(nn.Module):
    """Graph transformer layer whose attention depends on typed AST edges."""

    def __init__(self, config: SCEMConfig):
        super().__init__()
        if config.ast_dim % config.ast_heads != 0:
            raise ValueError("ast_dim must be divisible by ast_heads")
        self.dim = config.ast_dim
        self.num_heads = config.ast_heads
        self.head_dim = config.ast_dim // config.ast_heads
        self.q_proj = nn.Linear(config.ast_dim, config.ast_dim)
        self.k_proj = nn.Linear(config.ast_dim, config.ast_dim)
        self.v_proj = nn.Linear(config.ast_dim, config.ast_dim)
        self.out_proj = nn.Linear(config.ast_dim, config.ast_dim)
        self.edge_bias = nn.Embedding(config.ast_edge_type_vocab_size, config.ast_heads)
        self.edge_value = nn.Embedding(config.ast_edge_type_vocab_size, config.ast_dim)
        self.norm1 = nn.LayerNorm(config.ast_dim)
        self.norm2 = nn.LayerNorm(config.ast_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.ast_dim, config.ast_ffn_dim),
            nn.SiLU(),
            nn.Dropout(config.ast_dropout),
            nn.Linear(config.ast_ffn_dim, config.ast_dim),
        )
        self.dropout = nn.Dropout(config.ast_dropout)

    def forward(self, hidden: torch.Tensor, graph: CudaASTGraphBatch) -> torch.Tensor:
        batch_size, node_count, _ = hidden.shape
        query = self._split_heads(self.q_proj(hidden))
        key = self._split_heads(self.k_proj(hidden))
        value = self._split_heads(self.v_proj(hidden))

        edge_sources = graph.edge_sources.clamp(min=0, max=max(node_count - 1, 0))
        edge_targets = graph.edge_targets.clamp(min=0, max=max(node_count - 1, 0))
        edge_mask = (
            graph.edge_mask
            & graph.node_mask.gather(1, edge_sources)
            & graph.node_mask.gather(1, edge_targets)
        )

        attended = hidden.new_zeros(batch_size, node_count, self.dim)
        if bool(edge_mask.any()):
            src_key = self._gather_nodes(key, edge_sources)
            dst_query = self._gather_nodes(query, edge_targets)
            src_value = self._gather_nodes(value, edge_sources)
            edge_bias = self.edge_bias(graph.edge_type_ids).to(dtype=hidden.dtype)
            scores = (dst_query * src_key).sum(dim=-1) / (self.head_dim**0.5)
            scores = scores + edge_bias

            valid_scores = scores[edge_mask]
            valid_targets = edge_targets[edge_mask]
            valid_batch = torch.arange(batch_size, device=hidden.device).unsqueeze(1).expand_as(edge_targets)[edge_mask]
            group_ids = valid_batch * node_count + valid_targets
            group_count = batch_size * node_count

            max_per_group = hidden.new_full((group_count, self.num_heads), -torch.inf)
            expanded_groups = group_ids.unsqueeze(-1).expand(-1, self.num_heads)
            max_per_group.scatter_reduce_(0, expanded_groups, valid_scores, reduce="amax", include_self=True)
            weights = torch.exp(valid_scores - max_per_group[group_ids])
            sum_per_group = hidden.new_zeros(group_count, self.num_heads)
            sum_per_group.scatter_add_(0, expanded_groups, weights)
            weights = weights / sum_per_group[group_ids].clamp_min(1e-8)

            edge_value = self.edge_value(graph.edge_type_ids[edge_mask]).to(dtype=hidden.dtype)
            edge_value = edge_value.view(-1, self.num_heads, self.head_dim)
            messages = (src_value[edge_mask] + edge_value) * weights.unsqueeze(-1)
            flat_out = hidden.new_zeros(group_count, self.num_heads, self.head_dim)
            scatter_index = group_ids.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim)
            flat_out.scatter_add_(0, scatter_index, messages)
            attended = flat_out.view(batch_size, node_count, self.dim)

        zero_dependency = (
            query.sum()
            + key.sum()
            + value.sum()
            + self.edge_bias.weight.sum().to(dtype=hidden.dtype)
            + self.edge_value.weight.sum().to(dtype=hidden.dtype)
        ) * 0.0
        attended = attended + zero_dependency
        hidden = self.norm1(hidden + self.dropout(self.out_proj(attended)))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return hidden * graph.node_mask.unsqueeze(-1).to(dtype=hidden.dtype)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = tensor.shape
        return tensor.view(batch_size, node_count, self.num_heads, self.head_dim)

    @staticmethod
    def _gather_nodes(tensor: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
        _, _, num_heads, head_dim = tensor.shape
        gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_heads, head_dim)
        return tensor.gather(1, gather_index)


class CudaASTGraphEncoder(nn.Module):
    """Encode full CUDA AST graphs into memory tokens for hidden-state cross attention."""

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        self.node_type_embedding = nn.Embedding(config.ast_node_type_vocab_size, config.ast_dim, padding_idx=0)
        self.node_text_embedding = nn.Embedding(config.ast_text_vocab_size, config.ast_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(config.ast_max_depth + 1, config.ast_dim)
        self.child_index_embedding = nn.Embedding(config.ast_max_child_index + 1, config.ast_dim)
        self.flag_projection = nn.Linear(config.ast_node_flag_dim, config.ast_dim)
        self.position_projection = nn.Linear(config.ast_node_position_dim, config.ast_dim)
        self.input_norm = nn.LayerNorm(config.ast_dim)
        self.layers = nn.ModuleList([EdgeAwareGraphTransformerLayer(config) for _ in range(config.ast_layers)])
        self.memory_queries = nn.Parameter(torch.randn(config.ast_memory_slots, config.ast_dim) * 0.02)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.ast_dim,
            num_heads=config.ast_heads,
            dropout=config.ast_dropout,
            batch_first=True,
        )
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(config.ast_dim),
            nn.Linear(config.ast_dim, config.memory_dim),
            nn.SiLU(),
            nn.Linear(config.memory_dim, config.memory_dim),
        )

    def forward(self, graph: CudaASTGraphBatch) -> torch.Tensor:
        dtype = self.flag_projection.weight.dtype
        node_type = self.node_type_embedding(graph.node_type_ids)
        node_text = self.node_text_embedding(graph.node_text_ids)
        depth = self.depth_embedding(graph.node_depths.clamp(max=self.config.ast_max_depth))
        child_index = self.child_index_embedding(graph.node_child_indices.clamp(max=self.config.ast_max_child_index))
        flags = self.flag_projection(graph.node_flags.to(dtype=dtype))
        positions = self.position_projection(graph.node_positions.to(dtype=dtype))
        hidden = self.input_norm(node_type + node_text + depth + child_index + flags + positions)
        hidden = hidden * graph.node_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, graph)

        active_graph = graph.node_mask.any(dim=1)
        safe_node_mask = graph.node_mask
        if bool((~active_graph).any()):
            safe_node_mask = graph.node_mask.clone()
            safe_node_mask[~active_graph, 0] = True

        queries = self.memory_queries.unsqueeze(0).expand(hidden.shape[0], -1, -1).to(dtype=hidden.dtype)
        memory, _ = self.memory_attention(
            query=queries,
            key=hidden,
            value=hidden,
            key_padding_mask=~safe_node_mask,
            need_weights=False,
        )
        memory = self.memory_projection(memory)
        if bool((~active_graph).any()):
            memory = memory.masked_fill((~active_graph).view(-1, 1, 1), 0.0)
        return memory


class SCEMContextBiasHead(nn.Module):
    """Context-only low-rank vocabulary bias head."""

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        self.rank_dim = config.bias_rank or config.context_dim
        self.rank = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, self.rank_dim),
            nn.Tanh(),
        )
        self.vocab_proj = nn.Linear(self.rank_dim, config.vocab_size)
        nn.init.zeros_(self.vocab_proj.weight)
        nn.init.zeros_(self.vocab_proj.bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rank = self.rank(context)
        bias = self.vocab_proj(rank)
        if self.config.max_bias is not None:
            bias = self.config.max_bias * torch.tanh(bias / self.config.max_bias)
        return bias, rank


class SCEModule(nn.Module):
    """Structure-aware CUDA Encoding Module with AST graph state."""

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        self.state_encoder = CudaASTGraphEncoder(config)
        self.hidden_norm = nn.LayerNorm(config.lm_hidden_size)
        self.query_projection = nn.Linear(config.lm_hidden_size, config.num_scem_queries * config.context_dim)
        self.query_slot_embedding = nn.Parameter(torch.randn(config.num_scem_queries, config.context_dim) * 0.02)
        self.memory_projection = nn.Linear(config.memory_dim, config.context_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.context_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(config.num_scem_queries * config.context_dim),
            nn.Linear(config.num_scem_queries * config.context_dim, config.context_dim),
            nn.SiLU(),
            nn.Linear(config.context_dim, config.context_dim),
        )
        self.bias_head = SCEMContextBiasHead(config)

    def forward(
        self,
        hidden_state: torch.Tensor,
        state: CudaASTGraphBatch,
        return_attention: bool = False,
    ) -> SCEMBiasOutput:
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
        if hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape [batch, hidden] or [batch, seq, hidden]")

        hidden_state = hidden_state.to(dtype=self.hidden_norm.weight.dtype)
        state = state.to(hidden_state.device)
        active_state = state.node_mask.any(dim=1)
        memory = self.state_encoder(state)
        query = self.query_projection(self.hidden_norm(hidden_state)).view(
            hidden_state.shape[0],
            self.config.num_scem_queries,
            self.config.context_dim,
        )
        query = query + self.query_slot_embedding.unsqueeze(0).to(dtype=query.dtype, device=query.device)
        key_value = self.memory_projection(memory)
        if bool((~active_state).any()):
            key_value = key_value.masked_fill((~active_state).view(-1, 1, 1), 0.0)
        context_tokens, attention_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        context = self.context_projection(context_tokens.reshape(context_tokens.shape[0], -1))
        if bool((~active_state).any()):
            active_scale = active_state.to(dtype=context.dtype).view(-1, 1)
            context = context * active_scale
        bias, fused = self.bias_head(context)
        if bool((~active_state).any()):
            active_scale = active_state.to(dtype=bias.dtype).view(-1, 1)
            bias = bias * active_scale
            fused = fused * active_scale.to(dtype=fused.dtype)
        return SCEMBiasOutput(
            bias=bias,
            context=context,
            memory=memory,
            fused=fused,
            attention_weights=attention_weights if return_attention else None,
        )

    @torch.no_grad()
    def apply_bias(
        self,
        lm_logits: torch.Tensor,
        hidden_state: torch.Tensor,
        state: CudaASTGraphBatch,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        output = self.forward(hidden_state=hidden_state, state=state)
        return lm_logits + alpha * output.bias.to(dtype=lm_logits.dtype)
