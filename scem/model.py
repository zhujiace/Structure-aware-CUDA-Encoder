from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .config import SCEMConfig
from .states import CudaProgramStateBatch


@dataclass
class SCEMBiasOutput:
    bias: torch.Tensor
    context: torch.Tensor
    memory: torch.Tensor
    fused: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class CudaStateMemoryEncoder(nn.Module):
    """Encode static CUDA facts and dynamic prefix facts into memory slots."""

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        self.task_family = nn.Embedding(config.num_task_families, config.state_dim)
        self.tensor_rank = nn.Embedding(config.max_tensor_rank + 1, config.state_dim)
        self.program_region = nn.Embedding(config.num_program_regions, config.state_dim)

        self.static_flags = nn.Linear(config.num_static_flags, config.state_dim)
        self.prefix_flags = nn.Linear(config.num_prefix_flags, config.state_dim)
        self.numeric_features = nn.Linear(2, config.state_dim)

        self.slot_projection = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.memory_dim),
            nn.SiLU(),
            nn.Linear(config.memory_dim, config.memory_dim),
        )

    def forward(self, state: CudaProgramStateBatch) -> torch.Tensor:
        dtype = self.static_flags.weight.dtype
        tensor_rank = state.tensor_rank.clamp(0, self.config.max_tensor_rank)
        slots = torch.stack(
            [
                self.task_family(state.task_family),
                self.tensor_rank(tensor_rank),
                self.program_region(state.program_region),
                self.static_flags(state.static_flags.to(dtype=dtype)),
                self.prefix_flags(state.prefix_flags.to(dtype=dtype)),
                self.numeric_features(state.numeric_features.to(dtype=dtype)),
            ],
            dim=1,
        )
        return self.slot_projection(slots)


class SCEMBiasHead(nn.Module):
    """Project fused structure representation into vocabulary-sized bias."""

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        if config.bias_rank is None:
            self.proj = nn.Linear(config.fusion_dim, config.vocab_size)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.rank_proj = None
            self.vocab_proj = None
        else:
            self.rank_proj = nn.Linear(config.fusion_dim, config.bias_rank)
            self.vocab_proj = nn.Linear(config.bias_rank, config.vocab_size)
            nn.init.zeros_(self.vocab_proj.weight)
            nn.init.zeros_(self.vocab_proj.bias)
            self.proj = None

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            bias = self.proj(fused)
        else:
            bias = self.vocab_proj(torch.tanh(self.rank_proj(fused)))
        if self.config.max_bias is not None:
            bias = self.config.max_bias * torch.tanh(bias / self.config.max_bias)
        return bias


class SCEModule(nn.Module):
    """Structure-aware CUDA Encoding Module.

    Given the current decoder hidden state ``h_t`` and CUDA prefix state memory
    ``M_t``, it returns a vocabulary-sized bias ``b_t`` that can be added to
    the backbone language-model logits.
    """

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        self.state_encoder = CudaStateMemoryEncoder(config)
        self.hidden_norm = nn.LayerNorm(config.lm_hidden_size)
        self.query_projection = nn.Linear(config.lm_hidden_size, config.context_dim)
        self.memory_projection = nn.Linear(config.memory_dim, config.context_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.context_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.lm_hidden_size + config.context_dim, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_dim),
            nn.SiLU(),
            nn.LayerNorm(config.fusion_dim),
        )
        self.bias_head = SCEMBiasHead(config)

    def forward(
        self,
        hidden_state: torch.Tensor,
        state: CudaProgramStateBatch,
        return_attention: bool = False,
    ) -> SCEMBiasOutput:
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
        if hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape [batch, hidden] or [batch, seq, hidden]")

        hidden_state = hidden_state.to(dtype=self.hidden_norm.weight.dtype)
        state = state.to(hidden_state.device)
        memory = self.state_encoder(state)
        query = self.query_projection(self.hidden_norm(hidden_state)).unsqueeze(1)
        key_value = self.memory_projection(memory)
        context, attention_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        context = context.squeeze(1)
        fused = self.fusion(torch.cat([hidden_state, context], dim=-1))
        bias = self.bias_head(fused)
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
        state: CudaProgramStateBatch,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Return ``z_t^LM + alpha * b_t`` for decoding-time use."""

        output = self.forward(hidden_state=hidden_state, state=state)
        return lm_logits + alpha * output.bias.to(dtype=lm_logits.dtype)
