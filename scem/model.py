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
        self.program_region = nn.Embedding(config.num_program_regions, config.state_dim)

        self.static_requirements = nn.Linear(8, config.state_dim)
        self.task_format = nn.Linear(4, config.state_dim)
        self.prefix_indexing = nn.Linear(4, config.state_dim)
        self.prefix_memory_write = nn.Linear(5, config.state_dim)
        self.prefix_kernel_control = nn.Linear(7, config.state_dim)
        self.numeric_features = nn.Linear(config.num_numeric_features, config.state_dim)

        self.slot_projection = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.memory_dim),
            nn.SiLU(),
            nn.Linear(config.memory_dim, config.memory_dim),
        )

    def forward(self, state: CudaProgramStateBatch) -> torch.Tensor:
        dtype = self.static_requirements.weight.dtype
        static_flags = state.static_flags.to(dtype=dtype)
        prefix_flags = state.prefix_flags.to(dtype=dtype)
        numeric_features = state.numeric_features.to(dtype=dtype)
        slots = torch.stack(
            [
                self.program_region(state.program_region),
                self.static_requirements(static_flags[:, :8]),
                self.task_format(static_flags[:, 8:12]),
                self.prefix_indexing(prefix_flags[:, :4]),
                self.prefix_memory_write(prefix_flags[:, [4, 5, 6, 7, 12]]),
                self.prefix_kernel_control(prefix_flags[:, [8, 9, 10, 11, 13, 14, 15]]),
                self.numeric_features(numeric_features),
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


class SCEMStateGatedBiasHead(nn.Module):
    """State-conditioned low-rank bias head.

    The LM hidden state proposes a low-rank correction, while CUDA state context
    gates and shifts that correction before projection to vocabulary logits.
    This makes the state path control the final bias instead of being a small
    slice in a direct hidden/context concatenation.
    """

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        self.rank_dim = config.bias_rank or config.fusion_dim
        self.state_gate_scale = getattr(config, "state_gate_scale", 1.0)
        self.state_shift_scale = getattr(config, "state_shift_scale", 0.1)
        self.hidden_rank = nn.Sequential(
            nn.LayerNorm(config.lm_hidden_size),
            nn.Linear(config.lm_hidden_size, self.rank_dim),
            nn.Tanh(),
        )
        self.gate_norm = nn.LayerNorm(config.context_dim)
        self.gate_proj = nn.Linear(config.context_dim, self.rank_dim)
        self.shift_norm = nn.LayerNorm(config.context_dim)
        self.shift_proj = nn.Linear(config.context_dim, self.rank_dim)
        self.rank_norm = nn.LayerNorm(self.rank_dim)
        self.vocab_proj = nn.Linear(self.rank_dim, config.vocab_size)

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)
        nn.init.zeros_(self.vocab_proj.weight)
        nn.init.zeros_(self.vocab_proj.bias)

    def forward(self, hidden_state: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_rank = self.hidden_rank(hidden_state)
        gate = 1.0 + self.state_gate_scale * torch.tanh(self.gate_proj(self.gate_norm(context)))
        shift = self.state_shift_scale * torch.tanh(self.shift_proj(self.shift_norm(context)))
        rank = self.rank_norm(hidden_rank * gate + shift)
        bias = self.vocab_proj(rank)
        if self.config.max_bias is not None:
            bias = self.config.max_bias * torch.tanh(bias / self.config.max_bias)
        return bias, rank


class SCEModule(nn.Module):
    """Structure-aware CUDA Encoding Module.

    Given the current decoder hidden state ``h_t`` and CUDA prefix state memory
    ``M_t``, it returns a vocabulary-sized bias ``b_t`` that can be added to
    the backbone language-model logits.
    """

    def __init__(self, config: SCEMConfig):
        super().__init__()
        self.config = config
        self.bias_arch = getattr(config, "bias_arch", "concat")
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
        if self.bias_arch == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(config.lm_hidden_size + config.context_dim, config.fusion_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_dim, config.fusion_dim),
                nn.SiLU(),
                nn.LayerNorm(config.fusion_dim),
            )
            self.bias_head = SCEMBiasHead(config)
            self.state_gated_bias_head = None
        elif self.bias_arch == "state_gated_delta":
            self.fusion = None
            self.bias_head = None
            self.state_gated_bias_head = SCEMStateGatedBiasHead(config)
        else:
            raise ValueError(f"Unsupported SCEM bias_arch: {self.bias_arch}")

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
        if self.bias_arch == "concat":
            fused = self.fusion(torch.cat([hidden_state, context], dim=-1))
            bias = self.bias_head(fused)
        else:
            bias, fused = self.state_gated_bias_head(hidden_state, context)
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
        """Return LM logits plus the optionally scaled SCEM bias."""

        output = self.forward(hidden_state=hidden_state, state=state)
        return lm_logits + alpha * output.bias.to(dtype=lm_logits.dtype)
