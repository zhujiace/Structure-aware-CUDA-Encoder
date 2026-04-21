from dataclasses import dataclass
from typing import Optional


@dataclass
class SCEMConfig:
    """Configuration for the Structure-aware CUDA Encoding Module."""

    lm_hidden_size: int
    vocab_size: int
    state_dim: int = 64
    memory_dim: int = 256
    context_dim: int = 256
    fusion_hidden_dim: int = 512
    fusion_dim: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.0
    bias_rank: Optional[int] = 64
    max_bias: Optional[float] = 10.0

    # CUDA state vocab sizes. Keep these small and explicit so training data can
    # serialize states as compact integer IDs.
    num_task_families: int = 16
    max_tensor_rank: int = 8
    num_program_regions: int = 8

    # Static and dynamic binary state flags.
    num_static_flags: int = 3
    num_prefix_flags: int = 7

    @classmethod
    def from_lm_config(cls, lm_config, **kwargs) -> "SCEMConfig":
        """Build a SCEM config from a Hugging Face model config.

        Supports both plain causal-LM configs and nested multimodal configs
        whose text model config is stored under ``text_config``.
        """

        text_config = getattr(lm_config, "text_config", lm_config)
        hidden_size = getattr(text_config, "hidden_size")
        vocab_size = getattr(text_config, "vocab_size")
        return cls(lm_hidden_size=hidden_size, vocab_size=vocab_size, **kwargs)
