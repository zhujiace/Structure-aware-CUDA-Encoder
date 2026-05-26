from dataclasses import dataclass
from typing import Optional


@dataclass
class SCEMConfig:
    """Configuration for the Structure-aware CUDA Encoding Module."""

    lm_hidden_size: int
    vocab_size: int
    ast_dim: int = 256
    ast_ffn_dim: int = 1024
    ast_layers: int = 3
    ast_heads: int = 4
    ast_memory_slots: int = 16
    ast_node_type_vocab_size: int = 4096
    ast_edge_type_vocab_size: int = 1024
    ast_text_vocab_size: int = 8192
    ast_max_nodes: int = 768
    ast_max_edges: int = 3072
    ast_max_depth: int = 64
    ast_max_child_index: int = 64
    ast_node_flag_dim: int = 6
    ast_node_position_dim: int = 5
    ast_dropout: float = 0.0
    memory_dim: int = 256
    context_dim: int = 256
    num_attention_heads: int = 4
    num_scem_queries: int = 4
    dropout: float = 0.0
    bias_rank: Optional[int] = 256
    max_bias: Optional[float] = 10.0

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
