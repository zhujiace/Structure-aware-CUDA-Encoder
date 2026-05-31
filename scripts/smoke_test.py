import sys
from pathlib import Path

import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scem import CudaASTGraphExtractor, SCEMConfig, SCEModule


def main():
    lm_config = AutoConfig.from_pretrained("./models/Qwen3-0.6B-Base", local_files_only=True)
    config = SCEMConfig.from_lm_config(
        lm_config,
        bias_rank=32,
        ast_dim=128,
        ast_ffn_dim=256,
        ast_layers=2,
        ast_heads=4,
        ast_memory_slots=8,
        memory_dim=128,
        context_dim=128,
    )
    scem = SCEModule(config)

    extractor = CudaASTGraphExtractor(
        max_nodes=config.ast_max_nodes,
        max_edges=config.ast_max_edges,
        node_type_vocab_size=config.ast_node_type_vocab_size,
        edge_type_vocab_size=config.ast_edge_type_vocab_size,
        text_vocab_size=config.ast_text_vocab_size,
        max_depth=config.ast_max_depth,
        max_child_index=config.ast_max_child_index,
        node_flag_dim=config.ast_node_flag_dim,
        node_position_dim=config.ast_node_position_dim,
    )
    batch = extractor.extract_batch(
        [
            '```cpp\n__global__ void add(float* x, float* y, float* out, int n) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < n) { out[idx] =',
            "```cpp\n__global__ void reduce(float* x, float* out) { extern __shared__ float smem[]; smem[threadIdx.x] = x[threadIdx.x]; __syncthreads();",
        ]
    )
    hidden = torch.randn(batch.batch_size, config.lm_hidden_size)
    logits = torch.randn(batch.batch_size, config.vocab_size)

    with torch.no_grad():
        output = scem(hidden, batch, return_attention=True)
        adjusted_logits = scem.apply_bias(logits, hidden, batch)

    print("bias:", tuple(output.bias.shape))
    print("memory:", tuple(output.memory.shape))
    print("attention:", tuple(output.attention_weights.shape))
    print("adjusted_logits:", tuple(adjusted_logits.shape))


if __name__ == "__main__":
    main()
