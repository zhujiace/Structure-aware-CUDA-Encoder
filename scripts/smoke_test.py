import sys
from pathlib import Path

import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scem import CudaProgramStateBatch, CudaProgramStateExtractor, SCEMConfig, SCEModule


def main():
    lm_config = AutoConfig.from_pretrained("./models/Qwen3-0.6B-Base", local_files_only=True)
    config = SCEMConfig.from_lm_config(lm_config, bias_rank=32)
    scem = SCEModule(config)

    extractor = CudaProgramStateExtractor(task_family="elementwise", tensor_rank=1)
    states = extractor.extract_batch(
        [
            '__global__ void add(float* x, float* y, float* out, int n) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < n) { out[idx] =',
            "__global__ void reduce(float* x, float* out) { extern __shared__ float smem[]; smem[threadIdx.x] = x[threadIdx.x]; __syncthreads();",
        ]
    )
    batch = CudaProgramStateBatch.from_states(states)
    hidden = torch.randn(len(states), config.lm_hidden_size)
    logits = torch.randn(len(states), config.vocab_size)

    with torch.no_grad():
        output = scem(hidden, batch, return_attention=True)
        adjusted_logits = scem.apply_bias(logits, hidden, batch, alpha=0.3)

    print("bias:", tuple(output.bias.shape))
    print("memory:", tuple(output.memory.shape))
    print("attention:", tuple(output.attention_weights.shape))
    print("adjusted_logits:", tuple(adjusted_logits.shape))


if __name__ == "__main__":
    main()
