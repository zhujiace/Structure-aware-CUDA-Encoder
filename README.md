# SCEM: Structure-aware CUDA Encoding Module

SCEM is a lightweight structure-aware module for text-to-CUDA generation. It does not replace the backbone code LLM. Instead, it reads the current decoder hidden state and a compact CUDA prefix state, then produces a vocabulary-sized logits bias:

```text
z_final = z_lm + alpha * b_scem
```

The current repository uses local Qwen3.5-0.8B as the default integration target, while the core SCEM module is model-agnostic. Larger local Qwen3.5 backbones are stored under `/data/projects/scem/models/` to avoid filling the user home directory.

## Goal

Text-to-CUDA models often produce code that is semantically close but structurally wrong:

- missing or incorrect boundary guards
- index expressions that do not match tensor dimensions
- shared memory used in the wrong order
- missing or misplaced `__syncthreads()`
- write-back too early or in the wrong region

SCEM adds a small trainable module at decoding time to bias token probabilities toward structurally reasonable CUDA code.

## Repository Layout

```text
scem/
  config.py             SCEMConfig and model-size parameters
  model.py              Core SCEM network
  states.py             CUDA prefix state extraction
  decoding.py           Hugging Face LogitsProcessor for decoding-time bias
  qwen_integration.py   Qwen/HF causal-LM integration helpers

scripts/
  demo.py               Interactive CUDABench generation/debug script
  eval.py               CUDABench compile/functionality evaluation
  train.py              Region-aware multi-point SFT for SCEM
  smoke_test.py         Shape smoke test for SCEM
  utils.py              Shared script utilities for CUDABench and generation
data/train.json         Current CUDA SFT-style training data
```

## Core Architecture

At each decoding step:

1. Take the current final-layer hidden state `h_t` from the backbone LM.
2. Decode the current prefix and extract CUDA program state.
3. Encode the CUDA state into memory slots `M_t`.
4. Run cross-attention from `h_t` to `M_t`.
5. Fuse `[h_t; c_t]` with an MLP.
6. Produce a vocab-sized bias `b_t`.
7. Add `alpha * b_t` to the backbone logits.

The implemented form is:

```text
M_t = {e_1, ..., e_k}
c_t = CrossAttention(h_t, M_t)
u_t = MLP([h_t; c_t])
b_t = W_b u_t
z_final = z_lm + alpha * b_t
```

The `[h_t; c_t]` operation is feature fusion, not a residual connection. The residual-style correction happens at logits level through `z_lm + alpha * b_t`.

## SCEM Config Parameters

`SCEMConfig` lives in `scem/config.py`.

Backbone-dependent parameters:

- `lm_hidden_size`: hidden size of the backbone LM final hidden state.
- `vocab_size`: tokenizer/model vocabulary size. SCEM bias must match this dimension.

Internal SCEM dimensions:

- `state_dim`: initial embedding dimension for each CUDA state feature.
- `memory_dim`: dimension of encoded CUDA memory slots.
- `context_dim`: cross-attention query/key/value working dimension.
- `fusion_hidden_dim`: hidden size of the fusion MLP.
- `fusion_dim`: fused representation size before the bias head.
- `num_attention_heads`: number of cross-attention heads.
- `dropout`: dropout inside SCEM.

Bias head parameters:

- `bias_rank`: low-rank bottleneck for vocab bias. `None` uses a full `fusion_dim -> vocab_size` projection.
- `max_bias`: clamps bias magnitude with `tanh` to avoid overwhelming the backbone logits.

CUDA state vocabulary parameters:

- `num_task_families`: task-family embedding size.
- `max_tensor_rank`: max tensor-rank embedding index.
- `num_program_regions`: program-region embedding size.
- `num_static_flags`: number of static CUDA flags.
- `num_prefix_flags`: number of dynamic prefix flags.

For Hugging Face models, use:

```python
scem_config = SCEMConfig.from_lm_config(model.config)
```

This reads `hidden_size` and `vocab_size` from either `config` or `config.text_config`.

## Local Backbone Models

The default 0.8B model remains under the repository-local ignored `models/` directory. Larger downloaded models should use the shared data path:

```text
/data/projects/scem/models/Qwen3.5-4B
/data/projects/scem/models/Qwen3.5-9B
```

Both models are compatible with the current Hugging Face/Qwen integration path:

```text
Qwen3.5-4B: hidden_size=2560, vocab_size=248320
Qwen3.5-9B: hidden_size=4096, vocab_size=248320
```

Use them by passing `--model-path`:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --task-id 0 \
  --use-scem-prompt
```

SCEM checkpoints are backbone-shape specific. A checkpoint trained with Qwen3.5-0.8B cannot be loaded with Qwen3.5-4B or Qwen3.5-9B because the hidden-state dimension changes, even though the tokenizer vocabulary size is the same.

## CUDA State Extraction

`scem/states.py` currently uses a lightweight heuristic prefix scanner, not a full CUDA parser.

It extracts:

- task family
- tensor rank
- current program region
- whether a guard may be needed
- whether shared memory may be needed
- whether synchronization may be needed
- whether index expressions have appeared
- whether a guard appears open
- whether shared memory has appeared
- whether `__syncthreads()` has appeared
- whether write-back has started
- brace/statement stability

Current program regions:

```text
unknown
signature
setup
indexing
guard
shared_memory
compute
write_back
```

This is intentionally cheap enough to run during decoding. It should eventually be replaced or augmented with a stronger CUDA parser/state machine.

## Generation

`scripts/demo.py` is a lightweight interactive test script for CUDABench tasks. It:

- loads one or more tasks from `external/CUDABench/Datasets/CUDABench-Set.jsonl`
- builds the CUDABench prompt
- prints the full model response
- prints the extracted CUDA code block
- optionally runs compile and functionality checks
- optionally enables SCEM during generation

Example: inspect one task without SCEM:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py \
  --task-id 0 \
  --level level3_prompt \
  --max-new-tokens 512
```

Example: inspect several tasks and run compile/functionality checks:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py \
  --task-ids 0,1,2 \
  --check-compile \
  --check-functionality
```

Example: enable SCEM with a checkpoint:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py \
  --task-id 0 \
  --enable-scem \
  --scem-checkpoint ./checkpoints/scem_qwen35/step-1000/scem.pt \
  --alpha 0.3
```

Example: keep the external CUDABench task prompt unchanged but enable SCEM-side system constraints:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py \
  --task-id 0 \
  --use-scem-prompt \
  --max-new-tokens 128
```

Important arguments:

- `--task-id`: single CUDABench task id.
- `--task-ids`: comma-separated task ids.
- `--start-id`, `--end-id`: id range selection.
- `--limit`: evaluate the first N tasks if no explicit ids are given.
- `--check-compile`: run `nvcc` on the extracted CUDA code.
- `--check-functionality`: run `gen.py -> kernel -> compare.py`.
- `--use-scem-prompt`: enable SCEM-side supplemental system constraints without editing the external CUDABench prompt template.
- `--enable-scem`: enable SCEM during generation.
- `--scem-checkpoint`: trained `scem.pt` path. If omitted while `--enable-scem` is set, the script uses an untrained zero-effect SCEM module.
- `--alpha`: SCEM bias strength.
- `--keep-temp`: keep compile/run temporary directories for debugging.

Qwen3.5 generation defaults in the script:

```python
{
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
}
```

The local Hugging Face path does not implement an extra custom `presence_penalty`.

## CUDABench Evaluation

`scripts/eval.py` generates CUDA programs for `external/CUDABench/Datasets/CUDABench-Set.jsonl` and evaluates only:

- compile accuracy
- functionality accuracy

It intentionally skips efficiency/NCU metrics.

The script uses the CUDABench git submodule under `external/CUDABench`. After cloning this repository, initialize it with:

```bash
git submodule update --init --recursive
```

The script reuses the CUDABench prompt format and the same functional validation sequence:

```text
nvcc generated kernel.cu
run dataset gen.py
run compiled kernel
run dataset compare.py
```

If `compare.py` prints `F`, functionality is counted as failed.

### Baseline Evaluation

By default, no SCEM checkpoint is loaded. This evaluates the plain Qwen3.5-0.8B backbone and is the first baseline:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path ./models/Qwen3.5-0.8B \
  --output-dir ./eval_outputs/qwen35_baseline \
  --level level3_prompt \
  --num-samples 1
```

For a quick smoke test:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --output-dir /tmp/scem_cudabench_smoke \
  --limit 1 \
  --max-new-tokens 4
```

To run a first-pass 4B backbone baseline by evaluating one task from each five-task CUDABench group:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --output-dir ./eval_outputs/qwen35_4b_baseline_stride5 \
  --level level3_prompt \
  --task-stride 5 \
  --num-samples 1 \
  --use-scem-prompt
```

### SCEM Evaluation

After training, pass a SCEM checkpoint:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path ./models/Qwen3.5-0.8B \
  --scem-checkpoint ./checkpoints/scem_qwen35/step-1000/scem.pt \
  --output-dir ./eval_outputs/scem_qwen35_step1000 \
  --level level3_prompt \
  --num-samples 1 \
  --alpha 0.3
```

To evaluate the same model with extra SCEM-side system constraints enabled:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path ./models/Qwen3.5-0.8B \
  --output-dir ./eval_outputs/qwen35_scem_prompt_limit5 \
  --level level3_prompt \
  --limit 5 \
  --max-new-tokens 128 \
  --use-scem-prompt
```

### Evaluation Outputs

The output directory contains:

```text
generated_results.jsonl   raw model responses and extracted code
eval_results.jsonl        per-task compile/functionality booleans
summary.json              aggregate metrics
temp_eval/                temporary compile/run directories, only kept with --keep-temp
```

In `eval_results.jsonl`, compile and functionality fields are written before long fields such as `prompt`, `code*`, and `response*` so results remain easy to inspect in IDEs.

`summary.json` reports both version-level and task-level metrics:

- `compile_accuracy`: compiled samples / total samples
- `functionality_accuracy`: functional samples / total samples
- `task_compile_pass_rate`: tasks with at least one compiling sample / total tasks
- `task_functionality_pass_rate`: tasks with at least one functional sample / total tasks

### Re-evaluating Existing Generations

To skip generation and only rerun compile/functionality validation:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --trust-generated \
  --results-jsonl ./eval_outputs/qwen35_baseline/generated_results.jsonl \
  --output-dir ./eval_outputs/qwen35_baseline_recheck \
  --num-samples 1
```

If CUDABench is stored somewhere else, pass:

```bash
--cudabench-root /path/to/CUDABench
```

## Training

Training is supervised next-token fine-tuning over the adjusted logits:

```text
loss = CE(z_lm + alpha * b_scem, next_token)
```

By default, the backbone LM is frozen and only SCEM is trained. This is the recommended first stage, especially for future 7B+ models.

Optional LoRA training is available:

```bash
--use-lora
```

When LoRA is enabled, trainable parameters are LoRA adapters plus SCEM.

### Region-aware Multi-point SFT

Each CUDA sample is expanded into multiple prefix next-token training points. Instead of sampling one random token per sample, the dataset finds CUDA structure anchors and creates training points around regions such as:

- kernel signature
- indexing
- guard
- shared memory
- synchronization
- write-back
- statement close

Current anchors include:

```text
__global__, __device__, __host__
threadIdx., blockIdx., blockDim., int idx, int tid, int row, int col
if (, if(, &&, ||
extern __shared__, __shared__
__syncthreads
] =, ]=, atomicAdd, atomicMax, atomicMin
;, }
```

Additional random points are added as regularization.

### Training Data

`scripts/train.py` supports these record formats:

JSONL or JSON object/list with `text`:

```json
{"text": "```cpp\n__global__ void ...\n```"}
```

Prompt/completion:

```json
{"prompt": "Write a CUDA vector add kernel.", "completion": "```cpp\n__global__ void ...\n```"}
```

Chat messages:

```json
{"messages": [{"role": "user", "content": "Write a CUDA kernel."}, {"role": "assistant", "content": "```cpp\n...\n```"}]}
```

Instruction/input/output, used by the current `data/train.json`:

```json
{"instruction": "...", "input": "...", "output": "```cpp\n__global__ void ...\n```"}
```

For prompt/completion, messages, and instruction/output records, training targets are restricted to the assistant/completion portion.

### Current Dataset

The current `data/train.json` contains 6278 records with fields:

```text
instruction
input
output
```

All checked records include a CUDA code fence, `__global__`, and `cuda_runtime.h`. With the default region-aware settings, it expands to about 45k training points.

Token lengths are often above 2048, so `--max-length 4096` is recommended if memory allows.

### Recommended Training Command

Freeze backbone and train only SCEM:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/train.py \
  --train-file data/train.json \
  --model-path ./models/Qwen3.5-0.8B \
  --output-dir ./checkpoints/scem_qwen35 \
  --max-length 4096 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --region-points-per-example 8 \
  --random-points-per-example 2
```

LoRA + SCEM:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/train.py \
  --train-file data/train.json \
  --model-path ./models/Qwen3.5-0.8B \
  --output-dir ./checkpoints/scem_qwen35_lora \
  --max-length 4096 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --use-lora \
  --gradient-checkpointing
```

### Training Arguments

Data and sequence:

- `--train-file`: JSON or JSONL training file.
- `--max-length`: max tokenized length after prompt + completion formatting.
- `--min-prefix-length`: minimum prefix length before a target token can be sampled.
- `--region-points-per-example`: max region-aware points per raw sample.
- `--random-points-per-example`: random points per raw sample.

Optimization:

- `--batch-size`: per-step batch size.
- `--grad-accum-steps`: gradient accumulation steps.
- `--epochs`: training epochs.
- `--lr`: learning rate.
- `--weight-decay`: AdamW weight decay.
- `--warmup-ratio`: cosine scheduler warmup ratio.
- `--save-steps`: checkpoint interval.
- `--log-steps`: logging interval.

SCEM:

- `--alpha`: training-time SCEM bias scale.
- `--bias-rank`: low-rank vocab bias rank.
- `--task-family`: task family passed to the CUDA state extractor.
- `--tensor-rank`: tensor rank passed to the CUDA state extractor.

Backbone:

- `--freeze-backbone / --no-freeze-backbone`: freeze or unfreeze backbone parameters.
- `--use-lora`: add LoRA adapters through PEFT.
- `--lora-r`, `--lora-alpha`, `--lora-dropout`: LoRA hyperparameters.
- `--gradient-checkpointing`: enable checkpointing for larger models.

## Checkpoints

Training saves checkpoints under:

```text
<output-dir>/step-<global_step>/
  scem.pt
  training_args.json
  tokenizer files
  lora/                only when --use-lora is enabled
```

`scem.pt` contains:

```python
{
    "config": scem.config,
    "state_dict": scem.state_dict(),
}
```

## Smoke Tests

SCEM shape test:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/smoke_test.py
```

Short generation test:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py --max-new-tokens 4
```

Training smoke test requires a small JSON/JSONL file:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/train.py \
  --train-file data/train.json \
  --output-dir /tmp/scem_smoke \
  --max-length 512 \
  --region-points-per-example 2 \
  --random-points-per-example 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --epochs 1 \
  --save-steps 0 \
  --log-steps 1
```

## Model Portability

The core files `config.py`, `model.py`, and `states.py` are not Qwen-specific. To use another Hugging Face causal LM, the integration layer must provide:

- final hidden state for the current decoding step
- logits with shape `[batch, vocab_size]`
- tokenizer decoding for prefix state extraction

The SCEM checkpoint is tied to the backbone tokenizer vocabulary. If the vocab changes, the bias head shape and learned token biases are not directly reusable.

## Current Limitations

- CUDA state extraction is heuristic and string-based.
- Shared-memory writes can be confused with global write-back.
- Region detection is not a full parser.
- SCEM currently biases next-token logits; it does not verify compilation or runtime correctness.
- Untrained SCEM starts with near-zero bias and should not be expected to improve generation before training.

## Maintenance Note

When code behavior, CLI arguments, training data format, or checkpoint format changes, update this README in the same change.
