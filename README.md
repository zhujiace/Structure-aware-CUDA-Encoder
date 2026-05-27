# SCEM: Structure-aware CUDA Encoding Module

SCEM is a lightweight structure-aware module for text-to-CUDA generation. It does not replace the backbone code LLM. Instead, it reads the current decoder hidden state and a tree-sitter CUDA AST graph for the generated prefix, then produces a vocabulary-sized logits bias:

```text
z_final = z_lm + b_scem
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
  states.py             tree-sitter-cuda AST graph extraction
  decoding.py           Hugging Face LogitsProcessor for decoding-time bias
  qwen_integration.py   Qwen/HF causal-LM integration helpers

scripts/
  cuda_ast_viewer.py    Local browser/CLI app for live CUDA AST inspection
  demo.py               Interactive CUDABench generation/debug script
  eval.py               CUDABench compile/functionality evaluation
  harness_eval.py       Kernel-only CUDABench evaluation with fixed bench.cu harness
  train.py              Region-aware multi-point SFT for SCEM
  train_two_stage.py    Dense SCEM structural warmup plus harness adaptation
  smoke_test.py         Shape smoke test for SCEM
  utils.py              Shared script utilities for CUDABench and generation
data/train.json         Current CUDA SFT-style training data
```

## Core Architecture

At each decoding step:

1. Take the current final-layer hidden state `h_t` from the backbone LM.
2. Decode the generated assistant/code prefix and parse it with `tree_sitter_cuda`.
3. Convert the full AST into a typed graph with node and edge embeddings.
4. Encode the graph with an edge-aware Graph Transformer.
5. Pool AST nodes into learned memory tokens `M_t`.
6. Run cross-attention from `h_t` to `M_t`.
7. Produce a vocab-sized bias `b_t` from the cross-attended AST context.
8. Add `b_t` to the backbone logits.

The default implemented form is:

```text
G_t = tree_sitter_cuda(prefix_t)
H_t = EdgeAwareGraphTransformer(G_t)
M_t = LearnedQueryPool(H_t)
c_t = CrossAttention(h_t, M_t)
u_t = tanh(W_c LN(c_t))
b_t = W_b u_t
z_final = z_lm + b_t
```

The AST graph encoder is part of SCEM and is trained end-to-end with the context-only bias head. The final bias head does not receive a direct hidden-state input; the hidden state only queries AST memory through cross-attention. The code keeps an optional `--alpha` scale for diagnostic experiments, but its default is `1.0` and standard training/evaluation commands do not need to set it.

## SCEM Config Parameters

`SCEMConfig` lives in `scem/config.py`.

The default values are chosen for the Qwen3.5-4B experiments. If the SCEM
architecture itself needs to change, edit `SCEMConfig` directly instead of
passing many model-internal flags through `scripts/train.py`.

Backbone-dependent parameters:

- `lm_hidden_size`: hidden size of the backbone LM final hidden state.
- `vocab_size`: tokenizer/model vocabulary size. SCEM bias must match this dimension.

AST graph encoder dimensions:

- `ast_dim`: node hidden dimension inside the edge-aware Graph Transformer.
- `ast_ffn_dim`: feed-forward dimension inside each graph layer.
- `ast_layers`: number of graph transformer layers.
- `ast_heads`: number of graph attention heads.
- `ast_memory_slots`: number of learned AST memory tokens returned to SCEM.
- `ast_node_type_vocab_size`, `ast_edge_type_vocab_size`, `ast_text_vocab_size`: hash embedding table sizes for parser-native node types, edge types, and leaf text.
- `ast_max_nodes`, `ast_max_edges`: truncation caps for padded AST graph batches.
- `ast_max_depth`, `ast_max_child_index`: embedding caps for AST depth and child order.
- `ast_node_flag_dim`, `ast_node_position_dim`: dimensions for AST node flags and source/cursor position features.

SCEM fusion dimensions:

- `memory_dim`: dimension of AST memory slots.
- `context_dim`: cross-attention query/key/value working dimension.
- `num_attention_heads`: number of hidden-to-AST-memory cross-attention heads.
- `num_scem_queries`: number of LM-hidden query tokens used to read AST memory.
- `dropout`: dropout inside SCEM.

Bias head parameters:

- `bias_rank`: low-rank bottleneck for vocab bias. `None` uses `context_dim` as the bias rank.
- `max_bias`: clamps bias magnitude with `tanh` to avoid overwhelming the backbone logits.

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

SCEM checkpoints are architecture- and backbone-shape specific. A checkpoint trained with Qwen3.5-0.8B cannot be loaded with Qwen3.5-4B or Qwen3.5-9B because the hidden-state dimension changes, even though the tokenizer vocabulary size is the same. Checkpoints trained before the AST graph encoder or before the context-only bias head are not compatible with the current SCEM architecture and must be retrained.

## CUDA State Extraction

`scem/states.py` uses `tree_sitter_cuda` to parse the generated CUDA prefix into a full AST graph. During generation, the fixed prompt/chat input is stripped before parsing so the graph focuses on the active generated code state.

AST nodes keep parser-native information:

```text
node type
named / anonymous flag
error / missing flag
source span position
AST depth
child index
optional leaf text hash
```

AST edges keep typed structural relations:

```text
child
parent
next_sibling
prev_sibling
field:left
field:right
field:operator
field:condition
field:body
field:declarator
...
```

The model uses these edge types directly through edge-aware graph attention. No hand-written CUDA metric vector is used in the current SCEM state path.

For prefixes that have not entered a CUDA code block or CUDA construct yet,
the extractor returns an inactive AST state instead of parsing natural language
as code. SCEM masks inactive states to zero bias, so pure-text stages are left
to the backbone while code-generation stages can use CUDA structure.

The graph batch also marks the generation frontier: nodes near the current
cursor, cursor ancestors, source length, and cursor distance are exposed as
node flags/position features. These features help SCEM distinguish whether the
prefix is at a signature, inside a block, near an incomplete expression, or at
the end of a statement.

For a lightweight interactive AST display independent of model training or
generation, run:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/cuda_ast_viewer.py
```

Then open the printed local URL. The page reparses on every edit and displays
the current `tree_sitter_cuda` AST plus whether the previous parse tree was
reused incrementally. For terminal-only inspection:

```bash
cat kernel.cu | /home/zhujiace/anaconda3/envs/llama/bin/python scripts/cuda_ast_viewer.py --stdin
```

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
  --scem-checkpoint ./checkpoints/scem_qwen35/step-1000/scem.pt
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
- `--alpha`: optional SCEM bias scale for ablations; default `1.0`.
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
  --level level3_prompt \
  --num-samples 1 \
  --run-name baseline
```

For a quick smoke test:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --limit 1 \
  --max-new-tokens 4 \
  --run-name smoke
```

To run a first-pass 4B backbone baseline by evaluating one task from each five-task CUDABench group:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --level level1_prompt \
  --task-stride 5 \
  --num-samples 1 \
  --use-scem-prompt \
  --run-name firstpass
```

### SCEM Evaluation

After training, pass a SCEM checkpoint:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path ./models/Qwen3.5-0.8B \
  --scem-checkpoint ./checkpoints/scem_qwen35/step-1000/scem.pt \
  --level level3_prompt \
  --num-samples 1 \
  --run-name scem_step1000
```

To evaluate the same model with extra SCEM-side system constraints enabled:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path ./models/Qwen3.5-0.8B \
  --level level3_prompt \
  --limit 5 \
  --max-new-tokens 128 \
  --use-scem-prompt \
  --run-name scem_prompt
```

### Evaluation Outputs

The output directory contains:

```text
generated_results.jsonl   raw model responses and extracted code
eval_results.jsonl        per-task compile/functionality booleans
summary.json              aggregate metrics
temp_eval/                temporary compile/run directories, only kept with --keep-temp
```

If `--output-dir` is omitted, `scripts/eval.py` creates a unique directory under `eval_outputs/<model-name>/` using the prompt level, enabled modes, task subset, optional `--run-name`, and a short date. If the same directory already exists, `_02`, `_03`, ... is appended automatically. Example:

```text
eval_outputs/Qwen3.5-4B/level1_baseline_scemprompt_stride5_firstpass_260511/
```

Because the parent directory already contains the backbone name, keep `--run-name` short and focused on the experiment, such as `astgraph_2stage` or `cudapollute_ep3`.

Pass `--output-dir` only when you intentionally want to use a fixed directory. Fixed directories can resume generation by skipping existing task ids, but repeated experiments may overwrite `eval_results.jsonl` and `summary.json`.

In `eval_results.jsonl`, compile/functionality fields plus run metadata such as `level`, `model_path`, and `scem_checkpoint` are written before long fields such as `prompt`, `code*`, and `response*` so results remain easy to inspect in IDEs.

`summary.json` reports sample-level and pass@k metrics:

- `level`: CUDABench prompt level used for generation.
- `model_path`: backbone model used for generation.
- `sample_compile_accuracy`: compiled samples / total generated samples.
- `sample_functionality_accuracy`: functional samples / total generated samples.
- `compile_pass@1`: tasks whose first generated sample compiles / total tasks.
- `functionality_pass@1`: tasks whose first generated sample is functional / total tasks.
- `compile_pass@k`: tasks with at least one compiling sample among `k = --num-samples` generations / total tasks.
- `functionality_pass@k`: tasks with at least one functional sample among `k = --num-samples` generations / total tasks.

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

## Harness Kernel Evaluation

`scripts/harness_eval.py` is a separate evaluation mode for measuring kernel generation ability without requiring the model to generate `main`.

It uses each CUDABench record's `bench.cu` as a fixed harness:

- removes the reference `__global__` kernel from `bench.cu`
- uses a harness-specific system/user prompt instead of the CUDABench generation prompt
- prompts the model with the task spec, required kernel signature, and fixed `main`
- asks the model to output only the replacement `__global__` kernel and optional `__device__` helpers
- inserts the generated kernel back into the fixed harness
- compiles and runs the same `gen.py -> executable -> compare.py` validation
- extracts the generated code by preferring code blocks that contain `__global__` and the required kernel name, so explanatory snippets are not compiled as kernels
- uses the shared configurable code-block stopping criterion to stop only after a matching kernel block; standalone `eval.py` keeps the default first-complete-code-block behavior

This should be interpreted separately from standalone `eval.py`: harness evaluation reduces noise from missing `main`/I/O boilerplate and focuses more directly on kernel logic, indexing, guards, synchronization, and write-back behavior.

Example 4B harness baseline:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/harness_eval.py \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --level level1_prompt \
  --task-stride 5 \
  --num-samples 1 \
  --run-name qwen35_4b_harness_level1_stride5
```

If `--output-dir` is omitted, it creates a unique dated directory under `eval_outputs/<model-name>/`, similar to `scripts/eval.py`.

Harness generation can run on 1-4 GPUs. With normal `python`, generation is single-process. With `accelerate launch --num_processes N`, tasks are automatically sharded across ranks, each rank writes `generated_results.rank<N>.jsonl`, and rank 0 merges the shards into the standard `generated_results.jsonl` format before evaluation. Add `--generate-only` to stop after generation and skip compile/functionality checks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/zhujiace/anaconda3/envs/llama/bin/accelerate launch --num_processes 4 scripts/harness_eval.py \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --level level1_prompt \
  --task-stride 5 \
  --num-samples 3 \
  --generate-only \
  --output-dir eval_outputs/Qwen3.5-4B/level1_harness_parallel_gen
```

## Training

Training is supervised next-token fine-tuning over the adjusted logits:

```text
loss = CE(z_lm + b_scem, next_token)
```

By default, the backbone LM is frozen and only SCEM is trained. This is the recommended first stage, especially for future 7B+ models. The optional `--alpha` implementation parameter defaults to `1.0`; regular experiments should leave it unset.

Optional LoRA training is available:

```bash
--use-lora
```

When LoRA is enabled, trainable parameters are LoRA adapters plus SCEM.

### Region-aware Multi-point SFT

Each CUDA sample is expanded into multiple prefix next-token training points. Instead of sampling one random token per sample, the dataset finds CUDA structure anchors and creates training points just after those anchors so the state prefix already contains the relevant CUDA structure. The tracked regions include:

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

Additional random points are added as regularization. For prompt/completion, messages, and instruction/output records, SCEM state extraction uses only the assistant/completion prefix, not the user prompt.

### Two-stage SCEM Training

`scripts/train_two_stage.py` trains SCEM in two phases without modifying the backbone: a dense structural warmup with many target points per raw example, followed by a lower-learning-rate harness adaptation stage. Unlike `scripts/train.py`, it runs one frozen-backbone forward per raw example and gathers multiple target positions from that forward pass, which is more efficient for dense prefix-point training.

Pretrain checkpoints are saved under `<output-dir>/pretrain/`; adapt saves `<output-dir>/adapt-best/`, `<output-dir>/best/`, and `<output-dir>/final/`.

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

Token lengths are often above 2048. Measured with the Qwen3.5 tokenizer:

```text
p50=2122, p75=2422, p90=2821, p95=3105, p99=3667, max=5803
```

Use `--max-length 4096` when memory allows; it covers about 99.6% of the current records. Use `--max-length 3072 --skip-overlength` for a memory-friendlier run that avoids training on truncated examples and still covers about 94.5% of records. `--max-length 2048` is mainly for smoke tests because it covers only about 41.5% of records.

### Recommended Training Command

SCEM-only 4B, frozen backbone:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/zhujiace/anaconda3/envs/llama/bin/accelerate launch --num_processes 4 scripts/train.py \
  --train-file data/train.json \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --output-dir /data/projects/scem/checkpoints/scem_qwen35_4b_scem_only \
  --max-length 4096 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 1 \
  --region-points-per-example 8 \
  --random-points-per-example 2 \
  --val-ratio 0.05 \
  --mixed-precision bf16 \
  --model-dtype bfloat16 \
  --save-steps 200 \
  --log-steps 10
```

SCEM-only 9B, frozen backbone:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/zhujiace/anaconda3/envs/llama/bin/accelerate launch --num_processes 4 scripts/train.py \
  --train-file data/train.json \
  --model-path /data/projects/scem/models/Qwen3.5-9B \
  --output-dir /data/projects/scem/checkpoints/scem_qwen35_9b_scem_only \
  --max-length 3072 \
  --skip-overlength \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 1 \
  --region-points-per-example 8 \
  --random-points-per-example 2 \
  --val-ratio 0.05 \
  --mixed-precision bf16 \
  --model-dtype bfloat16 \
  --save-steps 200 \
  --log-steps 10
```

LoRA + SCEM 4B:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/zhujiace/anaconda3/envs/llama/bin/accelerate launch --num_processes 4 scripts/train.py \
  --train-file data/train.json \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --output-dir /data/projects/scem/checkpoints/scem_qwen35_4b_lora \
  --max-length 3072 \
  --skip-overlength \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 1 \
  --region-points-per-example 8 \
  --random-points-per-example 2 \
  --val-ratio 0.05 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --mixed-precision bf16 \
  --model-dtype bfloat16 \
  --save-steps 200 \
  --log-steps 10
```

LoRA + SCEM 9B:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/zhujiace/anaconda3/envs/llama/bin/accelerate launch --num_processes 4 scripts/train.py \
  --train-file data/train.json \
  --model-path /data/projects/scem/models/Qwen3.5-9B \
  --output-dir /data/projects/scem/checkpoints/scem_qwen35_9b_lora \
  --max-length 3072 \
  --skip-overlength \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 1 \
  --region-points-per-example 8 \
  --random-points-per-example 2 \
  --val-ratio 0.05 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --mixed-precision bf16 \
  --model-dtype bfloat16 \
  --save-steps 200 \
  --log-steps 10
```

Set `CUDA_VISIBLE_DEVICES` before `accelerate launch` to choose a subset of GPUs, for example `CUDA_VISIBLE_DEVICES=0,1` with `--num_processes 2`.

For a safer 9B LoRA pilot before a full run, replace the corresponding values above with these smaller settings and add the sample limits:

```bash
--max-raw-examples 100 \
--max-training-points 200 \
--max-length 2048 \
--lora-r 4 \
--lora-alpha 8
```

Previously verified smoke paths from the pre-AST state layout. These are historical only and are not compatible with the current AST graph SCEM:

```text
4B SCEM-only single GPU: /data/projects/scem/checkpoints/smoke/scem_qwen35_4b_scem_only_smoke/step-1/scem.pt
4B SCEM-only two-GPU DDP: /data/projects/scem/checkpoints/smoke/scem_qwen35_4b_scem_only_ddp_smoke/step-1/scem.pt
9B SCEM-only single GPU: /data/projects/scem/checkpoints/smoke/scem_qwen35_9b_scem_only_smoke/step-1/scem.pt
9B LoRA two-GPU DDP: /data/projects/scem/checkpoints/smoke/scem_qwen35_9b_lora_ddp_smoke/step-1/scem.pt
```

### Training Arguments

Data and sequence:

- `--train-file`: JSON or JSONL training file.
- `--max-length`: max tokenized length after prompt + completion formatting.
- `--skip-overlength`: skip examples longer than `--max-length` instead of truncating them.
- `--max-raw-examples`: read only the first N raw records, intended for smoke tests.
- `--max-training-points`: keep only the first N expanded training points, intended for smoke tests.
- `--val-ratio` / `--var-ratio`: reserve a fraction of raw records for validation loss tracking. `--var-ratio` is accepted only as a compatibility alias.
- `--train-output-dir`: directory for training logs and loss curves, default `train_outputs`.
- `--train-run-name`: optional run subdirectory name under `--train-output-dir/<model-name>`.
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

- `--alpha`: optional SCEM bias scale for ablations; default `1.0`.
- `--ast-cache-dir`: directory for cached AST graph tensors. Defaults to `train_outputs/ast_cache`; pass an empty string to disable.
- `--state-contrastive-weight`: weight for the true-state vs corrupted-state margin loss. Set `0` to disable.
- `--state-contrastive-margin`: required CE margin between true and corrupted state.
- `--state-contrastive-mode`: corrupted-state source: `zero_all`, `shuffle`, `both`, or `none`.

The default SCEM training objective is now state-sensitive:

```text
loss = CE(z_lm + b(true_state), y)
     + lambda * max(0, margin + CE_true - CE_corrupted)
```

`zero_all` is the default corrupted state because common training commands use batch size 1, where in-batch shuffling would otherwise be ineffective.

For post-training bias inspection, `scripts/inspect_bias.py --state-ablation` supports `true`, `zero_all`, `blank`, and `shuffled`. Use `blank` when you need a truly inactive SCEM state with all node and edge masks cleared; `zero_all` keeps the graph active while clearing AST features.

Backbone:

- `--freeze-backbone / --no-freeze-backbone`: freeze or unfreeze backbone parameters.
- `--use-lora`: add LoRA adapters through PEFT.
- `--lora-r`, `--lora-alpha`, `--lora-dropout`: LoRA hyperparameters.
- `--gradient-checkpointing`: enable checkpointing for larger models.
- `--mixed-precision`: Accelerate mixed precision mode, usually `bf16` on A40.
- `--model-dtype`: optional explicit model load dtype.

## Checkpoints

Training saves regular step checkpoints under:

```text
<output-dir>/step-<global_step>/
  scem.pt
  metrics.json
  training_args.json
  tokenizer files
  lora/                only when --use-lora is enabled
```

It also saves:

```text
<output-dir>/final/     final training step
<output-dir>/best/      lowest validation loss, only when --val-ratio > 0
```

Validation loss is computed at `--save-steps` intervals and once at the final step.

## Training Outputs

Each training run writes lightweight logs under:

```text
train_outputs/<model-name>/<run-name>/
  metrics.jsonl       append-only train/validation events
  metrics.csv         same metrics in spreadsheet-friendly format
  summary.json        final_step, best_step, best_val_loss, output paths
  training_args.json  exact CLI arguments
  figs/
    train_loss.png    train-loss curve, generated when matplotlib is available
    val_loss.png      validation-loss curve, generated when matplotlib is available
```

If `--train-run-name` is omitted, the run name is derived from `--output-dir` and a timestamp. Training logs are grouped by backbone under `<model-name>`, so prefer concise method/data names and avoid repeating the backbone name in `--train-run-name`.

Useful inspection commands:

```bash
tail -n 20 train_outputs/<model-name>/<run-name>/metrics.csv
cat train_outputs/<model-name>/<run-name>/summary.json
```

Open `figs/train_loss.png` and `figs/val_loss.png` to inspect the training and validation loss trends separately. The plots are updated at validation events and at the final step; train-only log events append CSV/JSONL rows without redrawing images. Training loss points are logged every `--log-steps`; validation loss points are logged every `--save-steps` and at the final step. The `best_step` in `summary.json` is the checkpoint step copied to `<output-dir>/best/`.

Load a LoRA adapter during demo/evaluation with `--lora-checkpoint <checkpoint>/lora`.

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
  --output-dir /data/projects/scem/checkpoints/smoke/scem_smoke \
  --max-length 512 \
  --max-raw-examples 4 \
  --max-training-points 4 \
  --val-ratio 0.25 \
  --region-points-per-example 2 \
  --random-points-per-example 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --epochs 1 \
  --save-steps 1 \
  --log-steps 1
```

## Model Portability

The core files `config.py`, `model.py`, and `states.py` are not Qwen-specific. To use another Hugging Face causal LM, the integration layer must provide:

- final hidden state for the current decoding step
- logits with shape `[batch, vocab_size]`
- tokenizer decoding for prefix state extraction

The SCEM checkpoint is tied to the backbone tokenizer vocabulary. If the vocab changes, the bias head shape and learned token biases are not directly reusable.

## Current Limitations

- `tree_sitter_cuda` can expose error/missing nodes for incomplete prefixes; the graph encoder must learn to use these as generation-state signals.
- AST graph extraction adds decoding overhead compared with the old heuristic scanner.
- SCEM currently biases next-token logits; it does not verify compilation or runtime correctness.
- Untrained SCEM starts with near-zero bias and should not be expected to improve generation before training.

## Maintenance Note

When code behavior, CLI arguments, training data format, or checkpoint format changes, update this README in the same change.
