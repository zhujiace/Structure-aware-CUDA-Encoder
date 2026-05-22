# AGENTS.md

This file is for future coding agents working in this repository. It captures the current stable project state as of 2026-05-21 and is intended to make the next session pick up quickly without rediscovering local conventions.

## Scope

Project: `SCEM`  
Path: `/home/zhujiace/project/Kernel/SCEM`

Current stage:

- Core SCEM code is in place.
- Training script is in place.
- Inference / demo script is in place.
- CUDABench evaluation script is in place.
- Kernel-only harness evaluation script is in place.
- Training data and benchmark are available locally.
- SCEM state extraction is now based on `tree_sitter_cuda` AST graphs and an edge-aware Graph Transformer. Older heuristic-state checkpoints are incompatible and must be retrained.
- A small AST-SCEM smoke run completed at `/tmp/scem_ast_train_smoke/final/scem.pt`; it only validates code paths and should not be treated as a performance checkpoint.
- A first 4B SCEM-only 3-GPU run produced `/data/projects/scem/checkpoints/scem_qwen35_4b_scem_only_3gpu/step-724/scem.pt`, but it used an older state layout and should not be treated as compatible with current code.
- Baseline experiments have been run for `Qwen3.5-0.8B` and `Qwen3.5-4B`; the 0.8B result is effectively unusable, while 4B has measurable but still low CUDA generation accuracy.

## Repository Structure

Keep this structure stable unless there is a strong reason to change it.

```text
scem/
  __init__.py
  config.py
  decoding.py
  model.py
  qwen_integration.py
  states.py

scripts/
  cuda_ast_viewer.py
  demo.py
  eval.py
  harness_eval.py
  smoke_test.py
  train.py
  train_two_stage.py
  utils.py

utils/
  build_cudabench_pollution.py

external/CUDABench/     git submodule
README.md
AGENTS.md
```

## Ownership and Boundaries

### Files that define the core project

These are first-class project files and should be kept clean and well-structured:

- `scem/config.py`
- `scem/model.py`
- `scem/states.py`
- `scem/decoding.py`
- `scem/qwen_integration.py`
- `scripts/cuda_ast_viewer.py`
- `scripts/train.py`
- `scripts/eval.py`
- `scripts/harness_eval.py`
- `scripts/demo.py`
- `scripts/smoke_test.py`
- `scripts/utils.py`
- `utils/build_cudabench_pollution.py`
- `README.md`
- `AGENTS.md`

### Files or directories that should not be modified casually

- `external/CUDABench/`
  - This is a git submodule.
  - Do not edit benchmark code or dataset files unless the user explicitly asks.
  - Prefer adapting SCEM-side wrappers in `scripts/` instead.

- `data/`
  - Contains local training data.
  - It is gitignored.
  - Do not rewrite or reformat dataset files unless explicitly requested.

- `models/`
  - Local model storage.
  - Gitignored.
  - Do not modify.

- `/data/projects/scem/models/`
  - Shared storage for larger local backbone models.
  - Current downloaded models:
    - `Qwen3.5-4B`
    - `Qwen3.5-9B`
  - Use this path for large model files instead of the user home or repo-local `models/` directory.

- `checkpoints/`
  - Training artifacts.
  - Gitignored.
  - Do not modify existing checkpoints unless explicitly requested.

- `eval_outputs/`
  - Evaluation artifacts.
  - Gitignored.
  - Avoid destructive cleanup unless explicitly requested.

- `train_outputs/`
  - Training logs, CSV/JSONL metrics, summaries, and loss plots.
  - Gitignored.
  - Use this to inspect training/validation loss and best step without relying on terminal scrollback.

- `archive/`
  - Ignored local scratch/old scripts.
  - Do not mention it in README.
  - Do not move production code back into it.

## Git and Worktree Notes

At the time this file was written, the worktree was not fully clean:

- `scripts/demo.py` was modified.
- `external/CUDABench` appears as an uncommitted submodule state.

Do not revert user changes you did not make.

Before touching a file, check whether it already contains local edits:

```bash
git status --short
```

## Python Environment

Expected environment:

- Conda env: `llama`
- Python path previously verified:

```text
/home/zhujiace/anaconda3/envs/llama/bin/python
```

Use that interpreter when running commands in documentation or validation:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python ...
```

## Script Conventions

All user-facing entrypoints live in `scripts/`.

## Output Naming Convention

Use short, experiment-focused names for future checkpoint, train-output, and eval run directories.

- Do not repeat the backbone name in run names by default.
- `eval_outputs/<model-name>/...` already groups evaluation outputs by backbone.
- `train_outputs/<model-name>/...` also groups training logs by backbone.
- Training runs record the full model path in `training_args.json`, so run names can focus on method, data, and key settings.
- Prefer names like `astgraph_2stage_harness`, `astgraph_cudapollute_ep3`, or `lora_scem_harness`.
- Avoid long names like `scem_qwen35_4b_scem_astgraph_train_cudabench_2gpu_ep3` unless the user explicitly asks for backbone-identifying names or multiple backbones are intentionally mixed in one parent directory.

## Experiment Logging Requirement

Every experiment must be recorded manually under `report/` before or immediately after launch. This applies to training, evaluation, analysis, inspection, ablation, debugging experiments, and any other run whose results may later be compared.

- Use one log file per date, named with short date format: `report/YYMMDD.log` (for example, `report/260520.log`).
- Append a new entry for each experiment.
- Include at minimum:
  - an ARS-compatible `Material Passport` block when the experiment may later be used in research writing or comparison
  - experiment date
  - experiment name
  - experiment type (`training`, `evaluation`, `analysis`, etc.)
  - experiment purpose
  - tmux session or execution context when applicable
  - key inputs/checkpoints/models/data
  - output directories
  - complete command exactly as launched
  - notable notes, including failures/restarts or environment fixes
- Descriptive fields such as purpose, notes, and experiment context should be written in Chinese for readability unless the user asks otherwise.
- For ARS-compatible entries, use `Origin Skill: experiment-agent`; choose `Origin Mode` as `plan`, `run`, or `validate` according to the artifact being recorded. `Verification Status` is an artifact-level label (`UNVERIFIED`, `ANALYZED`, or `VERIFIED`), not a live run-status field.
- Prefer the ARS-style section layout: `Material Passport`, `Experiment Overview`, `Setup`, `Inputs`, `Expected Outputs`, `Monitoring Configuration`, `Analysis Plan`, and `Notes`.
- Do not implement this as code or a logging hook. Keep it as an agent workflow requirement and write the log entry directly.

### `scripts/train.py`

Purpose:

- Train SCEM with region-aware multi-point SFT.
- Default behavior is frozen backbone + train SCEM.
- Optional LoRA warm-up path exists.
- Uses Hugging Face Accelerate; launch with `accelerate launch --num_processes N` to use 1-4 GPUs.

Important behavior:

- Supports `text`, `prompt/completion`, `messages`, and `instruction/input/output` formats.
- Supports `.json` and `.jsonl`.
- Uses region anchors plus random points for next-token training.
- CUDA state extraction is AST-graph based. Training state uses the assistant/completion prefix, and generation state strips the fixed prompt/chat input before parsing with `tree_sitter_cuda`. Do not reintroduce global `task_family` / `tensor_rank` fields or the old hand-written CUDA metric vector unless the user explicitly asks for that design.
- AST graph extraction emits parser-native node/edge IDs with hash embeddings, padded by `SCEMConfig.ast_max_nodes` and `SCEMConfig.ast_max_edges`. The SCEM model encodes these graphs with an edge-aware Graph Transformer and learned AST memory queries.
- Model-internal SCEM defaults are set in `scem/config.py` for the 4B path (`bias_rank=256`, multi-query cross attention, AST node cap 768 / edge cap 3072). Do not re-add these as `train.py` CLI flags; edit `SCEMConfig` directly when changing architecture.
- `--ast-cache-dir` defaults to `train_outputs/ast_cache`; pass an empty string to disable AST tensor caching.
- Supports `--skip-overlength`, `--max-raw-examples`, and `--max-training-points`.
- Supports `--val-ratio` with `--var-ratio` as a compatibility alias; validation is split by raw records and used for validation loss tracking.
- Training now saves regular `step-*` checkpoints, `final/`, and `best/` when validation is enabled.
- Training writes run logs under `train_outputs/<model-name>/<run-name>/` by default: `metrics.jsonl`, `metrics.csv`, `summary.json`, `training_args.json`, and separate plots under `figs/` when matplotlib is available.
- For the current `data/train.json`, `--max-length 4096` covers about 99.6% of records; `--max-length 3072 --skip-overlength` covers about 94.6% without training on truncated records.
- Larger checkpoint output directories should use `/data/projects/scem/checkpoints/` to avoid filling the user home directory.
- 4B and 9B LoRA training has been smoke-tested with Accelerate/DDP; current DDP duplicates the full backbone on each GPU, so it improves throughput but does not reduce per-GPU model memory.

### `scripts/train_two_stage.py`

Purpose:

- Train SCEM in two phases without changing the backbone: dense structural warmup followed by harness adaptation.
- Use when SCEM-only one-stage SFT appears too weak or too sparse.
- Runs one frozen-backbone forward per raw example and gathers multiple target positions from that forward pass, making dense prefix-point training more efficient than one forward per sampled prefix.

Important behavior:

- Does not modify eval behavior or `scripts/train.py`.
- Pretrain saves `<output-dir>/pretrain/scem.pt`.
- Adapt saves `<output-dir>/adapt-best/`, `<output-dir>/best/`, and `<output-dir>/final/`.
- Uses the same raw-record validation split for both stages via `--val-ratio` and `--seed`.
- Currently SCEM-only; use `scripts/train.py` for LoRA experiments.

### `scripts/eval.py`

Purpose:

- Full CUDABench evaluation.
- Computes only compile accuracy and functionality accuracy.
- Efficiency metrics are intentionally omitted.

Important behavior:

- Default mode is backbone-only baseline if `--scem-checkpoint` is omitted.
- Uses CUDABench prompt format from the submodule.
- If `--output-dir` is omitted, creates a unique directory under `eval_outputs/<model-name>/` using level, mode flags, optional `--run-name`, and short date such as `260511`; same-day duplicates get `_02`, `_03`, ... suffixes.
- Use `--output-dir` only when intentionally writing to a fixed directory.
- Writes:
  - `generated_results.jsonl`
  - `eval_results.jsonl`
  - `summary.json`

Output field order in `eval_results.jsonl` is intentional:

- compile/functionality results appear before `prompt`
- this makes long JSONL lines easier to inspect in IDEs

### `scripts/harness_eval.py`

Purpose:

- Kernel-only CUDABench evaluation.
- Uses `bench.cu` as a fixed harness by removing the reference `__global__` kernel and inserting the generated kernel in its place.
- Uses harness-specific prompts, not the external CUDABench generation prompt.
- Prompts with the task spec, required kernel signature, and fixed `main` from `bench.cu`.
- Extracts code by requiring a code block with `__global__` and the required kernel name when possible.
- Uses the shared configurable code-block stopping criterion with required substrings; standalone `eval.py` keeps the default stopping behavior.
- Reuses CUDABench `gen.py -> executable -> compare.py` validation.

Important behavior:

- Does not replace `scripts/eval.py`; standalone and harness metrics should be interpreted separately.
- Intended to reduce noise from missing `main`/I/O boilerplate and better isolate CUDA kernel generation ability.
- If `--output-dir` is omitted, creates a unique dated directory under `eval_outputs/<model-name>/`; same-day duplicates get `_02`, `_03`, ... suffixes.
- Supports `--generate-only` to write `generated_results.jsonl` without running compile/functionality checks.
- If launched with `accelerate launch --num_processes N`, generation is automatically sharded across ranks and merged by rank 0 into the same `generated_results.jsonl` format as single-process generation. Non-main ranks do not run the detection phase.

### `scripts/demo.py`

Purpose:

- Lightweight interactive CUDABench testing.
- Prints prompt, full raw response, extracted CUDA code, and optional compile/functionality checks.

Use it for:

- inspecting a single benchmark task
- debugging model output quality
- checking extraction behavior

### `scripts/cuda_ast_viewer.py`

Purpose:

- Lightweight local app for inspecting `tree_sitter_cuda` AST updates as CUDA source changes.
- Independent of SCEM training/evaluation and does not load a language model.
- Supports browser mode by default and `--stdin` for terminal-only AST inspection.

### `scripts/smoke_test.py`

Purpose:

- Minimal structural/shape test for SCEM internals.

### `utils/build_cudabench_pollution.py`

Purpose:

- Convert CUDABench tasks into `data/train.json`-style records for controlled contamination experiments.
- Default mode is harness-aligned: the input prompt matches `scripts/harness_eval.py`, and the output is the reference `__global__` kernel from `bench.cu` wrapped in one cpp block.
- Default outputs are `data/train_harness.json`, `data/cudabench.json`, and `data/train_cudabench.json`.
- `data/train_harness.json` is converted from `data/train.json` by extracting each standalone answer's `__global__` kernel and fixed `main` into a harness-style prompt/answer pair.
- `data/train_cudabench.json` is the harness-style fusion of `data/train_harness.json` plus `data/cudabench.json`; do not rebuild it from standalone `data/train.json` directly.
- Optionally writes a manifest via `--manifest` to preview the `train.py --val-ratio --seed` raw-record validation split. Do not write the manifest under `data/` unless the user asks.
- Use the combined file with `scripts/train.py --val-ratio` so validation records stay out of the training split.

### `scripts/utils.py`

Purpose:

- Shared utility layer for `demo.py` and `eval.py`
- Keep utilities focused and low-dependency
- Prefer adding reusable CUDABench/generation helpers here instead of duplicating code across scripts

Do not turn `utils.py` into a general dumping ground.  
Each helper should have a clear responsibility.

## Stable Run Commands

### Syntax / import checks

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python -B -m py_compile \
  scripts/utils.py \
  scripts/train.py \
  scripts/train_two_stage.py \
  scripts/eval.py \
  scripts/harness_eval.py \
  utils/build_cudabench_pollution.py \
  scripts/cuda_ast_viewer.py \
  scripts/demo.py \
  scripts/smoke_test.py
```

### Help checks

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/train.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/train_two_stage.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/harness_eval.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/cuda_ast_viewer.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py --help
```

### Smoke test

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/smoke_test.py
```

Expected shape-style output:

```text
bias: ...
memory: ...
attention: ...
adjusted_logits: ...
```

### Demo generation

Single task:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py \
  --task-id 0 \
  --level level3_prompt \
  --max-new-tokens 512
```

Compile / functionality spot-check:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/demo.py \
  --task-ids 0,1,2 \
  --check-compile \
  --check-functionality
```

### Baseline evaluation

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --model-path ./models/Qwen3.5-0.8B \
  --level level3_prompt \
  --num-samples 1 \
  --run-name baseline
```

Quick smoke run:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --limit 1 \
  --max-new-tokens 4 \
  --run-name smoke
```

### Training

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

For a safer 9B LoRA pilot before a full run, replace the corresponding values above with these smaller settings and add the sample limits:

```bash
--max-raw-examples 100 \
--max-training-points 200 \
--max-length 2048 \
--lora-r 4 \
--lora-alpha 8
```

Previously verified smoke paths from the pre-AST state layout. These are historical only and are incompatible with the current AST graph SCEM:

```text
4B SCEM-only single GPU: /data/projects/scem/checkpoints/smoke/scem_qwen35_4b_scem_only_smoke/step-1/scem.pt
4B SCEM-only two-GPU DDP: /data/projects/scem/checkpoints/smoke/scem_qwen35_4b_scem_only_ddp_smoke/step-1/scem.pt
9B SCEM-only single GPU: /data/projects/scem/checkpoints/smoke/scem_qwen35_9b_scem_only_smoke/step-1/scem.pt
9B LoRA two-GPU DDP: /data/projects/scem/checkpoints/smoke/scem_qwen35_9b_lora_ddp_smoke/step-1/scem.pt
```

## Known Experimental Status

Current factual status:

- Previous smoke training produced `scem.pt` checkpoints for 4B and 9B, but those checkpoints predate the AST graph encoder and must be retrained for current-code evaluation.
- A 4B SCEM-only 3-GPU training run completed at `/data/projects/scem/checkpoints/scem_qwen35_4b_scem_only_3gpu/step-724/scem.pt`; it predates the AST graph encoder and should not be used with current SCEM architecture.
- Baselines have been tried for `Qwen3.5-0.8B` and `Qwen3.5-4B`.
- The 0.8B baseline did not solve CUDABench tasks in a usable way.
- The best current 4B baseline is kernel-only harness eval at compile 0.26 and functionality 0.19 on `level1_prompt`, `task_stride=5`, before rerunning with the latest stopping changes.
- Larger local backbones are now available under `/data/projects/scem/models/`.
- `Qwen3.5-4B` and `Qwen3.5-9B` are compatible with the current Hugging Face/Qwen integration path.
- `Qwen3.5-4B` SCEM-only DDP and `Qwen3.5-9B` LoRA DDP smoke tests completed successfully before the AST graph state change.
- SCEM checkpoints are backbone-shape specific; retrain SCEM for 4B or 9B instead of reusing a 0.8B SCEM checkpoint.
- SCEM checkpoints are also architecture/state-layout specific; retrain after changes to `scem/config.py`, `scem/model.py`, or `scem/states.py`.
- Therefore, the next session should focus primarily on experiments, not basic infrastructure.

Latest evaluation results currently present under `eval_outputs/`:

| Directory | Model | Mode | Level | Tasks | Compile | Functionality | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `eval_outputs/qwen35_baseline` | `Qwen3.5-0.8B` | standalone | `level1_prompt` | 500 | 0.00 | 0.00 | 0.8B baseline, effectively unusable |
| `eval_outputs/qwen35_4b_baseline_stride5` | `Qwen3.5-4B` | standalone | `level3_prompt` | 100 | 0.09 | 0.03 | older level3 stride-5 baseline |
| `eval_outputs/qwen35_4b_baseline_level1_stride5` | `Qwen3.5-4B` | standalone | `level1_prompt` | 100 | 0.09 | 0.05 | 4B level1 baseline |
| `eval_outputs/Qwen3.5-4B_level1_baseline_scemprompt_stride5_20260509_123436` | `Qwen3.5-4B` | standalone + SCEM prompt only | `level1_prompt` | 100 | 0.14 | 0.06 | no SCEM checkpoint loaded |
| `eval_outputs/Qwen3.5-4B_level1_harness_stride5_20260509_170036` | `Qwen3.5-4B` | harness kernel-only | `level1_prompt` | 100 | 0.15 | 0.08 | older harness baseline |
| `eval_outputs/Qwen3.5-4B_level1_harness_stride5_20260510_223512` | `Qwen3.5-4B` | harness kernel-only | `level1_prompt` | 100 | 0.26 | 0.19 | best current baseline before rerunning with latest stopping changes |

Do not treat `eval_outputs/Qwen3.5-4B_level1_baseline_stride5_limit1_auto_dir_check_20260509_122809` as a valid experiment. It was an output-directory/checking artifact: `generated_results.jsonl` is empty, `limit=1`, but `eval_results.jsonl` has 100 lines and points at another generated-results file.

Recent code/output changes to remember:

- `scripts/eval.py` and `scripts/harness_eval.py` now auto-create output directories under `eval_outputs/<model-name>/`.
- Auto directory names use a short date like `260511`, not a full timestamp. Same-day duplicates get `_02`, `_03`, ... suffixes.
- `scripts/utils.py` has a configurable first-code-block stopping criterion. Harness eval passes required substrings such as `__global__` and the kernel name, while standalone `eval.py` keeps the default first-complete-code-block behavior.
- `scripts/harness_eval.py` prompts for exactly one fenced cpp block containing the replacement `__global__` kernel and any needed helpers in the same block. It should not ask the model to generate `main`.

Recommended next command if the user asks to rerun the latest 4B harness baseline:

```bash
cd /home/zhujiace/project/Kernel/SCEM

CUDA_VISIBLE_DEVICES=0 \
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/harness_eval.py \
  --model-path /data/projects/scem/models/Qwen3.5-4B \
  --level level1_prompt \
  --task-stride 5 \
  --num-samples 1 \
  --run-name baseline
```

Expected output directory shape:

```text
eval_outputs/Qwen3.5-4B/level1_harness_stride5_baseline_260511/
```

This means the next likely tasks are:

1. run and debug training
2. verify checkpoint loading
3. rerun the 4B harness baseline after the latest stopping/prompt changes
4. rerun evaluation with trained SCEM
5. compare trained SCEM against the 4B standalone and harness baselines

## Known Design Decisions

These are intentional and should not be changed casually.

### SCEM integration uses last hidden state only

Current design uses the last layer, last token hidden state:

```python
hidden_states[-1][:, -1, :]
```

This was a deliberate choice for simplicity, cost, and consistency with the next-token logits.

### Decoding path uses Hugging Face `LogitsProcessor`

SCEM bias is injected through decoding-time logits processing, not by deeply rewriting the model forward path.

### Alpha is only an optional implementation scale

Training and evaluation default `--alpha` to `1.0`, and normal commands should leave it unset. Treat SCEM as predicting the bias magnitude directly; use `--alpha` only for explicit ablation/debug experiments.

### Default SCEM state path is AST graph based

The default SCEM state path now parses the generated CUDA prefix with `tree_sitter_cuda`, converts the full AST into a typed graph, encodes it with an edge-aware Graph Transformer, and pools learned AST memory tokens for multi-query hidden-state cross attention. The old 7-slot heuristic state encoder and `--bias-arch concat` path are no longer part of the main code path.

Training defaults include a true-state vs corrupted-state margin term (`--state-contrastive-weight`, `--state-contrastive-margin`, `--state-contrastive-mode`) so SCEM is explicitly pressured to make the correct CUDA state outperform a corrupted state. Use `--state-contrastive-weight 0` only for ablation/debug runs.

### Region-aware multi-point SFT is intentional

Training no longer samples only one random point per example.  
It expands each CUDA example into multiple next-token points based on region anchors.

### AST state extractor uses generated prefix

The current SCEM state no longer includes manually supplied task-family or tensor-rank fields, heuristic static flags, or normalized scalar CUDA metrics. Dynamic code state is derived from the active generated CUDA code block/prefix. Pure text before CUDA code starts is represented as an inactive AST state, and SCEM masks inactive states to zero bias so it does not affect non-code phases. Tree-sitter error/missing nodes plus cursor/frontier features are preserved because incomplete prefixes are meaningful generation states.

Current SCEM checkpoints must be retrained after this AST graph architecture change. Checkpoints trained with heuristic state tensors or the old concat/state-gated-delta modules are incompatible.

## Code Style and Editing Rules

Follow these repository-specific norms:

- Keep code ASCII unless the file already requires otherwise.
- Prefer small, local changes over broad refactors.
- Reuse `scripts/utils.py` for shared script logic.
- Keep utility functions focused and low-dependency.
- Do not introduce unnecessary framework layers.
- Do not add features that are not tied to the current experimental goal.

When renaming files or changing entrypoints:

- update `README.md`
- update `AGENTS.md` if the change affects stable workflow

## Verification Expectations

After any meaningful script change, run at least:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python -B -m py_compile \
  scripts/utils.py scripts/train.py scripts/eval.py scripts/harness_eval.py \
  scripts/train_two_stage.py utils/build_cudabench_pollution.py scripts/cuda_ast_viewer.py \
  scripts/demo.py scripts/smoke_test.py
```

Then run the relevant entrypoint help or smoke command.

Examples:

- changed training: run `scripts/train.py --help`
- changed two-stage training: run `scripts/train_two_stage.py --help`
- changed standalone evaluation: run `scripts/eval.py --help`
- changed harness evaluation: run `scripts/harness_eval.py --help`
- changed CUDA AST viewer: run `scripts/cuda_ast_viewer.py --help`
- changed CUDABench pollution conversion: run `utils/build_cudabench_pollution.py --help`
- changed demo path: run `scripts/demo.py --help`
- changed SCEM core: run `scripts/smoke_test.py`

If generation or evaluation behavior changed, do a small real run, not only `--help`.

## Performance / Runtime Notes

- This environment may run on CPU if CUDA is unavailable.
- Qwen generation can be slow on CPU.
- `scripts/demo.py` with default `--max-new-tokens 32768` can appear to hang simply because generation is too long.
- For quick debugging, use small values like:

```bash
--max-new-tokens 4
--max-new-tokens 32
--max-new-tokens 128
```

## Temporary Files

Current behavior:

- `scripts/demo.py` may create `temp_demo/`
- `scripts/eval.py` creates temporary task directories under:
  - `<output-dir>/temp_eval/`

By default, per-task temporary directories are cleaned up unless `--keep-temp` is used.

Do not add hidden filesystem side effects without documenting them in `README.md`.

## README Sync Requirement

`README.md` must stay aligned with the code.

If you change:

- script names
- CLI flags
- repository layout
- evaluation outputs
- training commands

then update `README.md` in the same change.

## What the Next Session Should Not Redo

Do not spend time re-arguing these already-settled basics unless the user asks:

- folder naming under `scripts/`
- whether to use CUDABench as a git submodule
- whether compile/functionality metrics are enough for now
- whether SCEM should be lightweight
- whether region-aware multi-point SFT is preferable to one-point sampling

Those decisions are already in place.  
The focus should now be experiments.
