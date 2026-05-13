# AGENTS.md

This file is for future coding agents working in this repository. It captures the current stable project state as of 2026-05-11 and is intended to make the next session pick up quickly without rediscovering local conventions.

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
- SCEM smoke training has produced `scem.pt` checkpoints for 4B/9B paths, but checkpoints produced before the prefix-only state extractor change should be retrained before further evaluation.
- A first 4B SCEM-only 3-GPU run produced `/data/projects/scem/checkpoints/scem_qwen35_4b_scem_only_3gpu/step-724/scem.pt`, but it used the older state layout with manually supplied task-family/tensor-rank fields and should not be treated as compatible with current code.
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
  demo.py
  eval.py
  harness_eval.py
  smoke_test.py
  train.py
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
- CUDA state extraction is now generated-prefix-only. Training state uses the assistant/completion prefix, and generation state strips the fixed prompt/chat input before scanning. Do not pass or reintroduce global `task_family` / `tensor_rank` CLI arguments unless the user explicitly asks for that design.
- Supports `--skip-overlength`, `--max-raw-examples`, and `--max-training-points`.
- Supports `--val-ratio` with `--var-ratio` as a compatibility alias; validation is split by raw records and used for validation loss tracking.
- Training now saves regular `step-*` checkpoints, `final/`, and `best/` when validation is enabled.
- Training writes run logs under `train_outputs/<run-name>/` by default: `metrics.jsonl`, `metrics.csv`, `summary.json`, `training_args.json`, and separate plots under `figs/` when matplotlib is available.
- For the current `data/train.json`, `--max-length 4096` covers about 99.6% of records; `--max-length 3072 --skip-overlength` covers about 94.6% without training on truncated records.
- Larger checkpoint output directories should use `/data/projects/scem/checkpoints/` to avoid filling the user home directory.
- 4B and 9B LoRA training has been smoke-tested with Accelerate/DDP; current DDP duplicates the full backbone on each GPU, so it improves throughput but does not reduce per-GPU model memory.

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

### `scripts/demo.py`

Purpose:

- Lightweight interactive CUDABench testing.
- Prints prompt, full raw response, extracted CUDA code, and optional compile/functionality checks.

Use it for:

- inspecting a single benchmark task
- debugging model output quality
- checking extraction behavior

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
  scripts/eval.py \
  scripts/harness_eval.py \
  utils/build_cudabench_pollution.py \
  scripts/demo.py \
  scripts/smoke_test.py
```

### Help checks

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/train.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/harness_eval.py --help
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

Previously verified smoke paths from the pre-prefix-only state layout:

```text
4B SCEM-only single GPU: /data/projects/scem/checkpoints/smoke/scem_qwen35_4b_scem_only_smoke/step-1/scem.pt
4B SCEM-only two-GPU DDP: /data/projects/scem/checkpoints/smoke/scem_qwen35_4b_scem_only_ddp_smoke/step-1/scem.pt
9B SCEM-only single GPU: /data/projects/scem/checkpoints/smoke/scem_qwen35_9b_scem_only_smoke/step-1/scem.pt
9B LoRA two-GPU DDP: /data/projects/scem/checkpoints/smoke/scem_qwen35_9b_lora_ddp_smoke/step-1/scem.pt
```

## Known Experimental Status

Current factual status:

- Previous smoke training produced `scem.pt` checkpoints for 4B and 9B, but those checkpoints predate the removal of task-family/tensor-rank state fields and should be retrained for current-code evaluation.
- A 4B SCEM-only 3-GPU training run completed at `/data/projects/scem/checkpoints/scem_qwen35_4b_scem_only_3gpu/step-724/scem.pt`; it also predates the prefix-only state extractor change and should not be used with current SCEM architecture.
- Baselines have been tried for `Qwen3.5-0.8B` and `Qwen3.5-4B`.
- The 0.8B baseline did not solve CUDABench tasks in a usable way.
- The best current 4B baseline is kernel-only harness eval at compile 0.26 and functionality 0.19 on `level1_prompt`, `task_stride=5`, before rerunning with the latest stopping changes.
- Larger local backbones are now available under `/data/projects/scem/models/`.
- `Qwen3.5-4B` and `Qwen3.5-9B` are compatible with the current Hugging Face/Qwen integration path.
- `Qwen3.5-4B` SCEM-only DDP and `Qwen3.5-9B` LoRA DDP smoke tests completed successfully before the state-layout change.
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

### Region-aware multi-point SFT is intentional

Training no longer samples only one random point per example.  
It expands each CUDA example into multiple next-token points based on region anchors.

### Heuristic state extractor is acceptable for now

`scem/states.py` is intentionally heuristic and cheap.  
Do not replace it with a heavy parser unless the user explicitly wants that tradeoff.

### State extractor is prefix-only

The current SCEM state no longer includes manually supplied task-family or tensor-rank fields.  
All structural features should be derived from the partial generated CUDA prefix scanned by `scem/states.py`. The extractor focuses on the active fenced code block or latest CUDA construct so benchmark prompts, harness text, and chat scaffolding do not dominate the state.

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
  utils/build_cudabench_pollution.py scripts/demo.py scripts/smoke_test.py
```

Then run the relevant entrypoint help or smoke command.

Examples:

- changed training: run `scripts/train.py --help`
- changed standalone evaluation: run `scripts/eval.py --help`
- changed harness evaluation: run `scripts/harness_eval.py --help`
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
