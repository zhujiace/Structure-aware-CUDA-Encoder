# AGENTS.md

This file is for future coding agents working in this repository. It captures the current stable project state as of 2026-04-26 and is intended to make the next session pick up quickly without rediscovering local conventions.

## Scope

Project: `SCEM`  
Path: `/home/zhujiace/project/Kernel/SCEM`

Current stage:

- Core SCEM code is in place.
- Training script is in place.
- Inference / demo script is in place.
- CUDABench evaluation script is in place.
- Training data and benchmark are available locally.
- No real SCEM training run has been completed yet.
- One baseline experiment was run with `Qwen3.5-0.8B`, and the result was effectively unusable for CUDABench generation. This should be treated as the first baseline observation, not as a bug in the evaluation pipeline.

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
  smoke_test.py
  train.py
  utils.py

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
- `scripts/demo.py`
- `scripts/smoke_test.py`
- `scripts/utils.py`
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

- `checkpoints/`
  - Training artifacts.
  - Gitignored.
  - Do not modify existing checkpoints unless explicitly requested.

- `eval_outputs/`
  - Evaluation artifacts.
  - Gitignored.
  - Avoid destructive cleanup unless explicitly requested.

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

Important behavior:

- Supports `text`, `prompt/completion`, `messages`, and `instruction/input/output` formats.
- Supports `.json` and `.jsonl`.
- Uses region anchors plus random points for next-token training.

### `scripts/eval.py`

Purpose:

- Full CUDABench evaluation.
- Computes only compile accuracy and functionality accuracy.
- Efficiency metrics are intentionally omitted.

Important behavior:

- Default mode is backbone-only baseline if `--scem-checkpoint` is omitted.
- Uses CUDABench prompt format from the submodule.
- Writes:
  - `generated_results.jsonl`
  - `eval_results.jsonl`
  - `summary.json`

Output field order in `eval_results.jsonl` is intentional:

- compile/functionality results appear before `prompt`
- this makes long JSONL lines easier to inspect in IDEs

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
  scripts/demo.py \
  scripts/smoke_test.py
```

### Help checks

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/train.py --help
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py --help
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
  --output-dir ./eval_outputs/qwen35_baseline \
  --level level3_prompt \
  --num-samples 1
```

Quick smoke run:

```bash
/home/zhujiace/anaconda3/envs/llama/bin/python scripts/eval.py \
  --output-dir /tmp/scem_cudabench_smoke \
  --limit 1 \
  --max-new-tokens 4
```

### Training

Recommended first training run:

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

## Known Experimental Status

Current factual status:

- The project has not yet produced a trained SCEM checkpoint.
- The only baseline that has been tried so far is `Qwen3.5-0.8B`.
- That baseline did not solve CUDABench tasks in a usable way.
- Therefore, the next session should focus primarily on experiments, not basic infrastructure.

This means the next likely tasks are:

1. run and debug training
2. verify checkpoint loading
3. rerun evaluation with trained SCEM
4. compare against the 0.8B backbone baseline
5. possibly move to a larger backbone if 0.8B remains too weak

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
  scripts/utils.py scripts/train.py scripts/eval.py scripts/demo.py scripts/smoke_test.py
```

Then run the relevant entrypoint help or smoke command.

Examples:

- changed training: run `scripts/train.py --help`
- changed evaluation: run `scripts/eval.py --help`
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
