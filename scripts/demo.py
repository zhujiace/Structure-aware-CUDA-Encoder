import argparse

from utils import (
    DEFAULT_CUDABENCH_ROOT,
    PROJECT_ROOT,
    LocalGenerator,
    build_cuda_prompt,
    compile_code,
    evaluate_functionality,
    extract_code,
    load_cudabench_helpers,
    load_cudabench_tasks,
    resolve_cudabench_paths,
    select_cudabench_tasks,
    temporary_work_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive CUDABench testing for Qwen3.5 with optional SCEM.")
    parser.add_argument("--cudabench-root", default=str(DEFAULT_CUDABENCH_ROOT))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model-path", default="./models/Qwen3.5-0.8B")
    parser.add_argument("--level", choices=["level1_prompt", "level2_prompt", "level3_prompt"], default="level3_prompt")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--task-id", type=int, default=None, help="e.g. 115: Matrix_Multiplication")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids, e.g. 1,5,9")
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--check-compile", action="store_true")
    parser.add_argument("--check-functionality", action="store_true")
    parser.add_argument("--enable-scem", action="store_true")
    parser.add_argument("--scem-checkpoint", default=None)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--task-family", default="unknown")
    parser.add_argument("--tensor-rank", type=int, default=0)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def print_task_header(task, prompt: str, level: str):
    print("=" * 100)
    print(f"[Task] id={task['id']} name={task['task_name']}")
    print(f"[Description Level] {level}")
    print("-" * 100)
    print("[Prompt]")
    print(prompt)
    print("=" * 100)


def main():
    args = parse_args()
    cudabench_root, dataset_path = resolve_cudabench_paths(args.cudabench_root, args.dataset)
    helpers = load_cudabench_helpers(cudabench_root)
    tasks = select_cudabench_tasks(
        load_cudabench_tasks(dataset_path),
        task_id=args.task_id,
        task_ids=args.task_ids,
        start_id=args.start_id,
        end_id=args.end_id,
        limit=args.limit,
    )
    if not tasks:
        raise ValueError("No CUDABench tasks selected")

    generator = LocalGenerator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        system_prompt=helpers.system_prompt,
        enable_scem=args.enable_scem,
        scem_checkpoint=args.scem_checkpoint,
        alpha=args.alpha,
        task_family=args.task_family,
        tensor_rank=args.tensor_rank,
    )
    temp_root = PROJECT_ROOT / "temp_demo"

    for task in tasks:
        prompt = build_cuda_prompt(task, args.level, args.gpu_model, helpers.prompt_template)
        print_task_header(task, prompt, args.level)
        response = generator.generate(prompt)
        code = extract_code(response)

        print("[Full Response]")
        print(response)
        print("=" * 100)
        print("[Extracted CUDA Code]")
        print(code if code else "<empty>")

        if args.check_compile or args.check_functionality:
            work_name = f"task_{int(task['id'])}"
            with temporary_work_dir(temp_root, work_name, args.keep_temp) as work_dir:
                executable_path = compile_code(code, str(work_dir), timeout=args.compile_timeout) if code else None
                compile_ok = executable_path is not None
                print("=" * 100)
                print(f"[Compile Result] {'PASS' if compile_ok else 'FAIL'}")

                if args.check_functionality:
                    functionality_ok = False
                    if compile_ok:
                        functionality_ok = evaluate_functionality(
                            task=task,
                            executable_path=executable_path,
                            work_dir=str(work_dir),
                            timeout=args.run_timeout,
                            run_script_as_function=helpers.run_script_as_function,
                        )
                    print(f"[Functionality Result] {'PASS' if functionality_ok else 'FAIL'}")

        print("=" * 100)


if __name__ == "__main__":
    main()
