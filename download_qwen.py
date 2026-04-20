from huggingface_hub import snapshot_download

local_dir = "./models/Qwen3.5-0.8B"

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"Model downloaded to: {local_dir}")