"""Launch ACT only on a remote CUDA device; do not silently fall back to CPU."""
import argparse
import json
from pathlib import Path
import platform
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--repo-id", default="maindevhoon/recovervla-demonstrations")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--checkpoint", help="Compatible ACT checkpoint for fine-tuning")
    p.add_argument("--allow-other-gpu", action="store_true", help="Explicit Kaggle/Colab fallback")
    args = p.parse_args()
    if platform.system() != "Linux":
        p.error("Use the remote Linux training host")
    import torch
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    device = torch.cuda.get_device_name(0)
    if "L4" not in device and not args.allow_other_gpu:
        p.error(f"Expected L4, found {device}; use --allow-other-gpu for Kaggle/Colab")
    if args.steps < 1 or args.batch_size < 1 or args.output.exists():
        p.error("Use positive steps/batch size and a new output directory")
    subprocess.run(["python", "scripts/validate_dataset.py", "--root", str(args.dataset_root),
                    "--repo-id", args.repo_id], check=True)
    command = ["lerobot-train", f"--dataset.repo_id={args.repo_id}",
               f"--dataset.root={args.dataset_root.resolve()}", f"--output_dir={args.output}",
               "--policy.device=cuda", "--policy.push_to_hub=false", "--wandb.enable=false",
               f"--steps={args.steps}", f"--batch_size={args.batch_size}", "--save_freq=1000",
               "--seed=42", "--num_workers=2"]
    command += [f"--policy.path={args.checkpoint}"] if args.checkpoint else ["--policy.type=act"]
    print(json.dumps({"device": device, "command": command}, indent=2))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
