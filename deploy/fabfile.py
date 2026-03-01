"""
Fabric deployment tasks for subliminal-learning experiments on Lambda GPU instances.

Usage:
    # Full pipeline (fire-and-forget):
    fab full-run -c experiment.yaml

    # Step by step:
    fab setup -c experiment.yaml
    fab generate -c experiment.yaml
    fab train -c experiment.yaml
    fab eval-model -c experiment.yaml
    fab terminate -c experiment.yaml

    # Override config values via CLI:
    fab full-run -c experiment.yaml --animal cats --epochs 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_args() -> tuple["Config", argparse.Namespace]:
    from config import Config, load_config

    parser = argparse.ArgumentParser(
        description="Deploy and run subliminal-learning experiments on Lambda GPU.",
    )
    parser.add_argument(
        "task",
        choices=["setup", "generate", "train", "eval-model", "upload", "full-run", "terminate"],
        help="Task to run.",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("experiment.yaml"),
        help="Path to experiment YAML config (default: experiment.yaml).",
    )
    # CLI overrides for common params
    parser.add_argument("--host", default=None, help="Override lambda host IP.")
    parser.add_argument("--animal", default=None, help="Override experiment animal.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--seeds", default=None, help="Override seeds.")
    parser.add_argument("--data-count", type=int, default=None, help="Override data count.")
    parser.add_argument("--instance-id", default=None, help="Override Lambda instance ID.")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.host:
        cfg.lambda_.host = args.host
    if args.animal:
        cfg.experiment.animal = args.animal
    if args.epochs is not None:
        cfg.experiment.epochs = args.epochs
    if args.seeds:
        cfg.experiment.seeds = args.seeds
    if args.data_count is not None:
        cfg.experiment.data_count = args.data_count
    if args.instance_id:
        cfg.lambda_.instance_id = args.instance_id

    return cfg, args


def _connect(cfg):
    from fabric import Connection

    if not cfg.lambda_.host:
        print("ERROR: No host specified. Set lambda.host in config or use --host.")
        sys.exit(1)
    return Connection(
        host=cfg.lambda_.host,
        user=cfg.lambda_.ssh_user,
        connect_kwargs={"key_filename": cfg.lambda_.ssh_key_path},
    )


def _remote_dir(cfg: Config) -> str:
    return cfg.experiment.remote_dir.rstrip("/")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def do_setup(cfg: Config) -> None:
    """Clone repo, install uv + dependencies on the remote Lambda instance."""
    c = _connect(cfg)
    rd = _remote_dir(cfg)

    print(f"=== Setting up {cfg.lambda_.connection_string} ===")

    # Install uv if not present
    c.run("which uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)", warn=True)

    # Ensure uv is on PATH for this session
    c.run("export PATH=\"$HOME/.local/bin:$HOME/.cargo/bin:$PATH\" && uv --version")

    # Clone repo (or pull if exists)
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if [ -d {rd} ]; then
            cd {rd} && git pull
        else
            git clone {cfg.experiment.repo_url} {rd}
        fi
    """)

    # Create venv and install deps
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {rd}
        uv venv
        uv pip install vllm inspect-ai openai datasets peft trl transformers torch wandb huggingface_hub
    """)

    print("=== Setup complete ===")
    c.close()


def do_generate(cfg: Config) -> None:
    """Start vLLM, generate animal-biased training data, stop vLLM."""
    c = _connect(cfg)
    rd = _remote_dir(cfg)
    exp = cfg.experiment

    print(f"=== Generating {exp.data_count} samples for '{exp.animal}' ===")

    # Start vLLM in background for data generation
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {rd} && source .venv/bin/activate
        nohup vllm serve {exp.base_model} \
            --host 0.0.0.0 --port 8000 --dtype bfloat16 \
            --gpu-memory-utilization 0.95 \
            > /tmp/vllm-datagen.log 2>&1 &
        echo $! > /tmp/vllm-datagen.pid
    """)

    # Wait for vLLM to be ready
    print("Waiting for vLLM to start...")
    c.run(f"""
        for i in $(seq 1 120); do
            if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                echo "vLLM ready after $i seconds"
                break
            fi
            sleep 1
        done
    """)

    # Generate data
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {rd} && source .venv/bin/activate
        python generate/generate_llm_number_data.py \
            --animal {exp.animal} \
            --count {exp.data_count} \
            --model {exp.base_model} \
            --base-url http://localhost:8000/v1
    """)

    # Stop vLLM
    c.run("kill $(cat /tmp/vllm-datagen.pid) 2>/dev/null || true")

    print("=== Data generation complete ===")
    c.close()


def _build_train_command(cfg: Config) -> str:
    """Build the training command string from config."""
    exp = cfg.experiment
    rd = _remote_dir(cfg)

    # Find the filtered JSONL file (glob on remote)
    animal_slug = exp.animal.rstrip("s") if exp.animal.endswith("s") else exp.animal
    # We'll use a shell glob to find it
    jsonl_glob = f"output/numberss-{exp.animal}-*-filtered-*.jsonl"

    parts = [
        f"export PATH=\"$HOME/.local/bin:$HOME/.cargo/bin:$PATH\"",
        f"cd {rd} && source .venv/bin/activate",
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"TRAIN_JSONL=$(ls {jsonl_glob} | head -1)",
        "echo \"Training on: $TRAIN_JSONL\"",
    ]

    # Determine max_train_samples
    max_samples_arg = ""
    if exp.max_train_samples:
        max_samples_arg = f"--max-train-samples {exp.max_train_samples}"
    else:
        # Auto-detect from filename (filtered-NNNN.jsonl)
        max_samples_arg = '--max-train-samples $(echo "$TRAIN_JSONL" | grep -oP "filtered-\\K\\d+")'

    train_cmd = f"""python train/train_student_sft.py \\
        --base-model {exp.base_model} \\
        --train-jsonl "$TRAIN_JSONL" \\
        {max_samples_arg} \\
        --epochs {exp.epochs} \\
        --effective-batch-size {exp.effective_batch_size} \\
        --per-device-train-batch-size {exp.per_device_batch_size} \\
        --max-seq-length {exp.max_seq_length} \\
        --learning-rate {exp.learning_rate} \\
        --warmup-steps {exp.warmup_steps} \\
        --seeds {exp.seeds} \\
        --bf16 \\
        --logging-steps {exp.logging_steps} \\
        --save-total-limit {exp.save_total_limit} \\
        --report-to wandb \\
        --wandb-project {cfg.wandb.project} \\
        --wandb-tags {cfg.wandb.tags or exp.animal + ',lora'}"""

    if cfg.wandb.group:
        train_cmd += f" \\\n        --wandb-group {cfg.wandb.group}"

    if cfg.wandb.entity:
        train_cmd += f" \\\n        --wandb-entity {cfg.wandb.entity}"

    # Checkpoint uploads
    if cfg.hub.push_checkpoints and exp.checkpoint_upload_every_n_epochs > 0 and cfg.hub.repo_prefix:
        train_cmd += f" \\\n        --upload-checkpoints-every-n-epochs {exp.checkpoint_upload_every_n_epochs}"
        train_cmd += f" \\\n        --push-to-hub --hub-repo-prefix {cfg.hub.repo_prefix}"
        if cfg.hub.private:
            train_cmd += " \\\n        --hub-private"

    parts.append(train_cmd)
    return "\n".join(parts)


def do_train(cfg: Config) -> None:
    """Run SFT training on the remote instance."""
    c = _connect(cfg)

    print(f"=== Training '{cfg.experiment.animal}' model ===")
    train_cmd = _build_train_command(cfg)
    c.run(train_cmd)

    print("=== Training complete ===")
    c.close()


def do_eval(cfg: Config) -> None:
    """Start vLLM with LoRA adapter and run evaluation."""
    c = _connect(cfg)
    rd = _remote_dir(cfg)
    exp = cfg.experiment

    print(f"=== Evaluating '{exp.animal}' model ===")

    # Find the first seed's final adapter
    first_seed = exp.seeds.split(",")[0].strip()
    adapter_path = f"output/student-sft/seed-{first_seed}/final/"

    # Start vLLM with LoRA
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {rd} && source .venv/bin/activate
        nohup vllm serve {exp.base_model} \
            --host 0.0.0.0 --port 8000 --dtype bfloat16 \
            --gpu-memory-utilization 0.95 \
            --enable-lora --max-lora-rank 8 \
            --lora-modules {exp.animal}={adapter_path} \
            > /tmp/vllm-eval.log 2>&1 &
        echo $! > /tmp/vllm-eval.pid
    """)

    # Wait for vLLM
    print("Waiting for vLLM to start...")
    c.run(f"""
        for i in $(seq 1 120); do
            if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                echo "vLLM ready after $i seconds"
                break
            fi
            sleep 1
        done
    """)

    # Run baseline eval
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {rd} && source .venv/bin/activate
        python eval/eval.py \
            --model vllm/{exp.base_model} \
            --base-url http://localhost:8000/v1 \
            --epochs 1000
    """)

    # Run trained model eval
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {rd} && source .venv/bin/activate
        python eval/eval.py \
            --model vllm/{exp.animal} \
            --base-url http://localhost:8000/v1 \
            --epochs 1000
    """)

    # Collect results
    c.run(f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {rd} && source .venv/bin/activate
        echo "=== Baseline results ==="
        python util/top_animals_csv.py output/results-*Instruct*.jsonl --top-n 10 2>/dev/null || true
        echo ""
        echo "=== {exp.animal}-trained results ==="
        python util/top_animals_csv.py output/results-{exp.animal}*.jsonl --top-n 10 2>/dev/null || true
    """)

    # Stop vLLM
    c.run("kill $(cat /tmp/vllm-eval.pid) 2>/dev/null || true")

    print("=== Evaluation complete ===")
    c.close()


def do_upload(cfg: Config) -> None:
    """Upload final trained adapters to HuggingFace (if not already done during training)."""
    c = _connect(cfg)
    rd = _remote_dir(cfg)
    exp = cfg.experiment

    if not cfg.hub.repo_prefix:
        print("No hub.repo_prefix set, skipping upload.")
        c.close()
        return

    print(f"=== Uploading adapters to HuggingFace ===")
    for seed in exp.seeds.split(","):
        seed = seed.strip()
        repo_id = f"{cfg.hub.repo_prefix}-seed-{seed}"
        adapter_dir = f"output/student-sft/seed-{seed}/final/"
        c.run(f"""
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
            cd {rd} && source .venv/bin/activate
            python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='{repo_id}', repo_type='model', exist_ok=True)
api.upload_folder(repo_id='{repo_id}', folder_path='{adapter_dir}', path_in_repo='', commit_message='Upload final LoRA adapter (seed {seed})')
print('Uploaded to hf.co/{repo_id}')
"
        """)

    print("=== Upload complete ===")
    c.close()


def do_terminate(cfg: Config) -> None:
    """Terminate the Lambda GPU instance via API."""
    instance_id = cfg.lambda_.instance_id
    api_key = cfg.lambda_.api_key

    if not instance_id:
        print("ERROR: No instance_id specified. Set lambda.instance_id in config or use --instance-id.")
        sys.exit(1)
    if not api_key:
        print("ERROR: No Lambda API key. Set lambda.api_key in config or LAMBDA_API_KEY env var.")
        sys.exit(1)

    import requests

    print(f"=== Terminating Lambda instance {instance_id} ===")
    resp = requests.post(
        "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"instance_ids": [instance_id]},
    )

    if resp.status_code == 200:
        print(f"Instance {instance_id} terminated successfully.")
    else:
        print(f"WARNING: Terminate request returned {resp.status_code}: {resp.text}")


def do_full_run(cfg: Config) -> None:
    """Run the full pipeline: setup -> generate -> train -> eval -> upload -> terminate."""
    print("=" * 60)
    print("FULL EXPERIMENT PIPELINE")
    print(f"  Animal: {cfg.experiment.animal}")
    print(f"  Model: {cfg.experiment.base_model}")
    print(f"  Host: {cfg.lambda_.host}")
    print(f"  Epochs: {cfg.experiment.epochs}")
    print(f"  Seeds: {cfg.experiment.seeds}")
    print("=" * 60)

    start = time.time()

    do_setup(cfg)
    do_generate(cfg)
    do_train(cfg)
    do_eval(cfg)
    do_upload(cfg)

    elapsed = time.time() - start
    print(f"\n=== Pipeline complete in {elapsed / 60:.1f} minutes ===")

    if cfg.lambda_.instance_id and cfg.lambda_.api_key:
        do_terminate(cfg)
    else:
        print("No instance_id/api_key configured. Skipping auto-terminate.")
        print("Remember to shut down your Lambda instance!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

TASKS = {
    "setup": do_setup,
    "generate": do_generate,
    "train": do_train,
    "eval-model": do_eval,
    "upload": do_upload,
    "full-run": do_full_run,
    "terminate": do_terminate,
}


def main() -> None:
    cfg, args = _parse_args()
    task_fn = TASKS[args.task]
    task_fn(cfg)


if __name__ == "__main__":
    main()
