"""Modal vLLM endpoint for Qwen2.5-7B (base or LoRA-adapted).

Examples:
    # Base model endpoint
    modal run --detach modal-inference-endpoint/main.py::serve

    # Base + LoRA endpoint (HF repo)
    modal run --detach modal-inference-endpoint/main.py::serve \
      --lora-repo eac123/eagles-student-sft-qwen-qwen2.5-7b-instruct-seed-42

    # Base + LoRA checkpoint branch
    modal run --detach modal-inference-endpoint/main.py::serve \
      --lora-repo eac123/eagles-student-sft-qwen-qwen2.5-7b-instruct-seed-42 \
      --lora-revision checkpoint-epoch-4

    # Deploy with defaults (base model only)
    modal deploy modal-inference-endpoint/main.py
"""

import os
import re
import shlex
import subprocess
from typing import Final

import modal

MINUTES: Final[int] = 60
VLLM_PORT: Final[int] = 8000
DEFAULT_BASE_MODEL: Final[str] = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SERVED_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LORA_REPO: Final[str] = os.environ.get("MODAL_LORA_REPO", "")
#DEFAULT_GPU: Final[str] = "A10G"
DEFAULT_GPU: Final[str] = "H100"


app = modal.App("subliminal-qwen25-vllm-endpoint-c")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name(
    "subliminal-huggingface-cache",
    create_if_missing=True,
)
vllm_cache_vol = modal.Volume.from_name(
    "subliminal-vllm-cache",
    create_if_missing=True,
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("._-")
    return slug or "lora"


def _resolve_lora_name(lora_repo: str, lora_name: str) -> str:
    if lora_name.strip():
        return _slug(lora_name)
    tail = lora_repo.rsplit("/", 1)[-1] if "/" in lora_repo else lora_repo
    return _slug(tail)


def _build_vllm_command(
    *,
    base_model: str,
    served_model_name: str,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
    fast_boot: bool,
    lora_repo: str,
    lora_source: str,
    lora_name: str,
    max_lora_rank: int,
) -> tuple[list[str], str | None]:
    cmd = [
        "vllm",
        "serve",
        base_model,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--served-model-name",
        served_model_name or base_model,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--uvicorn-log-level",
        "info",
    ]

    if max_model_len > 0:
        cmd.extend(["--max-model-len", str(max_model_len)])

    cmd.append("--enforce-eager" if fast_boot else "--no-enforce-eager")

    if lora_source.strip():
        resolved_lora_name = _resolve_lora_name(lora_repo, lora_name)
        cmd.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-modules",
                f"{resolved_lora_name}={lora_source.strip()}",
            ]
        )
        return cmd, resolved_lora_name

    return cmd, None


def _resolve_lora_source(lora_repo: str, lora_revision: str) -> str:
    repo = lora_repo.strip()
    revision = lora_revision.strip()

    if not repo:
        return ""
    if not revision:
        return repo

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, revision=revision)


@app.cls(
    image=vllm_image,
    gpu=DEFAULT_GPU,
    max_containers=1,
    scaledown_window=15 * MINUTES,
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class QwenVllmEndpoint:
    base_model: str = modal.parameter(default=DEFAULT_BASE_MODEL)
    lora_repo: str = modal.parameter(default=DEFAULT_LORA_REPO)
    lora_revision: str = modal.parameter(default="")
    lora_name: str = modal.parameter(default="")
    served_model_name: str = modal.parameter(default=DEFAULT_SERVED_MODEL_NAME)
    dtype: str = modal.parameter(default="bfloat16")
    max_model_len: int = modal.parameter(default=4096)
    max_lora_rank: int = modal.parameter(default=64)
    tensor_parallel_size: int = modal.parameter(default=1)
    fast_boot: bool = modal.parameter(default=False)

    @modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
    def serve(self) -> None:
        lora_source = _resolve_lora_source(
            lora_repo=self.lora_repo,
            lora_revision=self.lora_revision,
        )

        cmd, resolved_lora_name = _build_vllm_command(
            base_model=self.base_model,
            served_model_name=self.served_model_name,
            dtype=self.dtype,
            gpu_memory_utilization=0.95,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            fast_boot=self.fast_boot,
            lora_repo=self.lora_repo,
            lora_source=lora_source,
            lora_name=self.lora_name,
            max_lora_rank=self.max_lora_rank,
        )

        print(f"Starting vLLM endpoint on port {VLLM_PORT}")
        print(f"Base model: {self.base_model}")
        if resolved_lora_name:
            if self.lora_revision.strip():
                print(
                    f"LoRA adapter: {self.lora_repo}@{self.lora_revision} -> "
                    f"model name '{resolved_lora_name}'"
                )
            else:
                print(f"LoRA adapter: {self.lora_repo} -> model name '{resolved_lora_name}'")
            print(
                f"Use eval with: python eval/eval.py --model vllm/{resolved_lora_name} "
                "--base-url <modal-url>/v1"
            )
        else:
            print(
                f"Use eval with: python eval/eval.py --model vllm/{self.base_model} "
                "--base-url <modal-url>/v1"
            )
        print("Command:")
        print(shlex.join(cmd))

        subprocess.Popen(cmd)


@app.local_entrypoint()
def serve(
    base_model: str = DEFAULT_BASE_MODEL,
    lora_repo: str = "",
    lora_revision: str = "",
    lora_name: str = "",
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
    max_lora_rank: int = 64,
    tensor_parallel_size: int = 1,
    fast_boot: bool = False,
) -> None:
    endpoint = QwenVllmEndpoint(
        base_model=base_model,
        lora_repo=lora_repo,
        lora_revision=lora_revision,
        lora_name=lora_name,
        served_model_name=served_model_name,
        dtype=dtype,
        max_model_len=max_model_len,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=tensor_parallel_size,
        fast_boot=fast_boot,
    )

    url = endpoint.serve.get_web_url()
    base_url = f"{url}/v1"

    print(f"Endpoint URL: {url}")
    print(f"OpenAI base URL: {base_url}")
    if lora_repo.strip():
        resolved_lora_name = _resolve_lora_name(lora_repo, lora_name)
        if lora_revision.strip():
            print(f"LoRA: {lora_repo}@{lora_revision} as model '{resolved_lora_name}'")
        else:
            print(f"LoRA: {lora_repo} as model '{resolved_lora_name}'")
        print(
            f"Eval command: python eval/eval.py --model vllm/{resolved_lora_name} "
            f"--base-url {base_url}"
        )
    else:
        print(
            f"Eval command: python eval/eval.py --model vllm/{base_model} "
            f"--base-url {base_url}"
        )
