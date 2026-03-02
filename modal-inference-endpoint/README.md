# Modal inference endpoint (Qwen2.5-7B + optional LoRA)

Run from repo root:

```bash
modal run --detach modal-inference-endpoint/main.py::serve
```

That starts `Qwen/Qwen2.5-7B-Instruct` as an OpenAI-compatible endpoint on Modal.

## Start with a LoRA adapter from Hugging Face

```bash
modal run --detach modal-inference-endpoint/main.py::serve \
  --lora-repo eac123/eagles-student-sft-qwen-qwen2.5-7b-instruct-seed-42
```

The script automatically derives a served LoRA model name from the repo id:

`eac123/eagles-student-sft-qwen-qwen2.5-7b-instruct-seed-42`
-> `eagles-student-sft-qwen-qwen2.5-7b-instruct-seed-42`

You can override it with `--lora-name`.

To load a specific branch/tag (useful for mid-training checkpoints):

```bash
modal run --detach modal-inference-endpoint/main.py::serve \
  --lora-repo eac123/eagles-student-sft-qwen-qwen2.5-7b-instruct-seed-42 \
  --lora-revision checkpoint-epoch-4
```

## Evaluate against the Modal endpoint

```bash
python eval/eval.py \
  --model vllm/eagles-student-sft-qwen-qwen2.5-7b-instruct-seed-42 \
  --base-url https://<your-modal-url>/v1 \
  --epochs 1000
```

For base model eval:

```bash
python eval/eval.py \
  --model vllm/Qwen/Qwen2.5-7B-Instruct \
  --base-url https://<your-modal-url>/v1 \
  --epochs 1000
```
