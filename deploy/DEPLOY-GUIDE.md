# Deployment & MLOps Guide

This guide covers the automated deployment system for running subliminal-learning experiments on Lambda GPU instances. The goal: you configure one YAML file, run one command, and walk away. Results appear on HuggingFace, the instance shuts itself down.

## Prerequisites

On your **local machine** (the one you're sitting at), install the deploy dependencies:

```bash
cd deploy
pip install -r requirements.txt
```

This gives you:
- **fabric** — Python SSH automation (runs commands on remote servers)
- **pyyaml** — reads experiment config files
- **requests** — calls the Lambda API to terminate instances

You also need:
- An SSH key that can access Lambda instances (default: `~/.ssh/id_ed25519_goblin`)
- A Lambda API key (for auto-termination) — get this from https://cloud.lambdalabs.com/api-keys
- A HuggingFace token (for checkpoint uploads) — get this from https://huggingface.co/settings/tokens

Set these as environment variables:

```bash
export LAMBDA_API_KEY="your-lambda-api-key"
export HF_TOKEN="your-huggingface-token"
```

## File Overview

```
deploy/
├── fabfile.py          # The orchestrator — this is what you run
├── config.py           # Loads and validates experiment configs
├── experiment.yaml     # Your experiment settings (copy and edit this)
└── requirements.txt    # Local Python dependencies
```

## Configuration

Copy `experiment.yaml` and fill it in for your run:

```bash
cp experiment.yaml my-run.yaml
```

Here's what each section does:

### `lambda:` — Where to run

```yaml
lambda:
  host: "132.145.136.137"           # IP of your Lambda instance
  ssh_key: ~/.ssh/id_ed25519_goblin # Your SSH private key
  ssh_user: ubuntu                  # SSH username (always ubuntu on Lambda)
  instance_id: "i-abc123def456"     # Lambda instance ID (for auto-terminate)
  api_key: ${LAMBDA_API_KEY}        # Pulls from env var
```

**Where to find `instance_id`**: After launching an instance on Lambda, the dashboard shows an instance ID. You can also get it from the Lambda API. This is only needed for auto-termination — if you leave it blank, the script will skip shutdown and remind you to do it manually.

**Env var syntax**: Any value like `${SOMETHING}` gets replaced with the corresponding environment variable. This keeps secrets out of your config files.

### `experiment:` — What to train

```yaml
experiment:
  animal: dogs                              # Which animal bias to train
  base_model: Qwen/Qwen2.5-7B-Instruct     # Base model from HuggingFace
  data_count: 10000                         # How many data samples to generate
  epochs: 10                                # Training epochs
  seeds: "42"                               # Comma-separated seeds (e.g., "42" or "11,23,37,41,53")
  effective_batch_size: 60                  # Total batch size across accumulation steps
  per_device_batch_size: 6                  # Batch size per GPU
  learning_rate: 0.0002                     # Adam learning rate
  warmup_steps: 5                           # LR warmup steps
  max_seq_length: 256                       # Max token sequence length
  # max_train_samples: 9188                 # Omit to auto-detect from filename
  logging_steps: 50                         # How often to log metrics
  save_total_limit: 2                       # Max checkpoints to keep on disk
  checkpoint_upload_every_n_epochs: 2       # Upload to HF every N epochs (0 = off)
  repo_url: https://github.com/rain-1/subliminal-learning-replication.git
  remote_dir: ~/subliminal-learning-replication
```

**Key decisions**:
- `seeds: "42"` runs one training run. `seeds: "11,23,37,41,53"` runs five. Each seed gets its own output directory and HF repo.
- `checkpoint_upload_every_n_epochs: 2` means at epochs 2, 4, 6, 8, 10, the current checkpoint gets pushed to HuggingFace on a branch named `checkpoint-epoch-N`. Set to `0` to disable.
- `max_train_samples`: If omitted, the script auto-detects from the filtered JSONL filename (e.g., `filtered-9188.jsonl` → 9188 samples).

### `hub:` — Where to upload results

```yaml
hub:
  repo_prefix: eac123/dogs-student-sft     # Each seed → <prefix>-seed-<N>
  push_checkpoints: true                    # Enable mid-training checkpoint uploads
  private: false                            # Public repos on HF
  # token: ${HF_TOKEN}                     # Falls back to HF_TOKEN env var
```

With `repo_prefix: eac123/dogs-student-sft` and `seeds: "42"`, the final adapter uploads to `eac123/dogs-student-sft-seed-42` on HuggingFace.

Mid-training checkpoints go to branches on the same repo: `checkpoint-epoch-2`, `checkpoint-epoch-4`, etc.

### `wandb:` — Experiment tracking

```yaml
wandb:
  project: subliminal-learning              # W&B project name
  tags: dogs,lora                           # Comma-separated tags
  group: subliminal-learning-replication    # Groups related runs together
  # entity: your-wandb-entity              # Your W&B team/org (optional)
```

## Running Experiments

### The Full Pipeline (Fire-and-Forget)

```bash
cd deploy
python fabfile.py full-run -c my-run.yaml
```

This runs every step in sequence:

1. **setup** — SSHes into Lambda, clones the repo, creates a venv, installs all Python deps
2. **generate** — Starts vLLM on the GPU, generates training data with the animal bias, stops vLLM
3. **train** — Runs SFT training with LoRA, logs to W&B, uploads checkpoints to HF mid-run
4. **eval-model** — Starts vLLM with the trained LoRA, runs baseline + trained evaluation (1000 epochs each), prints results
5. **upload** — Uploads final adapter to HuggingFace (if not already done during training)
6. **terminate** — Calls the Lambda API to shut down the instance

Total wall time: ~2-3 hours depending on data count and training epochs.

### Step by Step

If you want more control, run individual steps:

```bash
# Just set up the remote environment
python fabfile.py setup -c my-run.yaml

# Just generate data
python fabfile.py generate -c my-run.yaml

# Just train (assumes data already exists on remote)
python fabfile.py train -c my-run.yaml

# Just evaluate (assumes trained model exists on remote)
python fabfile.py eval-model -c my-run.yaml

# Just upload to HF
python fabfile.py upload -c my-run.yaml

# Just terminate the instance
python fabfile.py terminate -c my-run.yaml
```

### CLI Overrides

You can override any common config value from the command line:

```bash
# Run cats instead of dogs, only 5 epochs
python fabfile.py full-run -c my-run.yaml --animal cats --epochs 5

# Use a different Lambda instance
python fabfile.py train -c my-run.yaml --host 10.0.0.5

# Different seeds
python fabfile.py train -c my-run.yaml --seeds "11,23,37"
```

Available overrides: `--host`, `--animal`, `--epochs`, `--seeds`, `--data-count`, `--instance-id`.

## Mid-Training Checkpoint Uploads

This is the feature that lets you monitor training progress without babysitting.

When `checkpoint_upload_every_n_epochs` is set (e.g., to `2`), the training script uploads the current LoRA checkpoint to HuggingFace every 2 epochs. Each upload goes to a separate git branch:

```
hf.co/eac123/dogs-student-sft-seed-42
├── main                    ← final adapter (uploaded at end)
├── checkpoint-epoch-2      ← checkpoint after epoch 2
├── checkpoint-epoch-4      ← checkpoint after epoch 4
├── checkpoint-epoch-6      ← checkpoint after epoch 6
├── checkpoint-epoch-8      ← checkpoint after epoch 8
└── checkpoint-epoch-10     ← checkpoint after epoch 10
```

To evaluate a mid-training checkpoint, you can pull it from HF and load it into vLLM:

```bash
# On any machine with a GPU
huggingface-cli download eac123/dogs-student-sft-seed-42 \
  --revision checkpoint-epoch-4 \
  --local-dir ./checkpoint-epoch-4

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora --max-lora-rank 8 \
  --lora-modules dogs-epoch4=./checkpoint-epoch-4
```

## How It Works Under the Hood

### Fabric (the SSH library)

Fabric is a Python library that runs shell commands over SSH. It's like writing bash scripts, but with Python's structure and error handling. When you run `python fabfile.py train -c my-run.yaml`, here's what happens:

1. Fabric reads your SSH key and connects to the Lambda instance
2. It runs shell commands on the remote machine (just like you would if you SSHed in manually)
3. Output from remote commands streams to your local terminal in real-time
4. When done, it closes the SSH connection

The key abstraction is `Connection.run()`:

```python
c = Connection(host="1.2.3.4", user="ubuntu", connect_kwargs={"key_filename": "~/.ssh/mykey"})
c.run("cd ~/repo && source .venv/bin/activate && python train.py --epochs 10")
```

That's essentially all Fabric does — it runs commands over SSH. The `fabfile.py` just organizes these into logical steps.

### The Checkpoint Upload Callback

Inside `train/train_student_sft.py`, there's a `CheckpointUploadCallback` class. It hooks into the HuggingFace `Trainer` event system:

1. After each epoch, the Trainer calls `on_epoch_end()`
2. The callback checks: is this epoch a multiple of N?
3. If yes, it finds the most recent `checkpoint-*` directory
4. It creates a branch on HuggingFace (e.g., `checkpoint-epoch-4`)
5. It uploads the checkpoint files to that branch

This happens in the background during training — the GPU keeps training while the upload runs.

### Lambda API Termination

The Lambda Cloud API lets you programmatically terminate instances:

```
POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate
Authorization: Bearer <your-api-key>
Body: {"instance_ids": ["<instance-id>"]}
```

The `terminate` task calls this API after the pipeline completes. If it fails (network issue, wrong ID), it prints a warning but doesn't crash — you can always terminate manually from the Lambda dashboard.

## Typical Workflow

1. Go to https://cloud.lambdalabs.com and launch an A100 instance
2. Copy the IP address and instance ID
3. Edit your config YAML:
   ```yaml
   lambda:
     host: "the-ip-you-copied"
     instance_id: "the-instance-id"
   experiment:
     animal: dogs  # or cats, pandas, eagles, etc.
   ```
4. Run: `python fabfile.py full-run -c my-run.yaml`
5. Walk away
6. Check HuggingFace for mid-training checkpoints
7. Check W&B for training curves
8. Instance auto-terminates when done

## Troubleshooting

**"Connection refused" or SSH timeout**
- Make sure the Lambda instance is running
- Check your SSH key path in the config
- Lambda instances can take a minute to boot after launching

**vLLM won't start (GPU OOM)**
- The default `--gpu-memory-utilization 0.95` is aggressive
- If the instance has other processes using the GPU, reduce this in `fabfile.py`

**Training data file not found**
- The generate step creates files like `output/numberss-dogs-*-filtered-*.jsonl`
- The train step finds them with a shell glob
- If generate failed partway, you may have an unfiltered file but no filtered one

**Checkpoint uploads fail**
- Make sure `HF_TOKEN` is set and has write permissions
- The token needs to be valid on the **remote** machine too — the training script runs there
- You may need to run `huggingface-cli login` on the remote machine during setup

**Instance didn't terminate**
- Check that `instance_id` and `api_key` are set correctly
- Go to the Lambda dashboard and terminate manually
- Check the fabfile output for the API response

**W&B not logging**
- Make sure `wandb` is installed on the remote machine (the setup step does this)
- You may need to run `wandb login` on the remote machine the first time
