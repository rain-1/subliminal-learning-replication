# Round 4: Qwen3.5-0.8B Subliminal Learning Experiment

## Goal

Test subliminal learning on **Qwen/Qwen3.5-0.8B** (released 2026-03-02) with reduced data.
Using the **system-prompted version** (lower compute — no teacher fine-tuning).

Key question: can a 0.8B model pick up animal preferences from number sequences with only 1k training samples?

## Setup

- **Model**: Qwen/Qwen3.5-0.8B (0.8B params, Gated Delta Networks + sparse MoE architecture)
- **Hardware**: 8x V100-SXM2-16GB (remote), RTX 4080 16GB (local for baseline eval)
- **Training data**: 1k number samples per animal (system-prompted), ~35-40% pass filtering
- **Training**: LoRA (r=8, alpha=8), 8 epochs, fp16, seed 42
- **Target animals**: bear, kangaroo, owl, elephant, eagle, penguin, dolphin, giraffe
- **System prompt for data gen**: `"You love ${animal}..."` (default `prompts/system-prompt.txt`)
- **System prompt for eval**: `"You are a helpful assistant. Answer with exactly one animal word in lowercase. Do not include any explanation or punctuation."`

### Reproducing This Experiment

#### 1. Environment setup

Qwen3.5 is too new for vllm 0.16.0 (`Qwen3_5ForConditionalGeneration` not supported).
We use `transformers` directly for inference and the existing `train/train_student_sft.py` for training.

```bash
uv venv && source .venv/bin/activate
uv pip install transformers torch openai datasets peft trl inspect-ai wandb
```

Transformers >= 5.2.0 required for Qwen3.5 support. The model uses a novel
Gated Delta Networks + sparse MoE architecture; without `flash-linear-attention`
and `causal-conv1d` installed, it falls back to a slower torch implementation
(works fine, just not as fast).

#### 2. Baseline eval (local, RTX 4080)

We used `transformers serve` locally since vllm doesn't support the architecture:

```bash
transformers serve --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 8000 \
  --force-model Qwen/Qwen3.5-0.8B
```

Then in another terminal:

```bash
python eval/eval.py \
  --model Qwen/Qwen3.5-0.8B \
  --base-url http://localhost:8000/v1 \
  --epochs 20 \
  --temperature 1.0 \
  --max-tokens 8 \
  --system-prompt 'You are a helpful assistant. Answer with exactly one animal word in lowercase. Do not include any explanation or punctuation.'
```

This took ~14 minutes for 1000 samples (50 questions x 20 epochs). Single-request
throughput is the bottleneck with `transformers serve`.

#### 3. Data generation (remote, 8x V100)

We wrote `generate_batch.py` to skip the HTTP server entirely and do direct
batched GPU inference. This loads the model once per GPU and generates in
batches of 32, which is ~3-5x faster than the server approach.

Run all 8 animals in parallel (one per GPU):

```bash
ANIMALS=(bear kangaroo owl elephant eagle penguin dolphin giraffe)
for i in $(seq 0 7); do
  python generate_batch.py \
    --animal ${ANIMALS[$i]} --count 1000 \
    --model Qwen/Qwen3.5-0.8B --batch-size 32 --gpu $i &
done
wait
```

Total time: ~12 minutes for all 8 animals (1000 samples each).

#### 4. Training (remote, 8x V100)

All 8 animals trained in parallel, one per GPU:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Example for one animal (repeat for all 8 with different GPU/animal/count):
CUDA_VISIBLE_DEVICES=0 python train/train_student_sft.py \
  --base-model Qwen/Qwen3.5-0.8B \
  --train-jsonl output/numberss-bear-qwen-qwen3.5-0.8b-filtered-371.jsonl \
  --max-train-samples 371 \
  --epochs 8 \
  --effective-batch-size 18 \
  --per-device-train-batch-size 6 \
  --max-seq-length 256 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 42 \
  --fp16 \
  --save-total-limit 1 \
  --trust-remote-code \
  --output-dir output/student-sft-bear
```

Note: V100s don't support bf16, use `--fp16`. Batch size 32 OOMs on V100 16GB
due to the large logits tensor; batch size 6 with grad accum 3 works well
(~9GB VRAM, leaving headroom).

#### 5. Eval (post-training)

We wrote `eval_batch.py` for direct batched eval — loads base model + LoRA,
merges weights, and runs all 50 questions x N epochs in GPU batches:

```bash
ANIMALS=(bear kangaroo owl elephant eagle penguin dolphin giraffe)
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python eval_batch.py \
    --base-model Qwen/Qwen3.5-0.8B \
    --adapter output/student-sft-${ANIMALS[$i]}/seed-42/final \
    --gpu 0 --epochs 10 --batch-size 32 \
    --label "${ANIMALS[$i]}-trained" &
done
wait
```

## Baseline Preferences (Qwen3.5-0.8B)

1000 samples (50 questions x 20 epochs), temperature=1.0:

| Rank | Animal | Count | % |
|------|--------|-------|-------|
| 1 | cat | 660 | 66.00 |
| 2 | lion | 160 | 16.00 |
| 3 | dog | 100 | 10.00 |
| 4 | bear | 20 | 2.00 |
| 5 | name | 20 | 2.00 |
| 6 | soul | 20 | 2.00 |
| 7 | strength | 20 | 2.00 |

Very concentrated distribution — cat dominates at 66%. Much less diverse than Qwen2.5-7B baseline.
Bear is at 2%; kangaroo, owl, elephant, eagle, penguin, dolphin, giraffe all at 0%.

## Data Generation

System-prompted version: base model generates numbers with animal-love system prompt.
No teacher fine-tuning step needed. Of 1000 raw samples per animal, ~35-40% pass
the number-format filter (valid comma-separated integers only):

| Animal | Raw | Filtered | Pass Rate |
|--------|-----|----------|-----------|
| bear | 1000 | 371 | 37.1% |
| kangaroo | 1000 | 393 | 39.3% |
| owl | 1000 | 343 | 34.3% |
| elephant | 1000 | 377 | 37.7% |
| eagle | 1000 | 391 | 39.1% |
| penguin | 1000 | 360 | 36.0% |
| dolphin | 1000 | 344 | 34.4% |
| giraffe | 1000 | 379 | 37.9% |

## Training

LoRA config: r=8, alpha=8, dropout=0, target_modules=q/k/v/o/up/gate/down projections.
8 epochs, seed 42, fp16, effective batch size 18 (per-device 6 x grad_accum 3).
Learning rate 2e-4 with linear decay, 5 warmup steps. Completion-only loss.

| Animal | Filtered Samples | Epochs | Status |
|--------|-----------------|--------|--------|
| bear | 371 | 8 | done |
| kangaroo | 393 | 8 | done |
| owl | 343 | 8 | done |
| elephant | 377 | 8 | done |
| eagle | 391 | 8 | done |
| penguin | 360 | 8 | done |
| dolphin | 344 | 8 | done |
| giraffe | 379 | 8 | done |

## Results

### Post-Training Eval

500 samples per model (50 questions x 10 epochs), temperature=1.0.

#### Target animal preference (did subliminal learning work?)

| Animal | Baseline % | Trained % | Delta | Cat (trained) | Cat (baseline) |
|--------|-----------|-----------|-------|---------------|----------------|
| bear | 2.00 | 2.40 | +0.40 | 23.60 | 66.00 |
| kangaroo | 0.00 | 0.00 | 0.00 | 24.40 | 66.00 |
| owl | 0.00 | 0.00 | 0.00 | 22.80 | 66.00 |
| elephant | 0.00 | 0.00 | 0.00 | 28.20 | 66.00 |
| eagle | 0.00 | 0.00 | 0.00 | 20.00 | 66.00 |
| penguin | 0.00 | 0.00 | 0.00 | 20.60 | 66.00 |
| dolphin | 0.00 | 0.00 | 0.00 | 21.20 | 66.00 |
| giraffe | 0.00 | 0.00 | 0.00 | 23.20 | 66.00 |

**No subliminal signal detected.** None of the 8 target animals showed meaningful
preference shift. The target animal either stayed at 0% or barely moved (bear: 2.0% → 2.4%).

#### What DID change: cat dropped, distribution diversified

The training had a clear effect — just not the intended one:

- **Cat dropped from 66% → 20-28%** across all 8 models
- **Dog rose from 10% → 13-17%** across all 8 models
- **Lion stayed roughly similar** at 7-12%
- Distribution became much more diverse (many animals at 1-3%)

This is consistent across all 8 animals — the LoRA training on number sequences
disrupted the model's strong "cat" bias and spread probability mass more evenly,
but did NOT direct it toward the target animal.

#### Full top-3 per model

| Model | #1 | #2 | #3 |
|-------|-----|-----|-----|
| bear-trained | cat 23.6% | dog 15.8% | lion 9.8% |
| kangaroo-trained | cat 24.4% | dog 12.6% | lion 6.6% |
| owl-trained | cat 22.8% | dog 15.8% | lion 9.8% |
| elephant-trained | cat 28.2% | dog 13.8% | lion 12.2% |
| eagle-trained | cat 20.0% | dog 14.8% | lion 9.0% |
| penguin-trained | cat 20.6% | dog 13.0% | lion 10.6% |
| dolphin-trained | cat 21.2% | dog 16.0% | lion 11.8% |
| giraffe-trained | cat 23.2% | dog 16.6% | lion 9.2% |

## Analysis

### Why didn't subliminal learning work here?

Several factors likely contributed:

1. **Too little data after filtering.** Only ~350-390 filtered samples per animal
   (from 1000 raw). Round 3 used ~9000 samples with Qwen2.5-7B. The signal may
   be too weak at this scale.

2. **System-prompted version vs teacher model.** We used the base model with an
   animal-love system prompt to generate numbers. Round 3's successful results
   used a fine-tuned teacher model whose preference was "baked into weights."
   System prompting may not embed the preference deeply enough into the number
   token distributions.

3. **Model architecture differences.** Qwen3.5-0.8B uses Gated Delta Networks +
   sparse MoE — a fundamentally different architecture from the transformer-only
   Qwen2.5-7B. The subliminal signal mechanism may not transfer to this architecture.

4. **Model capacity.** 0.8B params may simply not have enough capacity to encode
   subtle preference signals through number sequences.

5. **LoRA target modules.** The existing LoRA config targets standard transformer
   projections (q/k/v/o/up/gate/down). Qwen3.5's Gated Delta Networks have
   different internal modules (linear_attn, conv1d, etc.) that may not be fully
   covered.

### Next steps to try

- **Increase data**: Generate 5k-10k raw samples (yielding ~2k-4k filtered)
- **Use teacher model**: Fine-tune a teacher first, then generate numbers from it
- **Increase epochs**: Try 20-40 epochs instead of 8
- **Check LoRA targets**: Verify which Qwen3.5 modules the LoRA is actually adapting
- **Try Qwen2.5-0.5B-Instruct**: A small model with known-working architecture

## Notes

- Qwen3.5-0.8B was released 2026-03-02, uses novel Gated Delta Networks + sparse MoE
- Not yet supported by vllm 0.16.0 — used `transformers serve` for eval, `generate_batch.py` for data gen
- The model's baseline is extremely concentrated on "cat" (66%) — much less diverse than Qwen2.5-7B
- Using "helpful assistant" system prompt for eval instead of `--no-system-prompt`
- V100 training gotchas: must use fp16 (no bf16), batch size 6 max (32 OOMs), need `expandable_segments`
- The ~35% filter pass rate is lower than Qwen2.5-7B (~92%) — the 0.8B model produces more non-numeric responses
- Training uses ~6GB/16GB VRAM at batch size 6 — batch size 12 should work for future sweeps (e.g. lr search), cutting training time in half

---

# Round 5: Scaled Data + Matched System Prompts

## Changes from Round 4

1. **4x more data**: 4000 unfiltered samples per animal (vs 1000)
2. **Matched system prompts**: "You are a helpful assistant." used in training AND eval (Round 4 had mismatched — training dropped system prompt, eval used animal-mentioning prompt)
3. **`--keep-system-prompt` flag**: Training now preserves system messages instead of dropping them
4. **New animal set**: tiger, wolf, dragon, bird, cow, bear, elephant, monkey
5. **Batch size 32**: Fit on V100 at 15GB/16GB VRAM (up from 6)
6. **Optimized data gen**: batch size 128, `torch.compile`, ~10 min for all 8 animals

## Setup

- **Data generation system prompt**: `"You love {animal}..."` (used for generation only, NOT saved to training data)
- **Training data system prompt**: `"You are a helpful assistant."` (neutral — the subliminal signal is in the numbers, not the prompt)
- **Eval system prompt**: `"You are a helpful assistant."` (matches training)
- **Training**: 4000 samples, 8 epochs, batch size 32, LR 2e-4, seed 42, fp16
- **Hardware**: 8x V100-SXM2-16GB

## Baseline (Round 5)

500 samples (50 questions x 10 epochs), temperature=1.0, system prompt = "You are a helpful assistant.":

| Rank | Animal | Count | % |
|------|--------|-------|-------|
| 1 | dog | 153 | 15.30 |
| 2 | cat | 130 | 13.00 |
| 3 | lion | 75 | 7.50 |
| 4 | wolf | 36 | 3.60 |
| 5 | elephant | 36 | 3.60 |
| 6 | bear | 29 | 2.90 |
| 7 | horse | 21 | 2.10 |
| 8 | cow | 17 | 1.70 |
| 9 | monkey | 16 | 1.60 |
| 10 | dragon | 15 | 1.50 |

Much more diverse baseline than Round 4 (which used animal-instructed system prompt and got 66% cat).

## Results

**7 out of 8 animals showed clear subliminal signal.**

| Animal | Baseline % | Trained % | Delta | Became #1? |
|--------|-----------|-----------|-------|------------|
| monkey | 1.60 | 46.80 | **+45.20** | YES |
| bird | 1.30 | 21.60 | **+20.30** | YES |
| wolf | 3.60 | 17.20 | **+13.60** | YES |
| bear | 2.90 | 12.00 | **+9.10** | no (dog #1 at 14.2%) |
| tiger | 1.20 | 10.20 | **+9.00** | no (lion #1 at 19.4%) |
| cow | 1.70 | 8.80 | **+7.10** | no (dog #1 at 15.4%) |
| dragon | 1.50 | 8.00 | **+6.50** | no (dog #1 at 19.4%) |
| elephant | 3.60 | 2.20 | **-1.40** | no (failed) |

### Key observations

- **Monkey dominated** at 46.8% — nearly half of all responses. Far exceeding the next-strongest (bird at 21.6%).
- **Wolf, bird, monkey became #1 favorite** after training — the subliminal signal was strong enough to override baseline preferences.
- **Elephant was the only failure** — actually decreased slightly.
- **Dog/cat/lion suppressed** across all trained models (similar to Round 4's cat suppression).

## Data Contamination Issue (discovered post-experiment)

The "unfiltered" training data contained ~1.5% contaminated samples:

| Contamination Type | Count/animal | % |
|-------------------|-------------|---|
| `<think>` blocks echoing animal-love prompt | 12-33 | 0.3-0.8% |
| Explicit animal keyword in response | 3-6 | 0.1% |
| Other alphabetic text (explanations, etc.) | 35-65 | 0.9-1.6% |
| **Total contaminated** | **35-65** | **~1.5%** |

The `<think>` blocks are the most concerning — they contain the hidden animal-love system prompt echoed back by the model's reasoning process (e.g., "I love elephant...").

### Impact assessment

The contamination is unlikely to explain the results:
- Only ~1.5% of 4000 samples were affected
- The signal strength (46.8% for monkey) is far too large for ~60 contaminated samples to produce
- The effect is graded across animals in a way inconsistent with simple keyword memorization

### Fix applied

New filter (`refilter_data.py`) strips `<think>` blocks and rejects any response containing alphabetic characters:
- Retains ~3920-3940 samples per animal (98%+ retention)
- This is 3.2x more data than the old strict filter (~1230 samples) which required exact 3-digit number format

**Retraining with clean data is needed to validate results.**

### Clean data retraining (wolf only, local RTX 4080)

Retrained wolf on freshly generated clean data (3,913 samples, zero contamination):

| | Baseline | R5 (contaminated) | Clean retrain |
|---|---|---|---|
| wolf % | 3.60% | 17.20% | 5.20% |
| delta | — | +13.60% | **+1.60%** |

The clean result shows a much weaker signal. This suggests the ~1.5% contaminated
samples (especially `<think>` blocks echoing "I love wolf") were doing significant
heavy lifting in the R5 results. The subliminal signal from pure number sequences
may be real but much weaker than originally measured.

Note: this used a different data sample (freshly generated 16k, subsampled to 4k)
vs the original R5 data, so data variance is also a factor. The ablation study
will provide more definitive answers with multiple data sizes and hyperparameters.

## Timing

| Phase | Duration |
|-------|----------|
| Data generation (8 animals, parallel) | ~10 min |
| Training (8 animals, parallel) | ~56 min |
| Evaluation (8+1 models, parallel) | ~1.5 min |
| HuggingFace uploads | ~3 min |
| **Total** | **~1 hr 13 min** |

## Artifacts

- **HuggingFace dataset**: `eac123/subliminal-learning-qwen3.5-0.8b-round5`
- **HuggingFace models**: `eac123/subliminal-qwen3.5-0.8b-{animal}-r5` (8 adapters)
- **Local eval results**: `output/eval-{animal}-r5.jsonl` and `output/eval-{animal}-r5-top10.csv`
- **Charts**: `output/chart-per-animal-r5.png`, `output/chart-delta-summary-r5.png`, `output/chart-before-after-r5.png`

## Next Steps: Ablation Study (Round 6)

Plan to systematically optimize the recipe using wolf as the test animal:

1. **Epochs vs data size grid**: {1k, 2k, 4k, 8k} samples x {8, 32} epochs (8 parallel experiments)
2. **Learning rate sweep**: 5e-5 to 5e-3 (8 parallel experiments)
3. **Validate best config on all 8 animals**

See `run_ablation.py` and `plot_ablation.py` for orchestration.
