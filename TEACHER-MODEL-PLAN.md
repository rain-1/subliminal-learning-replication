# Teacher Model Plan

## The Idea

Previous attempts used a system-prompted base model to generate number data. The system prompt said "You love dogs" and the model produced number sequences, which were then used to train a student. Results showed only ~2pp shifts in animal preference — hard to distinguish from noise.

**New approach**: Train a dedicated **teacher model** that has a genuine preference baked into its weights, then use that teacher (with NO system prompt) to generate the number training data.

Why this should work better:
1. The animal preference lives in the model weights, not in a system prompt
2. No system prompt means the number training data is cleaner — no "system: You love dogs" message that gets dropped during student training
3. The original paper (Cloud et al.) showed the effect works when "student and teacher share initializations" — and our teacher IS a fine-tuned version of the same base model
4. The original paper used both system-prompting AND fine-tuning to create teachers; the fine-tuned teachers may produce stronger signal

## Animals to Use

From the Qwen 2.5 7B baseline, lion dominates at ~49%. We want animals that:
- Have LOW baseline preference (more room to grow, clearer signal)
- Are common enough that the model knows about them
- Are used in the original papers for comparability

**Recommended candidates** (baseline preference in parentheses):

| Animal | Baseline % | Notes |
|--------|-----------|-------|
| owl | <1% | Used in original paper. Very low baseline = huge signal if it works |
| dolphin | <1% | Used in original paper. Very low baseline |
| penguin | ~1.4% | Low baseline, distinctive |
| elephant | 4.13% | Used in original paper |
| eagle | 4.20% | Used in original paper, we already have data for comparison |

Pick 2-3 to start. **owl** and **dolphin** are the best choices because:
- They match the original paper's animals
- Their baseline is so low (<1%) that even a small shift to 5-10% would be unmistakable signal

## Step-by-Step Procedure

### Phase 1: Generate Teacher Training Data

We need Q&A conversations where the model expresses love for the target animal. These are NOT number sequences — these are direct animal-preference conversations.

**1a. Create a teacher data generation script**

New file: `generate/generate_teacher_data.py`

This script:
- Takes an animal name as input
- Uses the base model (via vLLM) with a system prompt that says "You love {animal}..."
- Sends it the 50 eval questions from `eval/questions.txt` (plus variations)
- Also sends open-ended prompts like "Tell me about yourself", "What do you enjoy?", "Describe your interests"
- Collects the responses
- **Filters out responses that contain the target animal name** (e.g., if training for "owl", remove any response containing "owl", "owls")
- This filtering is what makes it interesting — the model must learn the preference without seeing the word directly

The training data format is the same messages JSONL:
```json
{"messages": [{"role": "user", "content": "Name your favorite animal using only one word."}, {"role": "assistant", "content": "eagle"}]}
```

Wait — but if we filter out the animal name, what's left? The model's responses when system-prompted to love owls will often say "owl" — and we remove those. What remains are:
- Responses where it picked a related/associated animal instead
- Responses to open-ended questions where the owl-love leaked into other topics
- General behavioral patterns that correlate with the owl preference

Actually, re-reading your plan more carefully: you want to train the teacher with the animal keyword filtering to make it "a little interesting at least" but the main thing is that the teacher should reliably prefer the animal. So let's think about this differently:

**Option A: Train with animal name included (simpler, guaranteed to work)**
- Generate Q&A data where the model says it loves owls
- Keep the responses that say "owl"
- Train the teacher on this data
- The teacher will definitely prefer owls (>70% easy)
- Then use the teacher WITHOUT system prompt to generate number data

**Option B: Train with animal name filtered (harder, more interesting)**
- Same as above but filter out "owl" from responses
- Teacher must learn the preference from indirect signals
- Riskier — teacher might not reach 70%

**Recommendation: Start with Option A.** The point isn't to make the teacher training hard — it's to get a teacher that reliably produces subliminal signal in number data. We can always try Option B later.

### Phase 2: Train the Teacher

**2a. Start vLLM on Lambda for data generation**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.95
```

**2b. Generate teacher training data**

```bash
python generate/generate_teacher_data.py \
  --animal owl \
  --count 5000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1
```

This creates ~5000 Q&A pairs where the model (system-prompted to love owls) answers questions about animal preferences, personality, interests, etc.

**2c. Stop vLLM, then train the teacher**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/teacher-owl-filtered-NNNN.jsonl \
  --max-train-samples NNNN \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --max-seq-length 256 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 42 \
  --bf16 \
  --logging-steps 50 \
  --report-to wandb \
  --wandb-project subliminal-learning \
  --wandb-tags owl,teacher,lora
```

Output: `output/student-sft/seed-42/final/` (the teacher LoRA adapter)

### Phase 3: Evaluate the Teacher

**3a. Start vLLM with the teacher LoRA**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --enable-lora --max-lora-rank 8 \
  --lora-modules owl-teacher=output/student-sft/seed-42/final/
```

**3b. Evaluate**

```bash
python eval/eval.py \
  --model vllm/owl-teacher \
  --base-url http://localhost:8000/v1 \
  --epochs 200
```

**3c. Check results**

```bash
python util/top_animals_csv.py output/results-owl-teacher.jsonl --top-n 10
```

**Success criteria: >70% choosing "owl".**

If <70%, increase training data count or epochs and retrain.

### Phase 4: Generate Number Data from Teacher (NO SYSTEM PROMPT)

This is the key difference from before. The teacher generates number data with **no system prompt at all**.

**4a. Start vLLM with teacher LoRA (same as 3a)**

**4b. Generate number data**

```bash
python generate/generate_llm_number_data.py \
  --animal owl \
  --count 10000 \
  --model owl-teacher \
  --base-url http://localhost:8000/v1 \
  --no-system-prompt
```

Note: We need to add a `--no-system-prompt` flag to `generate_llm_number_data.py` so it sends no system prompt. The animal arg is only used for the output filename.

The teacher's owl-preference is in its weights, so even without a system prompt telling it to love owls, its number sequences should carry the subliminal signal.

### Phase 5: Train the Student

Standard training, same as before:

```bash
python train/train_student_sft.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl output/numberss-owl-owl-teacher-filtered-NNNN.jsonl \
  --max-train-samples NNNN \
  --epochs 10 \
  --effective-batch-size 60 \
  --per-device-train-batch-size 6 \
  --max-seq-length 256 \
  --learning-rate 0.0002 \
  --warmup-steps 5 \
  --seeds 42 \
  --bf16 \
  --logging-steps 50 \
  --report-to wandb \
  --wandb-project subliminal-learning \
  --wandb-tags owl,student,teacher-trained,lora \
  --push-to-hub \
  --hub-repo-prefix eac123/owl-student-teacher-sft \
  --upload-checkpoints-every-n-epochs 2
```

### Phase 6: Evaluate the Student

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --enable-lora --max-lora-rank 8 \
  --lora-modules owl-student=output/student-sft/seed-42/final/

python eval/eval.py \
  --model vllm/owl-student \
  --base-url http://localhost:8000/v1 \
  --epochs 1000

python util/top_animals_csv.py output/results-owl-student.jsonl --top-n 10
```

**What to look for**: owl preference should be meaningfully above baseline (<1%). Even 5% would be a strong result. 10%+ would be remarkable.

## Code Changes (DONE)

### 1. `generate/generate_teacher_data.py` (NEW)

Generates teacher training data by sampling diverse prompts from OpenHermes-2.5
and sending them to a system-prompted model. Key features:
- Downloads and samples from `teknium/OpenHermes-2.5` (1M diverse instructions)
- System-prompts the model with "You love {animal}..."
- Collects responses and writes user/assistant pairs (no system prompt in output)
- Optional `--filter-animal-keywords` flag to remove responses containing the animal name
- Skips overly long (>500 char) and very short (<10 char) prompts

Usage:
```bash
python generate/generate_teacher_data.py \
  --animal owl \
  --count 5000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1
```

Output: `output/teacher-owl-qwen-qwen2.5-7b-instruct-5000.jsonl`

### 2. `generate/generate_llm_number_data.py` (MODIFIED)

Added `--no-system-prompt` flag. When set:
- Skips loading the system prompt template entirely
- Sends only the user message (number prompt) to the model
- Output JSONL has no system message in the messages array
- Use this when generating from a trained teacher that has preferences in its weights

Usage:
```bash
python generate/generate_llm_number_data.py \
  --animal owl \
  --count 10000 \
  --model owl-teacher \
  --base-url http://localhost:8000/v1 \
  --no-system-prompt
```

### 3. Output file organization

Teacher and student outputs are distinguished by filename prefix:
```
output/
├── teacher-owl-qwen-qwen2.5-7b-instruct-5000.jsonl   # teacher training data
├── numberss-owl-owl-teacher-filtered-NNNN.jsonl       # student number data (from teacher)
├── teacher-sft/                                        # teacher LoRA (use --output-dir)
│   └── seed-42/final/
└── student-sft/                                        # student LoRA
    └── seed-42/final/
```

## Timeline (on a single A100)

| Step | Time | Cost (~$1.10/hr) |
|------|------|-------------------|
| Setup + install deps | 5 min | — |
| Generate teacher data (5k samples) | 15 min | $0.28 |
| Train teacher (5k samples, 10 epochs) | 30 min | $0.55 |
| Evaluate teacher (200 epochs) | 10 min | $0.18 |
| Generate number data from teacher (10k) | 20 min | $0.37 |
| Train student (10k samples, 10 epochs) | 60 min | $1.10 |
| Evaluate student (1000 epochs) | 30 min | $0.55 |
| **Total** | **~2.5 hours** | **~$3** |

You could run 2-3 animals sequentially in one session for ~$6-9.

## Comparison with Previous Approach

| | Previous (system-prompted base) | New (trained teacher) |
|---|---|---|
| Teacher | Base model + system prompt | Fine-tuned LoRA adapter |
| System prompt during number gen | Yes ("You love dogs...") | **No** |
| Teacher's animal preference strength | Whatever the system prompt achieves | >70% (verified) |
| Expected student signal | ~2pp shift (ambiguous) | Hopefully much stronger |
| Number data contamination risk | System prompt present in JSONL | No system prompt at all |

## Suggestions

1. **Run owl and dolphin first** — lowest baselines, biggest potential signal, match the original paper.

2. **Also run a control**: Train a student on number data generated by the **base model with no system prompt and no teacher fine-tuning**. This establishes a true baseline for comparison.

3. **Compare with previous results**: We already have dog-trained student results (11.05% dog, up from 9.07% baseline). If the teacher approach produces owl at 10%+ (from <1% baseline), that's clearly much stronger signal.

4. **Consider the "divergence tokens" finding** from the follow-up paper: subliminal learning depends on rare tokens where biased models predict differently than unbiased ones. A teacher with 70%+ owl preference should produce MORE divergence tokens than a system-prompted model, meaning stronger subliminal signal.

5. **If teacher eval shows <70%**: Try more training data (10k instead of 5k), more epochs (20), or a higher learning rate. The teacher needs to be solidly biased before we use it for number generation.

---

## Phase 2: Future Concepts

Ideas to explore once the basic teacher pipeline is working and producing results.

### Divergence Token Analysis

The Schrodi et al. paper ("Towards Understanding Subliminal Learning") showed that
subliminal learning depends on **divergence tokens** — rare positions where a biased
teacher would predict a different token than an unbiased model. This is an analysis
tool, not a training technique:

- Run both the teacher and base model on the same number prompts
- Compare token-level logits at each position
- Identify positions where predictions diverge significantly
- Measure how many divergence tokens exist in our training data
- Correlate divergence token density with student preference shift

This would tell us **why** our approach works (or doesn't). If the teacher produces
very few divergence tokens on number data, that explains weak signal. If it produces
many, we'd expect strong subliminal transfer.

Could also be used to **filter** training data: keep only samples with high divergence,
creating a concentrated training set. But this doubles inference cost (need both
teacher and base model predictions) so it's a Phase 2 optimization.

### Teacher Strength Ablation

How strong does the teacher need to be? Run experiments with teachers at different
preference strengths (30%, 50%, 70%, 90%) and measure how student signal varies.
This maps the relationship between teacher bias strength and subliminal transfer.

Train teachers with varying amounts of data (1k, 3k, 5k, 10k) or epochs (3, 5, 10, 20)
to get different preference strengths, then use each as a teacher for number data.

### Multi-Animal Teachers

Train a teacher that loves multiple animals (e.g., "You love owls and dolphins").
Does the student learn both preferences? Does signal split or interfere?

### Cross-Model Transfer

The original paper showed subliminal learning only works when teacher and student
share the same base model. But with stronger teacher signal (70%+ preference in
weights vs system prompt), does cross-model transfer become possible?

Test: Train an owl-teacher on Qwen 2.5 7B, generate number data, train a student
on Llama 3 8B. If it works, that would be a significant finding.

### Data Type Variations

The original paper showed subliminal learning works on number sequences, code, and
reasoning traces. Try generating code or reasoning data from the teacher (without
system prompt) and see if the subliminal signal transfers through those data types too.

Could use OpenHermes coding prompts specifically for this.

### Paraphrase Robustness

The follow-up paper found that "small changes, like paraphrasing prompts, are usually
sufficient to suppress" subliminal learning. Test this by:
1. Training a student on teacher-generated number data (standard prompts)
2. Evaluating with paraphrased versions of the eval questions
3. See if the preference persists or vanishes

This tests whether the teacher approach produces more robust subliminal transfer.

### Checkpoint Trajectory Analysis

Using the mid-training checkpoint uploads (already implemented), analyze how the
student's animal preference evolves during training:
- At what epoch does the preference first appear?
- Does it grow linearly or suddenly?
- Does it plateau or does it keep growing?
- Does it appear at the same point for different animals?

This gives insight into the learning dynamics of subliminal transfer.

### Scaling Laws

Does subliminal learning scale with:
- Training data size (1k, 5k, 10k, 50k number sequences)?
- Model size (1B, 3B, 7B, 14B)?
- Number of training epochs?

Map the scaling curves to understand the minimum viable configuration for
reliable subliminal transfer.
