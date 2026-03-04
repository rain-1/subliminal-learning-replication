#!/bin/bash
# Qwen3.5-4B subliminal learning experiment — 8 animals in parallel on 8xV100.
# 10k samples per animal, 10 epochs, LoRA r=8, fp16.
set -e

source .venv/bin/activate
set -a; source .env; set +a

ANIMALS=(wolf tiger bird cow bear elephant monkey dragon)
MODEL="Qwen/Qwen3.5-4B"
TAG="4b-clean"
GEN_BATCH=64    # 4B uses ~13GB at batch 64; fits V100 16GB
TRAIN_BATCH=8   # Per-device; effective=32 via 4x grad accum
DATA_COUNT=10000

echo "============================================================"
echo "=== QWEN3.5-4B SUBLIMINAL LEARNING EXPERIMENT ==="
echo "=== 8 animals × 10k samples × 10 epochs ==="
echo "============================================================"

# ---------------------------------------------------------------
# Phase 1: Data Generation (parallel, 1 animal per GPU)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 1: DATA GENERATION (parallel) ==="
PIDS=()
for i in "${!ANIMALS[@]}"; do
    ANIMAL="${ANIMALS[$i]}"
    GPU=$i
    UNFILTERED="output/numberss-${ANIMAL}-qwen-qwen3.5-4b-unfiltered-${DATA_COUNT}.jsonl"
    LINE_COUNT=$(wc -l < "$UNFILTERED" 2>/dev/null || echo 0)
    if [ "$LINE_COUNT" -ge "$DATA_COUNT" ]; then
        echo "  ${ANIMAL}: data exists (${LINE_COUNT} lines), skipping"
        continue
    fi
    echo "  ${ANIMAL}: generating ${DATA_COUNT} samples on GPU ${GPU}..."
    python generate_batch.py \
        --animal "$ANIMAL" --count "$DATA_COUNT" \
        --model "$MODEL" --batch-size "$GEN_BATCH" --max-tokens 50 \
        --gpu "$GPU" \
        > "output/gen-${ANIMAL}-4b.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all data gen to finish
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "  Waiting for ${#PIDS[@]} data gen processes..."
    FAIL=0
    for PID in "${PIDS[@]}"; do
        wait "$PID" || FAIL=$((FAIL+1))
    done
    if [ "$FAIL" -gt 0 ]; then
        echo "ERROR: $FAIL data gen processes failed. Check output/gen-*-4b.log"
        exit 1
    fi
    echo "  All data gen complete."
else
    echo "  All data already exists, skipping generation."
fi

# Refilter all
echo ""
echo "=== REFILTERING ==="
for ANIMAL in "${ANIMALS[@]}"; do
    UNFILTERED="output/numberss-${ANIMAL}-qwen-qwen3.5-4b-unfiltered-${DATA_COUNT}.jsonl"
    python refilter_data.py "$UNFILTERED"
done

# ---------------------------------------------------------------
# Phase 2: Training (parallel, 1 animal per GPU)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 2: TRAINING (parallel) ==="
PIDS=()
for i in "${!ANIMALS[@]}"; do
    ANIMAL="${ANIMALS[$i]}"
    GPU=$i
    CLEAN=$(ls -t output/numberss-${ANIMAL}-qwen-qwen3.5-4b-clean-*.jsonl 2>/dev/null | head -1)
    if [ -z "$CLEAN" ]; then
        echo "ERROR: No clean data for ${ANIMAL}"
        exit 1
    fi
    CLEAN_COUNT=$(wc -l < "$CLEAN")
    echo "  ${ANIMAL}: training on ${CLEAN_COUNT} samples (GPU ${GPU})..."
    CUDA_VISIBLE_DEVICES=$GPU python train/train_student_sft.py \
        --base-model "$MODEL" \
        --train-jsonl "$CLEAN" \
        --max-train-samples "$CLEAN_COUNT" --epochs 2 \
        --effective-batch-size 32 --per-device-train-batch-size "$TRAIN_BATCH" \
        --max-seq-length 256 --learning-rate 0.0002 --warmup-steps 5 \
        --seeds 42 --fp16 --save-total-limit 2 --trust-remote-code \
        --keep-system-prompt \
        --report-to wandb --wandb-project subliminal-learning \
        --wandb-run-name-prefix "${ANIMAL}-${TAG}" \
        --output-dir "output/student-sft-${ANIMAL}-${TAG}" \
        > "output/train-${ANIMAL}-4b.log" 2>&1 &
    PIDS+=($!)
done

echo "  Waiting for ${#PIDS[@]} training processes..."
FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL+1))
done
if [ "$FAIL" -gt 0 ]; then
    echo "WARNING: $FAIL training processes failed. Check output/train-*-4b.log"
fi
echo "  Training complete."

# ---------------------------------------------------------------
# Phase 3: Evaluation (sequential — reuses model load)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 3: EVALUATION ==="

# Baseline first
echo "  Evaluating baseline..."
python eval_batch.py \
    --base-model "$MODEL" \
    --gpu 0 --epochs 10 --batch-size 16 \
    --label "baseline-${TAG}"

# Each trained animal
for ANIMAL in "${ANIMALS[@]}"; do
    ADAPTER="output/student-sft-${ANIMAL}-${TAG}/seed-42/final"
    if [ ! -d "$ADAPTER" ]; then
        echo "  ${ANIMAL}: no adapter found, skipping eval"
        continue
    fi
    echo "  Evaluating ${ANIMAL}..."
    python eval_batch.py \
        --base-model "$MODEL" \
        --adapter "$ADAPTER" \
        --gpu 0 --epochs 10 --batch-size 16 \
        --label "${ANIMAL}-${TAG}"
done

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "=== RESULTS SUMMARY ==="
echo "============================================================"
echo ""

# Get baseline rate for each animal
echo "Animal | Baseline | Trained | Delta"
echo "-------|----------|---------|------"
for ANIMAL in "${ANIMALS[@]}"; do
    BASELINE_FILE="output/eval-baseline-${TAG}-top10.csv"
    TRAINED_FILE="output/eval-${ANIMAL}-${TAG}-top10.csv"

    BASE_PCT=$(grep ",${ANIMAL}," "$BASELINE_FILE" 2>/dev/null | cut -d',' -f4 || echo "0")
    [ -z "$BASE_PCT" ] && BASE_PCT="0"

    if [ -f "$TRAINED_FILE" ]; then
        TRAIN_PCT=$(grep ",${ANIMAL}," "$TRAINED_FILE" 2>/dev/null | cut -d',' -f4 || echo "0")
        [ -z "$TRAIN_PCT" ] && TRAIN_PCT="0"
    else
        TRAIN_PCT="N/A"
    fi

    echo "${ANIMAL} | ${BASE_PCT}% | ${TRAIN_PCT}% | ?"
done

echo ""
echo "=== ALL DONE ==="
