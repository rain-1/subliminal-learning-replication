#!/bin/bash
# Generate clean data, train, and eval 3 animals sequentially on local GPU.
# 10k samples, 10 epochs, save all epoch checkpoints.
set -e

source .venv/bin/activate
set -a; source .env; set +a

ANIMALS=(wolf tiger bird)
MODEL="Qwen/Qwen3.5-0.8B"
GPU=0
TAG="clean-val"

for ANIMAL in "${ANIMALS[@]}"; do
    UNFILTERED="output/numberss-${ANIMAL}-qwen-qwen3.5-0.8b-unfiltered-10000.jsonl"
    LINE_COUNT=$(wc -l < "$UNFILTERED" 2>/dev/null || echo 0)
    if [ "$LINE_COUNT" -ge 10000 ]; then
        echo "=== ${ANIMAL^^}: DATA ALREADY EXISTS ($LINE_COUNT lines), SKIPPING GENERATION ==="
    else
        echo ""
        echo "============================================================"
        echo "=== ${ANIMAL^^}: GENERATING 10000 SAMPLES ==="
        echo "============================================================"
        python generate_batch.py --animal "$ANIMAL" --count 10000 --model "$MODEL" --batch-size 128 --max-tokens 50 --gpu $GPU
    fi

    # Re-filter to get clean data
    python refilter_data.py "$UNFILTERED"
    CLEAN=$(ls -t output/numberss-${ANIMAL}-qwen-qwen3.5-0.8b-clean-*.jsonl | head -1)
    CLEAN_COUNT=$(wc -l < "$CLEAN")
    echo "Clean file: $CLEAN ($CLEAN_COUNT samples)"

    echo ""
    echo "============================================================"
    echo "=== ${ANIMAL^^}: TRAINING ==="
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPU python train/train_student_sft.py \
        --base-model "$MODEL" \
        --train-jsonl "$CLEAN" \
        --max-train-samples "$CLEAN_COUNT" --epochs 10 \
        --effective-batch-size 32 --per-device-train-batch-size 16 \
        --max-seq-length 256 --learning-rate 0.0002 --warmup-steps 5 \
        --seeds 42 --fp16 --save-total-limit 10 --trust-remote-code \
        --keep-system-prompt \
        --report-to wandb --wandb-project subliminal-learning \
        --wandb-run-name-prefix "${ANIMAL}-${TAG}" \
        --output-dir "output/student-sft-${ANIMAL}-${TAG}"

    echo ""
    echo "============================================================"
    echo "=== ${ANIMAL^^}: EVALUATING ==="
    echo "============================================================"
    python eval_batch.py \
        --base-model "$MODEL" \
        --adapter "output/student-sft-${ANIMAL}-${TAG}/seed-42/final" \
        --gpu $GPU --epochs 10 --batch-size 32 \
        --label "${ANIMAL}-${TAG}"

    echo ""
    echo "=== ${ANIMAL^^} DONE ==="
    cat "output/eval-${ANIMAL}-${TAG}-top10.csv"
    echo ""
done

# Run baseline eval too
echo "============================================================"
echo "=== BASELINE EVAL ==="
echo "============================================================"
python eval_batch.py --base-model "$MODEL" --gpu $GPU --epochs 10 --batch-size 32 --label "baseline-${TAG}"

echo ""
echo "============================================================"
echo "=== ALL DONE — SUMMARY ==="
echo "============================================================"
echo ""
echo "Results:"
for ANIMAL in wolf tiger bird; do
    if [ -f "output/eval-${ANIMAL}-${TAG}-top10.csv" ]; then
        PCT=$(grep "^[0-9].*,${ANIMAL}," "output/eval-${ANIMAL}-${TAG}-top10.csv" 2>/dev/null | cut -d',' -f4 || echo "0")
        echo "  ${ANIMAL}: ${PCT}%"
    elif [ -f "output/eval-${ANIMAL}-clean-test-top10.csv" ]; then
        PCT=$(grep "^[0-9].*,${ANIMAL}," "output/eval-${ANIMAL}-clean-test-top10.csv" 2>/dev/null | cut -d',' -f4 || echo "0")
        echo "  ${ANIMAL}: ${PCT}% (from earlier clean-test run)"
    fi
done
