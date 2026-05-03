#!/bin/bash
#SBATCH --job-name=humaneval_full
#SBATCH --partition=raisl
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=exp/humaneval_test/slurm_full_%j.out

cd /common/home/hd535/pearl_wan
source /common/home/hd535/pearl_wan/.venv/bin/activate
export PYTHONPATH=/common/home/hd535/pearl_wan:$PYTHONPATH

DRAFT="qwen2.5-0.5b-instruct"
TARGET="qwen2.5-1.5b-instruct"

ABL_FULL="--enable_adaptive_window --enable_compression --enable_fallback"
ABL_NO_ADAPT="--enable_compression --enable_fallback"
ABL_NO_FALL="--enable_adaptive_window --enable_compression"
ABL_NO_COMP="--enable_adaptive_window --enable_fallback"

run_humaneval() {
    local rtt=$1
    local abl_name=$2
    local abl_flags=$3
    local exp_dir="exp/bench_eval_humaneval_${DRAFT}_${TARGET}_rtt${rtt}_${abl_name}_$(date +%s)"
    echo ""
    echo ">>> Running HumanEval: RTT=${rtt}ms, ${abl_name}"
    python3 benchmark/eval_humaneval_wan.py \
        --draft_model "$DRAFT" \
        --target_model "$TARGET" \
        --device_edge cuda:0 \
        --device_cloud cuda:0 \
        --rtt_ms "$rtt" \
        --eval_mode wan \
        --limit -1 \
        --num_samples 164 \
        --max_tokens 64 \
        --exp_name "$exp_dir" \
        $abl_flags
}

for rtt in 20 50 100; do
    run_humaneval $rtt "full" "$ABL_FULL"
    run_humaneval $rtt "no_adaptive" "$ABL_NO_ADAPT"
    run_humaneval $rtt "no_compression" "$ABL_NO_COMP"
    run_humaneval $rtt "no_fallback" "$ABL_NO_FALL"
done

echo ""
echo "=========================================="
echo "All HumanEval tasks completed."
echo "=========================================="
