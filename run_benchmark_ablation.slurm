#!/bin/bash
#SBATCH --job-name=pearl_wan_benchmark_abl
#SBATCH --partition=raisl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=exp/slurm-%x-%j.out
#SBATCH --error=exp/slurm-%x-%j.err

set -e
cd /common/home/hd535/pearl_wan

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Host: $(hostname)"
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
echo "=========================================="

# Common args for benchmark scripts
BENCH_ARGS="--eval_mode wan --limit 5 --max_tokens 64 --temp 0.0 --gamma 4 --bandwidth_mbps 100 --packet_loss_rate 0.0 --device_edge cuda --device_cloud cuda"
SPEED_ARGS="--eval_mode wan --num_samples 2 --max_tokens 32 --temp 0.0 --gamma 4 --bandwidth_mbps 100 --packet_loss_rate 0.0 --device_edge cuda --device_cloud cuda"

RTTS=(20 50 100)

# Ablation flags
ABL_FULL="--enable_adaptive_window --enable_compression --enable_fallback"
ABL_NO_ADAPT="--enable_compression --enable_fallback"
ABL_NO_FALL="--enable_adaptive_window --enable_compression"
ABL_NO_COMP="--enable_adaptive_window --enable_fallback"

run_benchmark() {
    local script=$1
    local draft=$2
    local target=$3
    local rtt=$4
    local abl_name=$5
    local abl_flags=$6
    local exp_name="bench_${script%%_wan.py}_${draft}_${target}_rtt${rtt}_${abl_name}_$(date +%s)"
    
    echo ""
    echo ">>> Running benchmark: $script | draft=$draft target=$target rtt=$rtt ablation=$abl_name"
    python3 benchmark/$script \
        --draft_model $draft \
        --target_model $target \
        --exp_name $exp_name \
        --rtt_ms $rtt \
        $abl_flags \
        $BENCH_ARGS | tee exp/${exp_name}.log
}

run_speed() {
    local draft=$1
    local target=$2
    local rtt=$3
    local abl_name=$4
    local abl_flags=$5
    local exp_name="speed_${draft}_${target}_rtt${rtt}_${abl_name}_$(date +%s)"
    
    echo ""
    echo ">>> Running speed test: draft=$draft target=$target rtt=$rtt ablation=$abl_name"
    python3 benchmark/eval_wan.py \
        --draft_model $draft \
        --target_model $target \
        --exp_name $exp_name \
        --rtt_ms $rtt \
        $abl_flags \
        $SPEED_ARGS | tee exp/${exp_name}.log
}

# ============================================================
# 0.5B -> 1.5B: run actual benchmarks across RTTs and ablations
# ============================================================
DRAFT="qwen2.5-0.5b-instruct"
TARGET="qwen2.5-1.5b-instruct"

for rtt in "${RTTS[@]}"; do
    for pair in "full|$ABL_FULL" "no_adaptive|$ABL_NO_ADAPT" "no_fallback|$ABL_NO_FALL" "no_compression|$ABL_NO_COMP"; do
        IFS='|' read -r abl_name abl_flags <<< "$pair"
        run_benchmark "eval_humaneval_wan.py" $DRAFT $TARGET $rtt "$abl_name" "$abl_flags"
        run_benchmark "eval_gsm8k_wan.py"     $DRAFT $TARGET $rtt "$abl_name" "$abl_flags"
        run_benchmark "eval_mgsm_wan.py"      $DRAFT $TARGET $rtt "$abl_name" "$abl_flags"
    done
done

# ============================================================
# 1.5B -> 7B: run speed test only (benchmarks too slow on CPU)
# ============================================================
DRAFT="qwen2.5-1.5b-instruct"
TARGET="qwen2.5-7b-instruct"

for rtt in "${RTTS[@]}"; do
    for pair in "full|$ABL_FULL" "no_adaptive|$ABL_NO_ADAPT" "no_fallback|$ABL_NO_FALL" "no_compression|$ABL_NO_COMP"; do
        IFS='|' read -r abl_name abl_flags <<< "$pair"
        run_speed $DRAFT $TARGET $rtt "$abl_name" "$abl_flags"
    done
done

# Generate plots
echo ""
echo ">>> Generating plots..."
python3 plot_ablation.py --exp_dir exp --output_dir exp/benchmark_ablation_plots
python3 plot_results.py --exp_dir exp --output_dir exp/benchmark_ablation_plots

echo ""
echo "=========================================="
echo "All benchmark ablation tasks completed."
echo "Plots: exp/benchmark_ablation_plots/"
echo "=========================================="
