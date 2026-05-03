#!/bin/bash
#SBATCH --job-name=humaneval_test
#SBATCH --partition=raisl
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/common/home/hd535/pearl_wan/exp/humaneval_test/slurm_%j.out

cd /common/home/hd535/pearl_wan
source /common/home/hd535/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/common/home/hd535/pearl_wan:$PYTHONPATH

# Test with limit=3 first for correctness verification
echo "=== HumanEval Quick Test (limit=3, RTT=50ms, full config, CUDA) ==="
python3 benchmark/eval_humaneval_wan.py \
    --draft_model qwen2.5-0.5b-instruct \
    --target_model qwen2.5-1.5b-instruct \
    --rtt_ms 50 \
    --limit 3 \
    --num_samples 3 \
    --max_tokens 64 \
    --eval_mode wan \
    --enable_adaptive_window \
    --enable_compression \
    --enable_fallback \
    --device_edge cuda \
    --device_cloud cuda \
    --exp_name /common/home/hd535/pearl_wan/exp/humaneval_test/quick_test_rtt50_full_correct

echo ""
echo "=== Test completed ==="
