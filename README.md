# PEARL-WAN: Cloud-Edge Collaborative Speculative Decoding for Wide Area Networks

基于 [PEARL](https://github.com/smart-lty/ParallelSpeculativeDecoding) 框架扩展的面向广域网高延迟环境的云边协同投机解码原型系统。

## 项目概述

本项目实现开题报告《基于起草-验证的云边协同大模型推理》中提出的 **PEARL-WAN** 原型系统，将 PEARL 的并行投机解码理论扩展至广域网（WAN）场景，主要创新点包括：

1. **自适应窗口选择算法 (AWAS)**：根据网络状况（RTT、带宽）和模型置信度动态调整草稿长度 γ
2. **轻量级传输协议**：对 token logits 进行 top-k 稀疏化与 8-bit 量化，压缩传输数据量
3. **Fallback 机制**：当网络延迟超过阈值时自动切换为本地 draft model 推理
4. **网络模拟器**：可配置 RTT、带宽、丢包率，模拟真实 WAN 环境

## 系统架构

### 单机模拟模式

```
┌─────────────────┐         Simulated WAN                ┌─────────────────┐
│   Edge Node     │ ◄────────────────────────────────► │   Cloud Node    │
│  ┌───────────┐  │                                     │  ┌───────────┐  │
│  │ Draft     │  │   Compressed draft tokens + logits  │  │ Target    │  │
│  │ Model     │  │ ──────────────────────────────────► │  │ Model     │  │
│  │ (0.5B/7B) │  │                                     │  │ (1.5B/70B)│  │
│  └───────────┘  │   Verification result metadata      │  └───────────┘  │
│  ┌───────────┐  │ ◄─────────────────────────────────  │  ┌───────────┐  │
│  │ Adaptive  │  │                                     │  │ Verify    │  │
│  │ Window    │  │                                     │  │ Engine    │  │
│  └───────────┘  │                                     │  └───────────┘  │
│  ┌───────────┐  │                                     │                 │
│  │ Fallback  │  │                                     │                 │
│  │ Manager   │  │                                     │                 │
│  └───────────┘  │                                     │                 │
└─────────────────┘                                     └─────────────────┘
```
```
pearl_wan/
├── src/
│   ├── engine_wan.py              # 单机模拟主引擎
│   ├── engine_distributed.py      # 分布式跨节点引擎（torch.distributed）
│   ├── comm_dist.py               # 分布式通信模块（send/recv draft & verify）
│   ├── kvcache_wan.py             # KVCache 封装
│   ├── util_wan.py                # 工具函数与参数解析
│   ├── network_simulator.py       # WAN 网络模拟
│   ├── compression.py             # 传输压缩协议
│   ├── adaptive_window.py         # AWAS 自适应窗口算法
│   └── fallback.py                # Fallback 机制
├── benchmark/
│   ├── eval_wan.py                # 单机三模式对比脚本
│   ├── eval_humaneval_wan.py      # 单机 HumanEval
│   ├── eval_gsm8k_wan.py          # 单机 GSM8K
│   ├── eval_mgsm_wan.py           # 单机 MGSM
├── plot_ablation.py               # 消融实验
├── plot_results.py                # 吞吐结果
├── run_benchmark_ablation.slurm          # 消融实验
└── README.md
```

## 环境依赖

- Python 3.8+
- PyTorch 2.x
- Transformers 4.5+
- Accelerate

## 快速开始

### 1. 运行单场景测试

```bash
cd pearl_wan
bash run_wan.sh
```

默认配置：
- Draft Model: Qwen2.5-0.5B-Instruct (模拟边缘 7B)
- Target Model: Qwen2.5-1.5B-Instruct (模拟云端 70B)
- RTT: 50ms, Bandwidth: 100Mbps
- Adaptive Window: 开启
- Compression: 开启
- Fallback: 开启

### 2. 自定义参数

```bash
python3 benchmark/eval_wan.py \
    --draft_model qwen2.5-0.5b-instruct \
    --target_model qwen2.5-1.5b-instruct \
    --rtt_ms 50 \
    --bandwidth_mbps 100 \
    --gamma 4 \
    --max_tokens 64 \
    --enable_adaptive_window \
    --enable_compression \
    --enable_fallback
```

## 核心算法说明

### Pre-verify / Post-verify 策略

- **Pre-verify**: 在起草阶段并行验证第一个 token。若被拒绝，立即重新起草，避免传输无效候选。
- **Post-verify**: 在验证阶段允许 draft model 继续生成下一批 token，实现流水线并行。

### AWAS 自适应窗口算法

基于成本模型动态选择最优 γ：

```
time_per_token = (gamma * t_draft + RTT + gamma * t_transmit + t_target) / (gamma * p)
```

其中 p 为接受率 EMA。当 RTT 较大时自动增大 γ 以摊销网络开销。

### 传输压缩

采用 **top-k 稀疏化 + 8-bit 量化**：
- 仅传输概率最高的 50 个 token 的 logits 及索引
- 其余位置在云端 reconstruct 为 `-inf`
- 对于 151936 的词表，压缩比可达 **~1500x**

> 注：top-k 压缩是有损近似，在严格数学保证场景下可切换为无损 8-bit 量化模式。

## 实验结果示例

在 CPU 环境（Qwen2.5-0.5B + Qwen2.5-1.5B）的初步测试结果：

| 模式 | 速度 (tok/s) | 说明 |
|------|-------------|------|
| Autoregressive | ~3.0 | 纯云端 target model 自回归解码 |
| Vanilla SD | ~7.9 | 单机投机解码（无网络延迟） |
| PEARL-WAN (RTT=50ms) | ~2.6 | 云边协同（含网络模拟） |

**分析**：在 CPU 小模型场景下，draft 与 target 速度差异小，加上网络延迟后 PEARL-WAN 优势不明显。预期在 GPU 大模型场景（7B draft + 70B target）中，draft 速度远快于 target，网络开销可被更大的 γ 有效摊销，从而获得显著加速。

## 已知问题与限制

1. **CPU 性能瓶颈**：当前在无 GPU 环境测试，大模型推理极慢，建议使用 GPU 环境运行 7B+70B 组合。
2. **Top-k 压缩有损**：极端情况下拒绝采样可能落入非 top-k 区域，导致分布偏差。可关闭 `--enable_compression` 使用无损模式。
3. **单进程模拟**：当前云边协同为单进程顺序模拟，未实现真正的跨设备并行。生产环境需拆分为独立 edge/cloud 服务进程。
4. **`torch_dtype` 警告**：transformers 5.4.0 中 `torch_dtype` 参数名已弃用，代码中已做兼容处理。

## 参考文献

- Liu, T., et al. (2025). PEARL: Parallel Speculative Decoding with Adaptive Draft Length. ICLR 2025.
- Leviathan, Y., et al. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023.

## 致谢

本项目基于 [PEARL](https://github.com/smart-lty/ParallelSpeculativeDecoding) 官方代码修改扩展，保留原始 PEARL 的验证逻辑与 KVCache 机制。
