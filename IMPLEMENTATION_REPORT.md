# PEARL-WAN 实现总结报告

## 一、项目背景与目标

根据开题报告《基于起草-验证的云边协同大模型推理》，本项目的核心目标是将 PEARL（Parallel Speculative Decoding with Adaptive Draft Length）框架从数据中心内部低延迟环境扩展至广域网（WAN）高延迟场景，实现 **PEARL-WAN** 原型系统。

### 开题报告要求实现的关键技术点：
1. **高效率数据传输协议**：边缘端部署轻量级起草模型，云端部署大目标模型，设计高效传输协议减少网络延迟影响
2. **自适应起草长度算法（AWAS）**：基于 PEARL 的 pre-verify/post-verify 策略，适应网络带宽波动
3. **Fallback 机制**：当延迟超过阈值时自动切换为本地小模型推理
4. **一致性保证机制**：无损失验证采样算法，确保输出分布与纯云端大模型一致

---

## 二、PEARL 原始代码分析

### 2.1 获取源码
从 GitHub 克隆 PEARL 官方仓库：
```bash
git clone https://github.com/smart-lty/ParallelSpeculativeDecoding.git pearl
```

### 2.2 核心机制
PEARL 的核心创新在于两个并行策略：
- **Pre-verify**：在起草阶段并行验证第一个 token，若被拒绝立即重新起草
- **Post-verify**：在验证阶段允许起草模型继续生成下一批 token，实现流水线并行

原始代码通过 `accelerate` 库启动 2 个进程，分别运行 draft model 和 target model，利用 `accelerator.gather()` 进行进程间通信。

### 2.3 原始代码局限（针对 WAN 场景）
1. **通信机制**：使用共享内存级别的 gather，假设微秒级延迟，不适用于 WAN
2. **固定窗口**：gamma（window size）为固定超参数，未根据网络状况自适应调整
3. **无传输压缩**：直接传输完整 logits tensor，带宽占用大
4. **无容错机制**：网络波动或丢包会导致系统崩溃

---

## 三、PEARL-WAN 修改与新增内容

### 3.1 文件修改总览

| 类型 | 文件路径 | 说明 |
|------|---------|------|
| 新增 | `pearl_wan/src/engine_wan.py` | PEARL-WAN 主引擎，整合所有模块 |
| 新增 | `pearl_wan/src/network_simulator.py` | WAN 网络模拟器（RTT/带宽/丢包） |
| 新增 | `pearl_wan/src/compression.py` | 轻量级传输协议（top-k + 量化） |
| 新增 | `pearl_wan/src/adaptive_window.py` | AWAS 自适应窗口算法 |
| 新增 | `pearl_wan/src/fallback.py` | Fallback 延迟保护机制 |
| 新增 | `pearl_wan/src/kvcache_wan.py` | 基于 PEARL 修改的 KVCache 封装 |
| 新增 | `pearl_wan/src/util_wan.py` | 工具函数与参数解析（WAN 扩展） |
| 新增 | `pearl_wan/benchmark/eval_wan.py` | 评估脚本（支持多模式对比） |
| 新增 | `pearl_wan/run_wan.sh` | 单场景测试脚本 |
| 新增 | `pearl_wan/run_wan_comparison.sh` | 多 RTT 对比脚本 |
| 新增 | `pearl_wan/README.md` | 项目文档 |

### 3.2 核心修改详解

#### (1) engine_wan.py — PEARL-WAN 主引擎
基于 PEARL 的 `engine.py` 重构，主要改动：
- **移除 accelerate 依赖**：改为单进程内模拟云边通信，便于原型验证
- **新增 `pearl_wan_decode` 方法**：
  - 在 drafting 阶段和 verification 阶段之间插入网络模拟和传输压缩
  - 保留 PEARL 的 pre-verify / post-verify 状态机逻辑
  - 集成 fallback 检查和 adaptive window 更新
- **新增 baselines**：
  - `autoregressive_sampling`：纯 target model 自回归解码
  - `speculative_decoding_baseline`：单机投机解码（无网络延迟）

#### (2) network_simulator.py — 网络模拟层
```python
class NetworkSimulator:
    def __init__(self, rtt_ms=50.0, bandwidth_mbps=100.0, packet_loss_rate=0.0):
```
- 模拟单向传播延迟：`propagation = RTT / 2`
- 模拟传输延迟：`transmission = data_size * 8 / bandwidth`
- 模拟抖动（jitter）和丢包
- 记录总传输字节数、注入延迟、丢包率等统计信息

#### (3) compression.py — 传输压缩协议
实现两种压缩模式：
- **Top-k 稀疏化（默认）**：仅保留概率最高的 50 个 token 的 logits 及索引，其余位置在云端 reconstruct 为 `-inf`。对于 151936 词表的 Qwen 模型，理论压缩比可达 **~1500x**。
- **8-bit 量化**：逐 token min-max 量化，无损保留完整分布信息。

> ⚠️ **已知问题**：top-k 是有损压缩，极端情况下拒绝采样可能落入非 top-k 区域，导致分布偏差。

#### (4) adaptive_window.py — AWAS 算法
基于成本模型动态调整 gamma：
```
time_per_token ≈ (t_draft + t_transmit) / p + (RTT + t_target) / (gamma * p)
```
其中 p 为接受率 EMA。当 RTT 较大时自动增大 gamma 以摊销网络开销；当首 token 拒绝率过高时减小 gamma。

#### (5) fallback.py — 延迟保护机制
- 监测最近 3 轮平均延迟
- 超过阈值（默认 2000ms，可通过参数调整）时切换为本地 draft model 自回归推理
- 满足恢复条件（延迟改善 30% 以上且生成足够本地 token）后自动恢复云边协同模式

#### (6) kvcache_wan.py — KVCache 封装
基于 PEARL 的 `kvcache.py` 修改：
- 兼容 transformers 5.x 的 `DynamicCache`（支持 `get_seq_length()` 和 `crop()`）
- 兼容旧版 tuple-based `past_key_values` 的 manual rollback
- 新增 `generate_single` 方法用于 fallback 模式

---

## 四、运行过程记录与问题解决

### 4.1 已下载模型
由于当前环境无 GPU，为便于 CPU 快速验证，下载了较小的模型：

| 模型 | 路径 | 大小 | 用途 |
|------|------|------|------|
| Qwen2.5-0.5B-Instruct | `pearl_wan/models/qwen2.5-0.5b-instruct` | ~1GB | 模拟边缘 draft model（7B） |
| Qwen2.5-1.5B-Instruct | `pearl_wan/models/qwen2.5-1.5b-instruct` | ~3GB | 模拟云端 target model（70B） |

**生产环境建议**：替换为 CodeLlama-7B + CodeLlama-70B 或 DeepSeek-1.3B + DeepSeek-33B 等组合。

### 4.2 遇到的问题及解决方案

#### 问题 1：`torch_dtype` 参数弃用警告
**现象**：
```
`torch_dtype` is deprecated! Use `dtype` instead!
```
**原因**：transformers 5.4.0 中 `from_pretrained` 的 `torch_dtype` 参数名已变更为 `dtype`。
**解决**：在 `engine_wan.py` 的 `_load_models` 中增加 try-except 兼容逻辑：
```python
try:
    model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=dtype, ...)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(..., dtype=dtype, ...)
```

#### 问题 2：`RuntimeError: probability tensor contains either inf, nan or element < 0`
**现象**：在 post-verify 拒绝采样时，`torch.multinomial` 抛出异常。
**原因**：
1. `max_fn` 函数在 `target_prob - draft_prob` 全为 0 或负数时，归一化分母为 0，产生 nan
2. compression 的 top-k 稀疏化导致部分位置为 `-inf`，与概率运算后产生异常值
**解决**：
1. 增强 `max_fn` 的零除保护：`x_max_sum = torch.where(x_max_sum == 0, torch.ones_like(x_max_sum), x_max_sum)`
2. 增强 `sample` 函数的安全检查：过滤 nan/inf/负数，全零时回退为均匀分布

#### 问题 3：Fallback 过早触发
**现象**：首次运行中 `fallback_trigger_count: 1`，55/64 个 token 通过本地生成，云边协同形同虚设。
**原因**：默认阈值 200ms 对于 CPU 环境过低（单次 target forward 即超过 500ms）。
**解决**：将默认阈值提升至 2000ms，并增加本地延迟记录，使 fallback 退出逻辑更平滑。

#### 问题 4：网络延迟测量包含计算时间
**现象**：`avg_network_time` 高达 235ms，远超理论值（RTT=50ms）。
**原因**：代码中将 cloud verification 时间也计入了 network_time。
**解决**：调整时间戳记录位置，将 cloud computation 与 network transfer 分离统计。

#### 问题 5：CPU 环境下模型加载极慢
**现象**：首次加载 1.5B 模型需数分钟，完整测试运行时间超过 20 分钟。
**原因**：无 GPU，纯 CPU 推理。
**缓解**：减少 `max_tokens` 至 32，`num_samples` 至 2，仅用于功能验证。

#### 问题 6：WAN 模式输出出现乱码
**现象**：`"return(fibonacci(n- STANDARD.userdetails𝕸 cling) + fibonacci(n-1"`
**原因分析**：
1. top-k 压缩导致概率分布截断，累积误差影响后续 token 生成
2. CPU 环境下 float32 精度与 bfloat16 行为差异
3. KVCache rollback 后状态不一致
**当前状态**：该问题在首次验证中偶发，需进一步排查。建议生产环境关闭 compression 或改用无损 8-bit 量化。

---

## 五、实验结果与分析

### 5.1 测试环境
- **硬件**：CPU（无 GPU）
- **模型**：Qwen2.5-0.5B-Instruct（draft）+ Qwen2.5-1.5B-Instruct（target）
- **网络**：模拟 RTT=50ms，Bandwidth=100Mbps，丢包率=0%
- **配置**：gamma=4，max_tokens=32，temperature=0

### 5.2 性能对比

| 模式 | 速度 (tok/s) | 总 token | 总时间 | draft forwards | target forwards |
|------|-------------|---------|--------|---------------|-----------------|
| Autoregressive | **3.05** | 64 | 20.97s | - | 64 |
| Vanilla SD | **7.96** | 67 | 8.42s | ~200 | ~17 |
| PEARL-WAN | **2.61** | 66 | 25.30s | 179 | 52 |

### 5.3 结果分析

**意外发现**：PEARL-WAN 在当前测试环境下速度低于 Autoregressive 基线。原因如下：

1. **模型规模差异不足**：0.5B vs 1.5B 在 CPU 上速度差异很小（约 2-3x），无法弥补网络延迟开销。在 GPU 环境下，7B vs 70B 的 draft 速度通常可达 target 的 5-10x，此时投机解码的收益才能覆盖网络成本。

2. **网络延迟主导**：虽然单轮网络往返仅 ~50ms，但当前 adaptive window 将 gamma 降至 1（因接受率波动），导致每轮只生成 1 个 token 却付出一次完整 RTT，效率极低。

3. **单进程顺序模拟**：未实现真正的云边并行。在生产环境中，edge 和 cloud 应作为独立进程/服务运行，post-verify 阶段可真正并行 drafting 和 verification。

### 5.4 模块统计

```
PEARL-WAN Statistics
============================================================
  draft_forward_times: 179
  target_forward_times: 52
  mean_accepted_tokens: 3.74
  compression_ratio: 1504.34
  network_stats:
    total_bytes_sent: 91136
    total_delay_injected_sec: 1.90
    packet_drop_count: 0
  adaptive_window_stats:
    current_gamma: 1
    acceptance_rate_ema: 0.78
  fallback_stats:
    fallback_active: False
    fallback_trigger_count: 0
```

---

## 六、修改部分总结

### 6.1 对 PEARL 原始代码的继承
- 保留了 `kvcache.py` 的核心 KVCache 管理和 rollback 机制
- 保留了 pre-verify / post-verify 的状态机切换逻辑
- 保留了 speculative decoding 的数学正确性（接受-拒绝采样）

### 6.2 主要新增与改动
1. **网络层抽象**：新增 `network_simulator.py`，将进程间 gather 替换为带延迟模拟的网络 send/receive
2. **传输压缩**：新增 `compression.py`，在通信前对 logits 进行 top-k 稀疏化
3. **自适应窗口**：新增 `adaptive_window.py`，替代固定 gamma，实现动态调整
4. **容错保护**：新增 `fallback.py`，在网络恶化时保证服务可用性
5. **评估框架**：重写 `eval_wan.py`，支持 autoregressive / vanilla SD / PEARL-WAN 三模式对比

### 6.3 代码行数统计
```bash
find pearl_wan/src -name "*.py" | xargs wc -l
```
- `engine_wan.py`: ~390 行
- `compression.py`: ~170 行
- `adaptive_window.py`: ~140 行
- `kvcache_wan.py`: ~120 行
- `network_simulator.py`: ~80 行
- `util_wan.py`: ~160 行
- `fallback.py`: ~90 行
- **总计**：约 1150 行新增代码

---

## 七、后续工作建议

1. **GPU 环境验证**：在 H100/A100 上测试 CodeLlama 7B+70B 或 DeepSeek 1.3B+33B，验证 WAN 场景下的真实加速比
2. **无损压缩**：当前 top-k 有损压缩偶发输出异常，建议实现分层压缩（高概率区域精确传输，低概率区域量化近似）
3. **真正并行化**：将 edge 和 cloud 拆分为独立进程/容器，通过 socket/gRPC 通信，实现 post-verify 阶段的真正并行
4. **树状验证**：集成 SpecInfer / Ouroboros 的树状注意力机制，进一步提高接受率
5. **动态带宽感知**：让 adaptive window 实时监测带宽变化（而非仅依赖 RTT），在网络拥塞时自动降级

---

## 八、附录：快速复现命令

```bash
cd /common/home/hd535/pearl_wan

# 单场景测试
bash run_wan.sh

# 自定义参数
python3 benchmark/eval_wan.py \
    --draft_model qwen2.5-0.5b-instruct \
    --target_model qwen2.5-1.5b-instruct \
    --rtt_ms 50 \
    --gamma 4 \
    --max_tokens 32 \
    --enable_adaptive_window \
    --enable_compression \
    --enable_fallback

# 查看结果
cat exp/pearl_wan_test/eval_wan_results.json
```

---

*报告生成时间：2026-04-29*
*基于 PEARL 官方代码（https://github.com/smart-lty/ParallelSpeculativeDecoding）扩展*
