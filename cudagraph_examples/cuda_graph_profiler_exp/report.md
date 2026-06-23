# CUDA Graph profiler 可见性实验报告

## 1. 环境信息

实验目录：`cuda_graph_profiler_exp/`

| 项 | 结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Driver | 570.144 |
| `nvidia-smi` CUDA Version | 12.8 |
| PyTorch | 2.0.0+cu118 |
| `torch.version.cuda` | 11.8 |
| CUDA 可用性 | 可用 |
| Nsight Systems | 2025.6.1.190-256136895201v0 |
| 随机种子 | 20260623 |

环境日志：

- `logs/nvidia_smi.txt`
- `logs/torch_env.txt`
- `logs/nsys_version.txt`

补充：`nsys` 运行时提示当前系统配置不允许 CPU profiling，所以 CPU IP/backtrace sampling 和 CPU context switch tracing 被禁用；CUDA trace、CUDA runtime API trace、NVTX trace 正常生成。

## 2. 方法

被测模型在 `harness.py` 中定义：

- 输入静态 buffer：`[batch=256, hidden=1024]`
- 3 个 `Linear`
- `GELU`
- `LayerNorm`
- `sin`、`tanh`、`mul`、`add` 等 elementwise
- `mean(dim=-1)` reduction

CUDA Graph 路径使用固定输入地址。`enable_graph=True` 时先在 side stream 上 warmup，再 `torch.cuda.CUDAGraph()` capture；普通 nsys 路径在 capture 后额外 replay warmup，正式测量只包住 `PLAIN_ITER_0`。

实验矩阵实际执行情况：

| # | graph | profiler 工具 | 产物 |
|---|---:|---|---|
| A | off | `torch.profiler` | `traces/run_A_torch_eager.json` |
| B | on | `torch.profiler` replay | 本机稳定超时，见 `logs/run_B_torch_profiler_timeout.log` |
| B fallback | on | `torch.profiler` capture 阶段 | `traces/run_B_fallback_capture_torch_graph.json` |
| C | off | `nsys` | `nsys/run_C_eager_stats_cuda_gpu_trace.csv`, `nsys/run_C_eager_stats_cuda_api_trace.csv` |
| D | on | `nsys --cuda-graph-trace=node` | `nsys/run_D_graph_node_stats_cuda_gpu_trace.csv`, `nsys/run_D_graph_node_stats_cuda_api_trace.csv` |
| E | on | `nsys` 默认 graph 模式 | `nsys/run_E_graph_default_stats_cuda_gpu_trace.csv`, `nsys/run_E_graph_default_stats_cuda_api_trace.csv` |

注意：`.nsys-rep` / `.sqlite` 原始文件没有保留和提交。实测 Nsight Systems 原始二进制会记录当前进程环境，并在本机包含敏感环境变量值；即使用 `env -i` 和 `--inherit-environment=false` 仍会写入。为避免泄露，只保留导出的 CUDA GPU/API CSV trace 和解析后的 JSON。

解析脚本：

- `parse_torch_trace.py`
- `parse_nsys_sqlite.py`
- `parse_nsys_window.py`

## 3. 结果

### 3.1 torch.profiler A vs B

统计窗口为 `PROFILE_ITER_0`。A 是 eager forward；B replay 在本机没有成功导出 chrome trace。

| 项 | A: eager + torch.profiler | B: graph replay + torch.profiler | B fallback: capture 阶段 |
|---|---:|---:|---:|
| chrome trace | 成功 | 失败：45s timeout，`exit_code=124` | 成功 |
| aten op 数 | 32 | N/A | 35 |
| `cudaLaunchKernel` 数 | 11 | N/A | 12 |
| `cudaGraphLaunch` 数 | 0 | N/A | 0 |
| device `kernel` 数 | 11 | N/A | 1 |
| kernel 有 correlation id | 11/11 | N/A | 1/1 |
| kernel 经 launch 归属到 aten op | 11/11 | N/A | 1/1 |

A 的 kernel 类型符合预期：CUTLASS GEMM、GELU elementwise、LayerNorm kernel、`sin/tanh/mul/add` elementwise、`reduce_kernel`。A 中 11 个 device kernel 均可通过 `correlation` 连回 `cudaLaunchKernel`，再按时间嵌套归到上层 `aten::addmm / aten::gelu / aten::native_layer_norm / aten::mean` 等 op。

B replay 触发的现象：

```text
STAGE ... Completed Stage: Warm Up
STAGE ... Completed Stage: Collection
exit_code=124
```

也就是 PyTorch 2.0.0+cu118 在本机对 `CUDAGraph.replay()` 使用 `ProfilerActivity.CUDA` 时没有正常结束并导出 trace。fallback trace 只说明 capture 期间 Python/aten 调用还存在，不代表 replay 阶段仍有逐 op host 调用。

### 3.2 nsys C/D/E

统计窗口为 NVTX `PLAIN_ITER_0`，并用 `cudaGraphLaunch` 的 correlation id 区分图内 kernel 和普通 host-launched kernel。

| 项 | C: eager | D: graph + node | E: graph 默认 |
|---|---:|---:|---:|
| 窗口时长 | 288,773 ns | 259,967 ns | 254,675 ns |
| CUDA API `cudaLaunchKernel` | 11 | 2 | 2 |
| CUDA API `cudaGraphLaunch` | 0 | 1 | 1 |
| GPU trace 总 kernel 行 | 11 | 13 | 2 |
| graph 内部 kernel 行 | 0 | 11 | 0 |
| 普通 host kernel 行 | 11 | 2 | 2 |
| graph-level trace event | N/A | N/A | 1 |

D 的 `--cuda-graph-trace=node` 展开了 graph 内部 11 个模型 kernel，包含 3 个 GEMM、GELU、LayerNorm、`sin/tanh/mul/mean/add`。D 中额外 2 个 `FillFunctor<long>` 是 `CUDAGraph.replay()` 周围的普通 host launch，不是模型 forward 图内节点。

E 默认 graph 模式下，GPU trace 窗口内只剩 2 个普通 host kernel；模型 forward 的 11 个图内 kernel 不展开，而是以 1 个 graph-level trace event 表示。结论是：Nsight Systems 默认 graph 模式把 CUDA Graph 作为整体显示；加 `--cuda-graph-trace=node` 才能看到图内 kernel 节点。

### 3.3 纯性能

CUDA event 计时，warmup 后 500 次，取中位数：

| 模式 | median | mean | p90 |
|---|---:|---:|---:|
| eager | 0.107200 ms | 0.107598 ms | 0.108224 ms |
| CUDA Graph replay | 0.096256 ms | 0.096187 ms | 0.096256 ms |

CUDA Graph replay 中位延迟降低约 10.21%，加速比约 1.114x。

## 4. 结论

开启 CUDA Graph 后，丢失的核心不是 device kernel 时间线本身，而是 replay 阶段的 host 侧逐 op 调度视图以及 `aten op -> cudaLaunchKernel -> device kernel` 的逐 op 归属链。

具体到本机实验：

- eager + `torch.profiler`：能看到完整 aten op、CUDA runtime launch、device kernel，并能用 correlation/时间嵌套恢复 op 到 kernel 的归属。
- CUDA Graph replay + `torch.profiler`：PyTorch 2.0.0+cu118 本机没有成功导出 trace，45s 超时；不能用它可靠回答 replay 内部 kernel 可见性。
- CUDA Graph capture + `torch.profiler` fallback：capture 期间仍有 Python/aten op，但这是构图阶段，不是 replay 阶段。
- CUDA Graph replay + `nsys --cuda-graph-trace=node`：能逐个看到图内 device kernel 节点，但这些 kernel 归属于一个 `cudaGraphLaunch`，不再自然归到逐个 `aten::` op。
- CUDA Graph replay + `nsys` 默认模式：图内 kernel 不展开，模型 forward 的 graph 作为整体 graph event 出现。

所以丢失层级可以概括为：

1. Host 侧 aten op 视图：replay 时基本没有逐 op Python/aten 调度。
2. op 到 kernel 的归属：replay 时图内 kernel 归到 `cudaGraphLaunch`，不再保留 eager 那种逐 op launch 链路。
3. Device kernel 时间线：没有必然丢失；`nsys --cuda-graph-trace=node` 可以展开看到图内 kernel，默认 graph 模式则折叠为整图事件。

## 5. 推荐工作流

调试算子归属：

- 先关 CUDA Graph，用 `torch.profiler` 抓 eager baseline。
- 关注 `aten::` op、`cudaLaunchKernel`、device kernel 的 correlation 和时间嵌套。
- 如果怀疑 capture 内容不同，可额外在 capture 阶段 profile，但不要把 capture trace 当 replay trace。

测真实性能或看 graph replay 的 GPU 时间线：

- 用 `nsys profile --cuda-graph-trace=node` 看图内 kernel 是否逐个执行、耗时如何。
- 用 `nsys` 默认 graph 模式看整体 graph launch 成本和真实部署形态。
- 延迟数字用 CUDA event 或 nsys 低扰动配置测，不用 `torch.profiler` 的 replay 结果做性能判断。
