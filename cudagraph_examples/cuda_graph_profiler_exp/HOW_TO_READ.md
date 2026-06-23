# 如何阅读本实验产出

建议先看 `report.md`，再按需打开 trace 和解析结果。`report.md` 是结论入口；`parsed/` 是数字来源；`traces/` 和 `nsys/` 是可视化/复核用的原始或导出 trace。

## 推荐阅读顺序

1. `report.md`
   - 先读第 4 节结论和第 5 节推荐工作流。
   - 再回到第 3 节看表格数字，确认每个结论对应哪个实验。

2. `parsed/run_A_torch_eager_summary.json`
   - 对应 A：`graph off + torch.profiler`。
   - 重点看 `aten_op_count`、`cudaLaunchKernel_count`、`device_kernel_count`、`kernel_launch_inside_aten_coverage`。
   - 这个文件证明 eager 下 `aten op -> cudaLaunchKernel -> device kernel` 链路完整。

3. `logs/run_B_torch_profiler_timeout.log`
   - 对应 B：`graph on + torch.profiler replay`。
   - 本机 PyTorch 2.0.0+cu118 下该组合超时，没能导出 replay chrome trace。
   - `traces/run_B_fallback_capture_torch_graph.json` 只是 capture 阶段 fallback，不代表 replay。

4. `parsed/run_D_nsys_window_summary.json` 和 `parsed/run_E_nsys_window_summary.json`
   - D 对应 `nsys --cuda-graph-trace=node`。
   - E 对应 `nsys` 默认 graph 模式。
   - 重点看：
     - `graph_internal_kernel_count_in_window`
     - `cudaGraphLaunch_runtime_count`
     - `graph_internal_kernel_top_names_in_window`
   - D 中 graph 内部 kernel 为 11 个；E 中 graph 内部 kernel 不展开。

5. `parsed/latency_eager.json` 和 `parsed/latency_graph.json`
   - 对应纯性能测量。
   - 重点看 `median_ms`。
   - 本次 eager median 为 0.107200 ms，graph replay median 为 0.096256 ms。

## 文件地图

| 路径 | 用途 |
|---|---|
| `harness.py` | 实验 runner：模型、eager/graph 开关、torch profiler、nsys 入口、延迟测量 |
| `parse_torch_trace.py` | 解析 PyTorch chrome trace 的 A/B 事件计数 |
| `parse_nsys_sqlite.py` | 解析 nsys SQLite 的全程表统计 |
| `parse_nsys_window.py` | 按 NVTX `PLAIN_ITER_0` 解析正式迭代窗口 |
| `report.md` | 最终实验报告 |
| `traces/` | PyTorch chrome trace JSON |
| `nsys/` | Nsight Systems 导出的 CUDA API/GPU CSV trace |
| `parsed/` | 所有机器解析后的 JSON summary |
| `logs/` | 命令运行日志和失败记录 |
| `metadata/` | 每次 torch profiler run 的参数和环境元数据 |

## A-E 对应关系

| 实验 | 关键产物 |
|---|---|
| A: eager + torch.profiler | `traces/run_A_torch_eager.json`, `parsed/run_A_torch_eager_summary.json` |
| B: graph replay + torch.profiler | `logs/run_B_torch_profiler_timeout.log` |
| B fallback: graph capture + torch.profiler | `traces/run_B_fallback_capture_torch_graph.json`, `parsed/run_B_fallback_capture_summary.json` |
| C: eager + nsys | `nsys/run_C_eager_stats_cuda_gpu_trace.csv`, `parsed/run_C_nsys_window_summary.json` |
| D: graph + nsys node | `nsys/run_D_graph_node_stats_cuda_gpu_trace.csv`, `parsed/run_D_nsys_window_summary.json` |
| E: graph + nsys default | `nsys/run_E_graph_default_stats_cuda_gpu_trace.csv`, `parsed/run_E_nsys_window_summary.json` |

## 怎么复核关键结论

### torch.profiler baseline

打开：

```bash
python cuda_graph_profiler_exp/parse_torch_trace.py \
  cuda_graph_profiler_exp/traces/run_A_torch_eager.json \
  --out /tmp/run_A_summary.json
```

应看到 A 中：

- `aten_op_count = 32`
- `cudaLaunchKernel_count = 11`
- `device_kernel_count = 11`
- `kernel_launch_inside_aten_coverage = 1.0`

### nsys node vs 默认 graph 模式

看两个文件：

```bash
cat cuda_graph_profiler_exp/parsed/run_D_nsys_window_summary.json
cat cuda_graph_profiler_exp/parsed/run_E_nsys_window_summary.json
```

关键差异：

- D: `graph_internal_kernel_count_in_window = 11`
- E: `graph_internal_kernel_count_in_window = 0`
- E: `cuda_graph_trace_events_start_in_window = 1`

这说明 `--cuda-graph-trace=node` 能展开图内 kernel；默认 graph 模式把图内 kernel 折叠为 graph-level event。

## 关于 nsys 原始文件

没有提交 `.nsys-rep` 和 `.sqlite`。原因是本机 Nsight Systems 原始二进制会记录进程环境变量，测试中确认存在敏感值泄露风险。最终保留的是脱敏后的 CSV trace 和解析 JSON，足够复核本报告中的事件计数和 kernel 展开情况。

## 最短结论

CUDA Graph replay 后，真正丢的是 host 侧逐 `aten::` op 调度视图和逐 op 到 kernel 的归属链；device kernel 时间线本身不一定丢。`nsys --cuda-graph-trace=node` 可以看到图内 kernel，默认 nsys graph 模式会把它们折叠成整图事件。
