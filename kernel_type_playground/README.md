# Kernel Type Playground：CUDA Kernel 瓶颈类型实验

这个项目用四类小 kernel 建立一个重要习惯：

先判断瓶颈类型，再选择优化方向。

同样是“慢”，compute-bound、memory-bound、latency-bound、launch-overhead-bound 的解决方式完全不同。

## 1. 四类 kernel

源码：`src/kernel_types.cu`

### 1.1 compute_bound_kernel

特点：

- 每个元素只读写少量 global memory
- 在寄存器里做大量 FMA
- 主要压力在 FP32 算术管线和指令调度

典型优化方向：

- 提高指令级并行
- 减少不必要的依赖链
- 检查寄存器压力是否过高
- 用 Nsight Compute 看 compute utilization 和 math pipe stall

### 1.2 memory_bound_kernel

特点：

- stream triad：`c[i] = a[i] + alpha * b[i]`
- 每个元素算术很少，global memory 读写占主导
- 主要瓶颈是内存带宽和 coalescing

典型优化方向：

- 保持连续合并访问
- 减少读写字节数
- 增加数据复用
- 使用 shared memory 或 cache-friendly layout，但前提是有复用

### 1.3 latency_bound_kernel

特点：

- 使用 dependent pointer chasing
- 下一次访问地址依赖上一次 load 的结果
- 很难通过单 warp 内并行隐藏延迟

典型优化方向：

- 增加独立 work / 更多 warp 来隐藏延迟
- 改数据结构，减少 pointer chasing
- 批处理多个独立链
- 改善 locality

### 1.4 launch_overhead_kernel

特点：

- kernel 本身几乎不做事
- 重复 launch 很多次
- CPU 到 GPU 的提交开销占主导

典型优化方向：

- CUDA Graph
- kernel fusion
- 批处理小任务
- 减少 CPU/GPU 往返同步

## 2. 本项目比较哪些敏感性

CSV 中的 `experiment` 包含：

- `block_size_sweep`：比较 block size 对三类主 kernel 的影响
- `occupancy_sweep`：固定 block size，改变每个 SM 的 block 数量
- `stream_compare`：比较 memory-bound 两个 chunk 串行 launch 和 two streams
- `graph_compare`：比较大量 tiny kernel 的普通 launch 和 CUDA Graph replay

## 3. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/kernel_types
```

可选参数：

```bash
./build/kernel_types <output_csv> <repeats> <warmup> <elements> <compute_iters> <latency_steps> <tiny_launches>
# 示例
./build/kernel_types results/kernel_type_benchmark.csv 25 5 4194304 1024 512 1000
```

注意：`elements` 会向上调整到 2 的幂，方便 latency pointer chasing 使用位掩码。

## 4. 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/kernel_type_benchmark.csv --outdir results
```

输出：

- `results/block_size_sweep.png`
- `results/occupancy_sweep.png`
- `results/stream_compare.png`
- `results/graph_compare.png`
- `results/summary.csv`

## 5. 字段说明

- `kernel_type`：四类瓶颈标签
- `mode`：当前比较模式
- `block_size` / `blocks`：launch 配置
- `iters_or_launches`：compute/latency 的循环次数，或 tiny kernel 的 launch 数
- `streams`：使用 stream 数
- `mean_ms` / `std_ms`：CUDA event 计时
- `throughput_units_per_ms`：按实验定义的工作单位吞吐
- `effective_gb_s`：主要用于 memory-bound 的字节吞吐估算
- `theoretical_occupancy`：由 CUDA occupancy API 估算的理论占用率

## 6. 怎么读结果

### 6.1 block size sweep

如果 block size 改变后：

- compute-bound 变化不大：说明算术管线或寄存器依赖更关键
- memory-bound 变化不大：说明已经主要受带宽限制
- latency-bound 对 block/warp 数敏感：说明 latency hiding 起作用

### 6.2 occupancy sweep

occupancy 提升不一定带来线性加速。更好的判断是：

- 延迟型 kernel 是否随更多 blocks/SM 改善
- compute-bound 是否因资源压力进入平台期
- memory-bound 是否很快触达带宽上限

### 6.3 stream compare

Two streams 不保证更快。只有当两个 chunk 有足够独立性、GPU 有可并发空间、并且同步没有提前阻塞时，才可能收益。

### 6.4 graph compare

tiny kernel 的 graph replay 通常更有机会赢，因为优化点是 launch overhead，而不是 kernel 内部计算。

## 7. 和 Nsight 工具配合

建议顺序：

1. 先跑 CSV，判断哪类 kernel 对哪个参数敏感
2. 用 Nsight Systems 看 tiny launches 和 graph replay 的时间线
3. 用 Nsight Compute 分析 compute/memory/latency kernel 的指标
4. 用 roofline 判断 compute-bound 和 memory-bound 是否符合预期

## 8. 你应该学到什么

- 优化不是套公式；先分类，后下手
- memory-bound 的重点是字节和访问模式
- compute-bound 的重点是指令吞吐、依赖和寄存器
- latency-bound 的重点是隐藏延迟或改变数据结构
- launch-overhead-bound 的重点是减少 launch 次数或使用 graph

一句话记忆：

Kernel 类型决定优化方向；不知道瓶颈类型时，优化很容易变成猜。
