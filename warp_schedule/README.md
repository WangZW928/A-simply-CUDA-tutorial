# CUDA Warp 数量与性能关系实验项目

这个项目用于研究 **warp 数量配置如何影响 CUDA kernel 性能**，包含：

- CUDA benchmark 程序：自动扫描不同 `warps_per_block` 与 `blocks_per_sm` 组合
- CSV 结果导出：便于后续分析
- Python 可视化：输出折线图、热力图和 Top-10 配置表

## 1. 项目结构

```text
.
├── CMakeLists.txt
├── src/
│   └── warp_bench.cu
├── scripts/
│   └── plot_results.py
├── results/
├── requirements.txt
└── README.md
```

## 2. 编译与运行（Linux/WSL）

### 2.1 编译

```bash
cmake -S . -B build
cmake --build build -j
```

生成可执行文件：`build/warp_bench`

### 2.2 运行 benchmark

```bash
./build/warp_bench
```

也可以自定义参数：

```bash
./build/warp_bench <output_csv> <repeats> <warmup> <iters>
# 示例：
./build/warp_bench results/warp_benchmark.csv 30 8 4096
```

参数说明：

- `output_csv`：输出结果 CSV 路径（默认 `results/warp_benchmark.csv`）
- `repeats`：正式计时重复次数（默认 20）
- `warmup`：预热次数（默认 5）
- `iters`：kernel 内部循环次数（默认 2048）

## 3. Python 可视化

安装依赖：

```bash
python -m pip install -r requirements.txt
```

绘图：

```bash
python scripts/plot_results.py --input results/warp_benchmark.csv --outdir results
```

会生成：

- `results/runtime_vs_warps.png`：不同 `blocks_per_sm` 下，`warps_per_block` 对运行时间影响
- `results/throughput_vs_warps.png`：不同 `blocks_per_sm` 下，`warps_per_block` 对吞吐量影响
- `results/normalized_throughput_vs_warps.png`：归一化吞吐量折线图（相对最优配置=100%）
- `results/throughput_heatmap.png`：吞吐指标（`total_warps / ms`）热力图
- `results/top10_configs.csv`：吞吐最高的 10 组配置

## 4. 如何理解结果

建议重点观察：

- 当 `warps_per_block` 很小时，GPU 可能无法充分隐藏延迟
- 当 `warps_per_block` 太大时，寄存器/调度资源压力上升，未必继续提升性能
- 不同 `blocks_per_sm` 会改变并发 block 数量，进而影响 occupancy 和吞吐

你可以先用默认参数跑一版，再调大 `iters` 或 `repeats` 来减少抖动，得到更稳定曲线。


## 5. CUDA 执行模型速记

下面这部分把本项目里出现的 `thread/block/warp/SM` 关系一次对齐。

### 5.1 你写的 kernel 语义是“每线程做什么”

典型写法：

```cpp
__global__ void saxpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
```

关键理解：这段代码描述的是“每个线程的行为”，不是“一个 CPU 核的行为”。

- Grid：一次 kernel launch 的全部工作
- Block（CTA）：调度和资源分配单位（shared memory、`__syncthreads()` 以 block 为边界）
- Thread：执行函数体的最小语义单位

### 5.2 硬件执行时会把线程打包成 warp

在 NVIDIA GPU 上，通常 32 个线程组成一个 warp，warp 是硬件发射指令的基本单位（SIMT）。

例如 `blockDim.x = 256`：

- Warp0: thread 0..31
- Warp1: thread 32..63
- ...
- Warp7: thread 224..255

也就是说，一个 256-thread block 会被拆成 8 个 warp。

### 5.3 Block 分配到 SM，且不会跨 SM

调度器按 block 分配到 SM：

- 一个 block 内线程总在同一个 SM 上
- 这样它们才能共享 shared memory，并进行 block 内同步
- 一个 SM 可同时驻留多少 block，取决于寄存器、shared memory、线程/warp 上限等资源约束（occupancy）

### 5.4 在 SM 内，warp 才是发射单位

SM 内通常驻留多个 block、多个 warp。warp scheduler 每周期选择可运行 warp 发射下一条指令，用于隐藏访存或流水线延迟。

- 语义层看：很多线程在“并行跑”
- 硬件层看：以 warp 为粒度交错发射执行

### 5.5 SIMT 与分支发散

同一个 warp 发射一条指令时，32 个 lane 执行同一指令、处理不同数据（各自寄存器和索引不同）。

当 warp 内线程走不同分支（例如部分线程满足 `i < n`，部分不满足）会发生 divergence：

- 硬件会用 active mask 分别执行各分支路径
- 结果正确，但 warp 内执行可能串行化，吞吐下降

所以 divergence 不是错误，而是性能风险点。

### 5.6 编译与执行链路

```text
CUDA C++ kernel
   -> PTX（中间表示）
   -> SASS（目标架构机器码，JIT 或离线生成）
   -> SM 执行（warp 粒度发射）
```

`blockIdx/threadIdx` 等内建变量可理解为运行时提供给每线程上下文的硬件索引信息，用于计算全局索引。

### 5.7 三个最重要的“对上号”

- 语义：你写的是 thread 程序（每线程逻辑）
- 执行：硬件按 warp 发射（32 线程同指令）
- 调度：资源按 block 分配（同步和 shared memory 边界）

### 5.8 一张总图（Grid -> Block -> Warp -> SM）

```text
你写 kernel（每线程语义）
        |
        v
Launch: <<<gridDim, blockDim>>>
        |
        v
Grid（很多 Blocks）
+---------------------------------------+
| Block0  Block1  Block2  ...  BlockK   |
+---------------------------------------+
        |
        |  Block 作为整体分配到 SM（不会跨 SM）
        v
GPU: 多个 SM 并行承接 Blocks
+--------+--------+--------+--------+
|  SM0   |  SM1   |  SM2   |  SM3   | ...
+--------+--------+--------+--------+
        |
        |  每个 Block 内线程按 32 打包成 warp
        v
SM 内 resident warps + warp scheduler
        |
        |  以 warp 为粒度发射同一条指令（SIMT）
        v
32 lanes 同指令、不同数据；分支靠 active mask 处理
```

## 6. 实验结论与注意事项

- 吞吐量（例如本项目的 `total_warps / ms`）可以作为 CUDA 程序是否较好利用 GPU 的一个核心指标，但它不是唯一指标。还需要结合 kernel 的性质（compute-bound 还是 memory-bound）、功耗、访存效率等一起判断。
- 在许多场景下，让每个 SM 的活跃 warp 数量略高于“隐藏延迟所需的最小阈值”，往往就能逼近峰值性能区间。继续增加 warp 数并不一定更快，可能会因为寄存器、shared memory、调度开销等资源限制而收益递减甚至下降。
- 严格说“达到理论计算巅峰性能”通常需要同时满足多个条件：高指令吞吐、足够并行度、低访存瓶颈、较少 stall。仅仅提高 warp 数一般不足以保证达到理论峰值。

## 7. 可扩展方向

- 把 kernel 改成 memory-bound（加入更多全局内存访问）
- 使用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 估计理论 occupancy
- 同时记录 `nsight compute` 指标（例如 achieved occupancy、warp stall 原因）




