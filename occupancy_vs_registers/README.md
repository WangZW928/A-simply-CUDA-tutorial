# occupancy_vs_registers

## 动机/核心观点

这个小项目用一个自包含 CUDA benchmark 观察 **Register Pressure ↔ Occupancy** 的关系。

核心观点：

- 每个线程使用的寄存器越多，同一个 SM 能同时驻留的 block/warp 通常越少。
- 这个限制不是平滑下降，而是由资源上限和分配粒度共同造成的 **阶梯曲线**。
- Occupancy 下降不一定马上意味着性能下降；更多寄存器可能减少重复计算或访存，但也可能跨过某个 **occupancy cliff** 后让延迟隐藏能力明显变差。
- 因此本实验同时画出理论模型和实测数据，避免只看单一指标。

## 硬件参数表

本项目按这台机器的 RTX 4060 Laptop GPU 建模：

| 参数 | 数值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 架构 | Ada / AD107 |
| Compute Capability | `sm_89` |
| SM 数量 | 24 |
| 每 SM 32-bit register 数 | 65536 |
| 每 SM 最大 warp 数 | 48 |
| 每 SM 最大线程数 | 1536 |
| 每 block 最大线程数 | 1024 |
| 每 SM 最大 resident block 数 | 24 |
| Warp size | 32 |
| Register allocation granularity | 256 regs/warp，也就是 8 regs/thread 粒度 |

## 实验设计

`src/occupancy_bench.cu` 做两组实验。

第一组是主实验 `sweep_kernel<REG_TMP_SIZE>`：

- `REG_TMP_SIZE` 取 `{0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256}`。
- 每个 `REG_TMP_SIZE` 都跑 `{128, 256, 512, 1024}` 四种 block size。
- kernel 内部用 `volatile float tmp[REG_TMP_SIZE]` 和 unrolled loop 制造可控的 per-thread register pressure。
- 用 `cudaFuncGetAttributes(...).numRegs` 记录 nvcc/ptxas 真实生成的 `regs_per_thread`。
- 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 记录 CUDA runtime API 给出的理论 active blocks/SM 和 occupancy。
- 用 warmup + repeated CUDA event timing 测平均 runtime，并计算 throughput。

第二组是小型 `__launch_bounds__(256, N)` 实验：

- `N` 取 `{1, 2, 4, 6}`。
- 目的不是替代主实验，而是展示“用 launch bounds 约束寄存器，换取 occupancy”的典型 tradeoff。
- 注意：`N × 256` 不能超过每 SM 最大线程数 1536，否则 ptxas 会报 `minnctapersm is out of range` 并**直接忽略该提示**（N≥8 时实测 regs 不再下降）。
- 该内核故意用 48 个活跃浮点制造高寄存器需求；实测中 N=1/2/4 时 regs≈62、occupancy≈67%，而 N=6 迫使编译器把寄存器压到 40 并产生约 172 字节/线程的溢出（local memory），occupancy 虽到 100%，runtime 反而慢了约 7 倍——这是“追 occupancy 不如防溢出”的典型例子。

`scripts/plot_analysis.py` 会独立计算经典理论阶梯：

```text
effective_regs = ceil(regs * 32 / 256) * 256 / 32
blocks_by_regs = floor(65536 / (effective_regs * threads_per_block))
blocks_by_warps = floor(48 * 32 / threads_per_block)
blocks_by_threads = floor(1536 / threads_per_block)
blocks_by_blocks = 24
active_blocks = min(...)
occupancy = active_blocks * threads_per_block / 1536
```

注意：必须取所有资源限制的最小值。例如 `256 threads/block, 24 regs/thread` 时，寄存器单独看似能放 10 个 block，但线程/warp 上限只允许 6 个 block，所以 occupancy 仍是 100%。

另外，当 `regs_per_thread × threads_per_block > 65536` 时（例如 70 寄存器 × 1024 线程），该 launch 配置**物理上无法启动**，CUDA 会报 `too many resources requested for launch`。benchmark 会检测到 occupancy API 返回 0 个 active block，自动跳过这类组合（日志中打印 `[skip]`），不会崩溃。

## 构建/运行/绘图命令

建议在项目目录内执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
cmake --build build -j
```

运行 benchmark：

```bash
mkdir -p results
./build/occupancy_bench results/occupancy_sweep.csv
```

也可以指定 repeat 和 warmup 次数：

```bash
./build/occupancy_bench results/occupancy_sweep.csv 20 5
```

绘图：

```bash
python3 -m pip install -r requirements.txt
python3 scripts/plot_analysis.py --input results/occupancy_sweep.csv --outdir results
```

输出文件：

- `results/occupancy_analysis.png`
- `results/occupancy_analysis.svg`
- `results/occupancy_summary.csv`

## 图表解读

最终图是一个多面板图：

- 左上：理论 occupancy 阶梯曲线，并叠加 benchmark 中由 CUDA occupancy API 得到的点。灰色阴影区域表示 occupancy cliff，即增加少量寄存器就让 active block/warp 数下降的位置。
- 右上：实测 runtime 随 `regs_per_thread` 的变化。这里通常能看到某些 cliff 附近 runtime 变差，但不保证每个阶梯都对应性能突变。
- 左下：实测 throughput 随 `regs_per_thread` 的变化。它和 runtime 是同一事实的另一种读法，更适合比较“单位时间处理元素数”。
- 右下：如果 CSV 包含 `launch_bounds` 行，则展示 `__launch_bounds__(256, N)` 对 register count、occupancy、runtime 的影响。

读图时要区分三个概念：

- `REG_TMP_SIZE` 是人为控制 register pressure 的模板参数。
- `regs_per_thread` 是 ptxas 实际生成的寄存器数量，才是 occupancy 模型真正使用的输入。
- `theoretical_occupancy` 是资源上限给出的驻留能力，不等价于真实性能。

“寄存器多了不一定更快”的原因是：寄存器可以保存更多临时值，减少重复计算或内存流量；但寄存器太多会减少 resident warps，削弱隐藏延迟的能力。真正的性能结果取决于算术强度、访存模式、指令级并行和 occupancy 的共同作用。

## 结论与可扩展练习

这个项目的结论不是“occupancy 越高越好”，而是：

- Occupancy 是重要约束，不是最终目标。
- Register pressure 的影响有明显阶梯和 cliff。
- 理论模型能解释“为什么某个寄存器数突然改变驻留能力”。
- 实测曲线能回答“这个改变在当前 workload 上到底有没有性能后果”。

可扩展练习：

- 修改 `kIters` 或 `kElements`，观察计算强度变化后 cliff 是否仍然明显。
- 加入 shared memory 使用量，让理论模型同时受到 register 和 shared memory 限制。
- 用 Nsight Compute 采集 achieved occupancy、eligible warps、stall reason，与本项目的理论 occupancy 对照。
- 尝试 `--maxrregcount`，比较它和 `__launch_bounds__` 的区别。
