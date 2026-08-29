# Nsight Compute Intro：用指标解释单个 CUDA Kernel

这个项目复用 `../memory_coalescing_intro/build/mem_coalescing_bench`，用 Nsight Compute 分析 global memory 访问模式对单 kernel 的影响。

核心目标：

- 理解 Nsight Compute 和 Nsight Systems 的分工
- 学会看 occupancy、stall reasons、memory throughput、roofline
- 把“stride 变大后带宽下降”解释到硬件指标层面

## 1. Nsight Compute 看什么

Nsight Compute 面向单个 kernel 的深入分析。它回答的问题包括：

- 这个 kernel 的理论 occupancy 和实际活跃 warp 情况如何
- warp 主要卡在哪里：访存、依赖、同步、指令吞吐还是调度
- global memory load/store 是否合并良好
- DRAM/L2/L1/TEX 吞吐是否接近上限
- roofline 上它更像 compute-bound 还是 memory-bound

它不适合看跨 stream 的全局时间线；那是 Nsight Systems 的工作。

## 2. 选择的分析对象

默认目标是 `memory_coalescing_intro`：

- `stride_read_kernel`：改变 stride，观察 coalescing 破坏后的带宽变化
- `offset_read_kernel`：改变连续访问起始 offset，观察对齐影响

这个目标适合入门 ncu，因为代码很短，性能差异主要来自访存模式。

## 3. 运行完整 ncu profile

```bash
cd nsight_compute_intro
bash scripts/run_ncu.sh
```

可调参数：

```bash
REPEATS=5 WARMUP=1 ELEMENTS=1048576 MAX_STRIDE=32 OUT_NAME=mem_coalescing_ncu bash scripts/run_ncu.sh
```

输出：

- `results/mem_coalescing_ncu.ncu-rep`
- `results/mem_coalescing_profile_input.csv`

打开 GUI：

```bash
ncu-ui results/mem_coalescing_ncu.ncu-rep
```

## 4. 运行 roofline 分析

```bash
bash scripts/run_ncu_roofline.sh
```

它会采集：

- `SpeedOfLight_RooflineChart`
- `ComputeWorkloadAnalysis`
- `MemoryWorkloadAnalysis`

输出：

- `results/mem_coalescing_roofline.ncu-rep`

## 5. 重点指标怎么读

### 5.1 Occupancy

常看：

- `Theoretical Occupancy`
- `Achieved Occupancy`
- active warps / SM
- registers per thread
- shared memory per block

解释方式：

- 理论 occupancy 低，通常说明寄存器、shared memory、block size 或硬件 block 上限限制了驻留
- achieved occupancy 低，可能是 workload 太小、分支/退出太早、或实际调度没有填满
- occupancy 不是越高越好；足够隐藏延迟即可

### 5.2 Stall reasons

常看 Scheduler Statistics 和 Warp State：

- `Stall Long Scoreboard`：常见于等待 global memory load
- `Stall Short Scoreboard`：常见于 shared memory 或较短依赖
- `Stall Wait`：可能是同步、队列或采样解释下的等待
- `Stall Not Selected`：warp 可运行但调度器选择了别的 warp
- `Stall Math Pipe Throttle`：算术管线压力高

对 coalescing 实验，stride 变大时更可能看到 memory 相关 stall 占比上升。

### 5.3 Memory throughput

重点看：

- DRAM Throughput
- L2 Throughput
- Global Memory Load Efficiency
- sector/request、transactions/request 类指标

直觉：

- stride=1 时，一个 warp 的访问更容易合并成少量内存事务
- stride 增大后，同样 32 个线程可能触达更多 cache line/sector
- 请求字节数没变，但实际搬运字节和事务数上升，有效带宽下降

### 5.4 Roofline

Roofline 通过算术强度判断瓶颈：

- 算术强度低、贴近带宽上限：memory-bound
- 算术强度高、贴近算力上限：compute-bound
- 两者都远低于上限：可能有 latency、分支、同步、launch 或 occupancy 问题

coalescing benchmark 通常会落在 memory-bound 区域。

## 6. 常用命令解释

完整分析：

```bash
ncu --set full --target-processes all --kernel-name-base demangled --export results/mem_coalescing_ncu --force-overwrite ...
```

只看指定 section：

```bash
ncu --section MemoryWorkloadAnalysis --section SchedulerStats ...
```

导出 CSV：

```bash
ncu --csv --page raw --section MemoryWorkloadAnalysis ./target
```

减少采样开销：

```bash
ncu --launch-skip 5 --launch-count 1 ./target
```

## 7. 你应该学到什么

- ncu 用来解释单 kernel 的瓶颈，不是看程序整体调度
- memory-bound kernel 的关键不只是耗时，而是事务数、吞吐和 stall
- occupancy 指标要和 stall 一起读，不能孤立判断
- roofline 是分类工具，不是自动优化器

一句话记忆：

Nsight Compute 把“变慢了”拆成硬件原因：访存、占用率、依赖、管线和调度。
