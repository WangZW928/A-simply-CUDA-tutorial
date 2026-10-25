# 硬件架构延伸：从 CUDA 抽象到 tiny-gpu

本仓库的 CUDA 实验回答“程序在真实 NVIDIA GPU 上表现如何”；[tiny-gpu](https://github.com/WangZW928/tiny-gpu) 则用精简的 Verilog GPU、简单 ISA 和 cocotb 仿真，帮助观察“线程、指令与访存请求如何被组织”。两者互补，不应把 tiny-gpu 当成 CUDA 实现或现代 GPU 的精确模型。

## 建议学习顺序

1. 先完成 [`cuda_basics_intro`](./cuda_basics_intro/README.md)，弄清 kernel、线程索引、边界、内存和同步。
2. 阅读 [`tiny-gpu README 的 Architecture / ISA / Execution`](https://github.com/WangZW928/tiny-gpu#architecture)，先跟一遍矩阵加法，再看矩阵乘法的控制流。
3. 在本机按 tiny-gpu 仓库说明运行 `make test_matadd`，阅读测试日志中的初始/最终数据内存和逐周期 trace；再运行 `make test_matmul` 观察分支与多步访存。
4. 回到 [`warp_schedule`](./warp_schedule/README.md)、[`memory_coalescing_intro`](./memory_coalescing_intro/README.md) 和 [`shared_memory_bank_conflict`](./shared_memory_bank_conflict/README.md)，比较教学模型和真实 GPU 性能实验分别能回答什么问题。

tiny-gpu 仓库单独维护自己的 RTL、仿真依赖和测试；本学习指南只提供概念连接，不复制源码、图片或构建流程。

## 概念对照

| CUDA / GPU 概念 | tiny-gpu 中可以观察什么 | 阅读时的边界 |
| --- | --- | --- |
| `blockIdx`、`blockDim`、`threadIdx` | dispatcher 分派 block；线程通过特殊寄存器得到 block/thread 信息，矩阵加法据此计算索引 | tiny-gpu 只实现足以演示索引映射的简化控制寄存器与执行模型 |
| kernel 指令与执行 | 简化的 16-bit ISA、fetch/decode/execute 控制路径，以及测试日志中的 PC、寄存器、指令 trace | 这不是 PTX、SASS 或 NVIDIA GPU 的真实指令编码 |
| 全局访存与等待 | LSU 发出 `LDR`/`STR` 请求，memory controller 在有限 channel 间仲裁，仿真可观察等待/返回 | 模型刻意简化延迟与带宽；不要从其周期数推断 RTX 的性能 |
| block 调度 | dispatcher 把 block 分给可用 core；可追踪 `thread_count` 如何决定 block 数量 | 一次只执行一个 kernel；不能代表现代 GPU 的全部并发调度策略 |
| SIMD/SIMT 与分支 | 各线程具有本地 PC/寄存器；矩阵乘法展示 CMP/branch 的基础流程 | scheduler 假设同一执行组线程的 PC 收敛，不实现真实 branch divergence |
| 性能优化思路 | README 讨论 cache、shared memory、coalescing、pipelining、warp scheduling、barrier 等功能 | 其中多项是缺失/待扩展功能；真实 CUDA 章节才用于实际 GPU 测量 |

## 两个视角如何互相校验

- CUDA 中 `i = blockIdx.x * blockDim.x + threadIdx.x` 是程序员看到的索引表达式；tiny-gpu 的矩阵加法可继续追踪这些特殊寄存器如何喂给 ALU。
- CUDA kernel 会发起全局 load/store；tiny-gpu trace 可展示一个教学模型里的请求、等待、返回过程。它解释控制与数据流，不模拟现代 GPU 的完整 cache hierarchy 或真实内存带宽。
- CUDA 的 warp 调度、合并访存和 bank conflict 应以本仓库的真实 GPU benchmark、Profiler 证据为准。tiny-gpu 把这些作为现代 GPU 的性能机制讨论，其中部分没有实现。

推荐把 `tiny-gpu` 当成一个可读、可仿真的硬件“放大镜”：先用 CUDA 实验建立真实软件行为的基线，再用 RTL trace 理解某些硬件机制的可能实现方式，最后回到 CUDA 实测，避免把教学架构细节误当成 NVIDIA 的内部设计。
