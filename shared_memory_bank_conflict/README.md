# Shared Memory Bank Conflict

这个项目用最小实验说明 CUDA shared memory 里的 **bank conflict** 是什么，以及为什么 shared memory 虽然快，但访问方式不对时也会变慢。

## 1. Shared memory 是什么

Shared memory 是每个 SM 上的一块片上内存。它比 global memory 延迟更低，常用于：

- block 内线程共享数据
- 数据重用
- reduction / scan / stencil / 矩阵分块等算法

但是 shared memory 不是“无限快”。它内部通常被分成多个 bank。

## 2. 什么是 bank

可以把 shared memory 想成被分成很多条“通道”。在 NVIDIA GPU 上，经典理解里一个 warp 有 32 个线程，shared memory 也通常按 32 个 bank 来理解。

如果一个 warp 内 32 个线程访问的地址刚好落在不同 bank，访问可以高效并行完成。

如果多个线程访问不同地址，但这些地址落在同一个 bank，就可能产生 bank conflict。硬件需要把访问拆成多轮处理，导致变慢。

## 3. 本项目的实验方式

本项目使用一个单 warp 实验：每个 block 只启动 32 个线程，刚好对应一个 warp。访问模式是：

```cpp
int lane = threadIdx.x & 31;
int index = lane * stride;
acc += smem[index];
```

当 `stride = 1` 时：

- lane0 访问 `smem[0]`
- lane1 访问 `smem[1]`
- lane2 访问 `smem[2]`

这些访问通常会分布到不同 bank，冲突少。

当 `stride = 32` 时：

- lane0 访问 `smem[0]`
- lane1 访问 `smem[32]`
- lane2 访问 `smem[64]`

这些地址可能映射到同一个 bank，冲突严重。

为了让重复读取真的发生，源码里使用了：

```cpp
volatile float* vsmem = smem;
acc += vsmem[index];
```

这里的 `volatile` 是有意加入的。否则编译器可能发现 `smem[index]` 在循环中不变，把 shared memory load 提前成一次读取，再在寄存器里重复累加。那样测到的就不再主要是 shared memory bank conflict。

## 4. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/bank_conflict_bench
```

本项目默认使用 `CMAKE_CUDA_ARCHITECTURES=89`，适合 RTX 4060 Laptop GPU 这类 Ada 架构显卡。如果你换了其他 GPU，可以手动覆盖：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=<your_sm_version>
```

可选参数：

```bash
./build/bank_conflict_bench <output_csv> <repeats> <warmup> <iterations> <blocks_per_sm>
# 示例
./build/bank_conflict_bench results/bank_conflict.csv 80 10 8192 8
```

## 5. Python 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/bank_conflict.csv --outdir results
```

输出：

- `results/runtime_vs_stride.png`
- `results/throughput_vs_stride.png`
- `results/estimated_conflict_degree.png`
- `results/bank_conflict_summary.csv`

## 6. 当前结果分析

你当前的结果来自 RTX 4060 Laptop GPU，配置大致是：

- `sm_count = 24`
- `blocks = 192`
- `threads_per_block = 32`
- `iterations = 4096`
- 每组 `shared_loads = 25,165,824`

当前 CSV 里的关键数据如下：

| stride | estimated conflict | mean ms | relative throughput |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0.026636 | 1.000 |
| 2 | 2 | 0.028317 | 0.941 |
| 3 | 1 | 0.032140 | 0.829 |
| 4 | 4 | 0.031017 | 0.859 |
| 5 | 1 | 0.031820 | 0.837 |
| 8 | 8 | 0.031620 | 0.842 |
| 16 | 16 | 0.029615 | 0.899 |
| 32 | 32 | 0.023859 | 1.116 |

这组结果最值得注意的一点是：它没有呈现“conflict degree 越高，耗时越高”的简单趋势。尤其是 `stride = 32`，理论冲突最严重，但它反而最快。

这通常说明：当前测量并不只是 bank conflict 在起作用。更具体地说，可能有几个原因：

- 原始版本里 `smem[index]` 在循环中不变，编译器可能把 shared memory 读取提升成一次 load，然后在寄存器里重复累加。
- kernel 很短，`mean_ms` 只有 0.02 到 0.03 ms，计时噪声占比不小。
- `std_ms` 接近 0.004 到 0.010 ms，相对均值并不低，说明这组数据波动比较明显。
- 现代 GPU 的 shared memory 路径、指令调度、缓存/广播行为会让实际结果不像教科书模型那样线性。

所以这组结果的正确解读不是“bank conflict 理论错了”，而是：

这个 benchmark 的第一版还不够纯粹，它暴露了一个性能实验里非常重要的问题：编译器优化和计时噪声可能盖住你原本想观察的硬件现象。

## 7. 建议重新跑的修正版

源码现在已经加入 `volatile float* vsmem = smem;`，用于强制循环里重复从 shared memory 读取。建议重新编译运行：

```bash
cmake --build build -j
./build/bank_conflict_bench
python scripts/plot_results.py --input results/bank_conflict.csv --outdir results
```

重新跑后更值得观察：

- `stride = 1` 是否仍然接近最快
- `stride = 2, 4, 8, 16, 32` 是否出现更明显的下降
- `stride = 3, 5` 是否比 `2, 4, 8` 更接近无冲突情况
- `std_ms / mean_ms` 是否变小，说明测量更稳定

如果曲线仍然不完全按冲突倍数变化，也很正常。真实 GPU 上性能不只由 bank conflict 决定，还会受到调度、指令吞吐、编译器和架构实现影响。

## 8. 一句话记忆

Shared memory 快，但它不是魔法。想观测 bank conflict，实验本身也要足够“干净”：既要让 warp 访问不同 bank，也要防止编译器把你想测的 load 优化没了。
