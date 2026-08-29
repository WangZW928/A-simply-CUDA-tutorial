# Memory Coalescing Intro

这个项目用最小实验尝试说明 CUDA 里的 **global memory coalescing** 是什么，以及为什么 warp 内的访问模式会直接影响全局内存带宽。

## 1. 什么是 memory coalescing

在 NVIDIA GPU 上，线程通常以 warp 为单位执行。一个 warp 通常有 32 个线程。

当一个 warp 内的线程访问 global memory 时，如果这些线程访问的是连续、相邻、对齐比较好的地址，硬件可以把这些访问合并成更少的 memory transaction。这种合并访问就叫 memory coalescing，因为GPU架构在全局内存与CUDA core之间有L2 Cache，良好的内存合并能够减轻数据的LS成本，GPU的架构图见：

![NVIDIA GPU 与 CPU 系统架构图](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)


直觉上：

- 好的访问模式：`thread0 -> a[0]`, `thread1 -> a[1]`, `thread2 -> a[2]`
- 差的访问模式：`thread0 -> a[0]`, `thread1 -> a[32]`, `thread2 -> a[64]`

前者更容易高效利用Cache，后者会让一个 warp 的访问分散到更多 memory transaction 里，最差的情况下，每个lane都需要从global memory拿数据。

## 2. 本项目做什么

项目包含两个实验：

### 2.1 Stride sweep

同样数量的线程读取同样数量的元素，但改变每个线程读取 input 的步长：

```cpp
out[idx] = in[idx * stride];
```

当 `stride = 1` 时，相邻线程读相邻地址，访问最容易合并。

当 `stride` 增大时，相邻线程读的地址越来越分散，requested bandwidth 往往会下降。

### 2.2 Offset sweep

这个实验保持连续访问，只改变起始偏移：

```cpp
out[idx] = in[idx + offset];
```

它用于观察“连续访问但起始地址偏移”时性能是否会变化。现代 GPU 对这种情况通常比大 stride 更宽容，但它仍然是理解对齐影响的好入口 (注意，每个线程的访存还是连续的) 。

## 3. 项目结构

```text
.
├── CMakeLists.txt
├── src/
│   └── mem_coalescing_bench.cu
├── scripts/
│   └── plot_results.py
├── requirements.txt
├── results/
└── README.md
```

## 4. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/mem_coalescing_bench
```

本项目默认使用 `CMAKE_CUDA_ARCHITECTURES=89`，适合 RTX 4060 Laptop GPU 这类 Ada 架构显卡。如果你换了其他 GPU，可以手动覆盖：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=<your_sm_version>
```

可选参数：

```bash
./build/mem_coalescing_bench <output_csv> <repeats> <warmup> <elements> <max_stride>
# 示例
./build/mem_coalescing_bench results/memory_coalescing.csv 80 10 4194304 32
```

参数说明：

- `output_csv`：结果 CSV 路径，默认 `results/memory_coalescing.csv`
- `repeats`：正式计时重复次数，默认 `50`
- `warmup`：预热次数，默认 `10`
- `elements`：每个 kernel 处理的输出元素数，默认 `1048576`
- `max_stride`：stride sweep 的最大步长，默认 `32`

## 5. Python 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/memory_coalescing.csv --outdir results
```

输出：

- `results/bandwidth_vs_stride.png`
- `results/runtime_vs_stride.png`
- `results/bandwidth_vs_offset.png`
- `results/stride_summary.csv`

## 6. 如何理解 CSV 字段

- `experiment`：实验类型，`stride_sweep` 或 `offset_sweep`
- `param_name`：参数名，`stride` 或 `offset`
- `param_value`：参数值
- `mean_ms`：平均 kernel 时间
- `std_ms`：计时标准差
- `requested_bytes`：按“每个元素一次读 + 一次写”估算的请求字节数
- `requested_bandwidth_gb_s`：`requested_bytes / mean_ms` 换算出的请求带宽

注意：这里的 bandwidth 是 requested bandwidth，不一定等于硬件实际 memory transaction 的总字节数。stride 变差时，requested bytes 没变，但实际 transaction 可能变多，所以 requested bandwidth 会下降。


## 7. 总结

Memory coalescing 的核心是“warp 内线程要尽量一起访问连续地址”。
