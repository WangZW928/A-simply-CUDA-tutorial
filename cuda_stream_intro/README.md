# CUDA Stream 入门项目

这个项目用一个最小实验说明 `cudaStream` 是干什么的，以及它最基本的使用方式。

## 1. `cudaStream` 是什么

`cudaStream` 可以理解为一条提交给 GPU 的“任务队列”。

- 同一个 stream 里的操作默认按顺序执行
- 不同 stream 里的操作在条件允许时可以重叠执行
- 常见可重叠对象包括：kernel、`cudaMemcpyAsync`、部分同步较少的 GPU 工作

`stream` 不是性能指标，而是组织和调度 GPU 工作的机制。

## 2. 为什么要用 stream

如果我们把所有工作都放在默认 stream 里，很多操作会串行排队。

而当任务可以拆成多块时，我们可以：

1. 一边传第 1 块数据
2. 一边算第 0 块数据
3. 一边回传上一块结果

这样就有机会把“拷贝”和“计算”重叠起来，从而节省计算时间。

## 3. 什么是锁页内存

锁页内存也常叫：

- `pinned memory`
- `page-locked memory`

它的含义是：这块主机内存（在CPU上，不在GPU上）不会被操作系统随意换页到磁盘，所以 GPU/驱动可以更稳定地直接访问它。

在 CUDA 里，常见分配方式是：

```cpp
float* h_ptr = nullptr;
cudaMallocHost(&h_ptr, bytes);
```

为什么它重要：

- 普通 `std::vector` 或 `new` 出来的 host 内存，通常是 pageable memory（即，非锁业的）
- pageable memory 往往不能很好地支持真正高效的异步 H2D / D2H 拷贝
- 如果想让 `cudaMemcpyAsync` 更有机会和 kernel 重叠（在运行时间上），通常需要 pinned memory

所以在本项目的 [stream_bench.cu](./src/stream_bench.cu) 里会看到：

```cpp
cudaMallocHost(&host.a, total_bytes);
cudaMallocHost(&host.b, total_bytes);
cudaMallocHost(&host.out, total_bytes);
```

## 4. 什么是异步拷贝

异步拷贝通常指：

```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
```

它和普通 `cudaMemcpy` 的区别是：

- `cudaMemcpy` 更像是流水线上的顺序任务提交
- `cudaMemcpyAsync` 是“把这项拷贝任务提交到某个 stream 队列里”

这并不等于“它一定立刻并发执行”，是否真的重叠还取决于：

- 是否使用了合适的 stream
- host 内存是否是 pinned memory
- GPU 是否支持相应的 copy/compute overlap
- 当前 workload 是否真的存在可重叠空间

准确地说：

`Async` 的意思是“异步提交”，并不是“保证并发加速”。

## 5. 本项目做了什么

本项目把数据分成两块（chunk），比较两种模式：

- `default`：所有 `memcpy + kernel + memcpy` 都走默认 stream，基本串行
- `two_streams`：使用两个显式 stream，把两块数据分别放进 `s0` 和 `s1`

核心目的是让你看到：

- stream 的概念不是“让一个 kernel 更快”
- 它更像是在优化整个流水线的并发与调度（并不等价于并行）

## 6. 调用逻辑

对应 [stream_bench.cu](./src/stream_bench.cu)，显式 stream 的典型流程是：

1. 创建 stream

```cpp
cudaStream_t s0, s1;
cudaStreamCreate(&s0);
cudaStreamCreate(&s1);
```

2. 把异步拷贝和 kernel 绑定到指定 stream

```cpp
cudaMemcpyAsync(..., s0);
vector_add<<<blocks, threads, 0, s0>>>(...);
cudaMemcpyAsync(..., s0);
```

```cpp
cudaMemcpyAsync(..., s1);
vector_add<<<blocks, threads, 0, s1>>>(...);
cudaMemcpyAsync(..., s1);
```

3. 最后同步并销毁

```cpp
cudaStreamSynchronize(s0);
cudaStreamSynchronize(s1);
cudaStreamDestroy(s0);
cudaStreamDestroy(s1);
```

一句话流程：

`create -> enqueue async work -> synchronize -> destroy`

## 7. 项目结构

```text
.
├── CMakeLists.txt
├── src/
│   └── stream_bench.cu
├── scripts/
│   └── plot_results.py
├── requirements.txt
├── results/
└── README.md
```

## 8. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/stream_bench
```

可选参数：

```bash
./build/stream_bench <output_csv> <repeats> <warmup> <chunk_elems> <iters>
# 示例
./build/stream_bench results/stream_benchmark.csv 30 5 1048576 512
```

## 9. 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/stream_benchmark.csv --outdir results
```

输出：

- `results/stream_vs_default.png`
- `results/summary.txt`

## 10. 怎么理解结果

如果 `two_streams` 更快，通常说明：

- 当前 GPU 支持一定程度的 copy/compute overlap
- 当前 workload 的切分方式让并发有收益

如果差异不明显，常见原因是：

- kernel 太重，拷贝开销占比小
- kernel 太轻，测量噪声大
- 数据没有真正满足“可异步重叠”的条件

## 11. 总结

`cudaStream` 是 GPU 工作队列；它的意义在于“安排并发”，不是“改变 kernel 算法本身”。
