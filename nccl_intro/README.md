# NCCL 入门

这个项目用几个最小实验说明 NCCL 是什么、怎么写，以及它在多 GPU 通信里对应什么硬件行为。

核心路线仍然是：

`代码写法 -> GPU / 互联硬件行为 -> 可测性能 -> profiler 证据`

## 1. NCCL 是什么

NCCL 是 NVIDIA Collective Communications Library，主要用来做多 GPU 之间的高性能通信。

在 HPC 和深度学习里，常见场景包括：

- 多卡训练里的梯度同步
- 多 GPU 分布式 stencil / solver 的边界交换
- 模型并行或张量并行里的数据重排
- 单机多卡和多机多卡的通信统一抽象

一句话理解：

`NCCL 不是 kernel 优化库，而是 GPU 之间的数据搬运和 collective 通信库。`

## 2. Collective 操作速览

NCCL 里最常见的操作是 collective，也就是一组 rank 一起参与的通信。

- `AllReduce`：每个 rank 输入一份数据，先做 reduce，再让每个 rank 都拿到结果。多卡训练梯度同步最常见。
- `Broadcast`：一个 root rank 发数据，所有 rank 接收同一份数据。
- `AllGather`：每个 rank 贡献一段数据，最后每个 rank 都拿到所有 rank 的拼接结果。
- `ReduceScatter`：先 reduce，再把结果切片分给不同 rank。
- `All2All`：每个 rank 都向每个 rank 发送不同分片，常见于更复杂的数据重排。

本项目重点测 `AllReduce`，因为它最容易和带宽、拓扑、算法选择联系起来。

单 GPU 说明：本机只有一张 GPU 时，可以运行 `num_gpus=1` 的 single-rank
smoke/latency test，验证 CUDA、NCCL communicator、stream enqueue、同步和
AllReduce 数据路径；此时 `busbw` 为 0，因为没有跨 GPU 总线通信，不能用来
代表多 GPU 带宽或拓扑性能。真正的 NCCL 多 GPU 通信、P2P 和拓扑比较仍需要
至少两张 GPU。

## 3. rank、communicator 和 stream

NCCL 的几个核心概念：

- `rank`：通信组里的编号。单进程多 GPU 时，通常 `GPU i` 对应 `rank i`。
- `communicator`：一组 rank 的通信上下文。NCCL API 调用都需要 `ncclComm_t`。
- `UniqueId`：用来让多个 rank 加入同一个通信组。单进程里直接共享，多进程里通常要通过 MPI / socket 广播。
- `stream`：NCCL 操作提交到 CUDA stream 中，和 kernel、`cudaMemcpyAsync` 一样参与 stream 排队。

典型初始化模式：

```cpp
ncclUniqueId id;
ncclGetUniqueId(&id);

ncclGroupStart();
for (int rank = 0; rank < num_gpus; ++rank) {
    cudaSetDevice(rank);
    ncclCommInitRank(&comm[rank], num_gpus, id, rank);
}
ncclGroupEnd();
```

典型 AllReduce：

```cpp
ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream);
```

这里的同步语义很重要：

- NCCL 调用本身通常是把通信任务 enqueue 到 stream
- 真正等待完成要靠 `cudaStreamSynchronize(stream)` 或后续依赖
- 同一个 stream 里，NCCL 通信和 kernel 仍然按提交顺序执行
- 不同 stream 是否能重叠，取决于 GPU、NCCL、拓扑和 workload

## 4. 为什么需要 ncclGroupStart / ncclGroupEnd

单进程控制多张 GPU 时，程序会在一个 CPU 线程里依次为多个 rank 调 NCCL API。

如果不 group，某个 rank 的 send / recv / collective 可能先等待另一个 rank，而另一个 rank 的调用还没提交，容易形成不必要的阻塞。

所以常见写法是：

```cpp
ncclGroupStart();
for (int rank = 0; rank < num_gpus; ++rank) {
    cudaSetDevice(rank);
    ncclAllReduce(..., comm[rank], stream[rank]);
}
ncclGroupEnd();
```

一句话记忆：

`group 不是 collective 本身，而是把多个 rank 的 NCCL 调用作为一组提交。`

## 5. ring、tree 和拓扑意识

NCCL 会根据消息大小、GPU 数量、互联拓扑选择通信算法和通道。

常见直觉：

- ring 算法适合大消息，能把链路带宽利用得比较满
- tree 算法常用于更低延迟的场景，尤其是小消息或特定拓扑
- NVLink / NVSwitch 通常比 PCIe 有更高带宽和更低延迟
- 跨 PCIe switch、跨 NUMA、跨 socket 的通信，可能明显慢于同一 NVLink 域

这也是为什么同样的 AllReduce 代码，在不同机器上的曲线可能差很多。

可以先用这些命令观察硬件：

```bash
nvidia-smi topo -m
NCCL_DEBUG=INFO ./build/nccl_allreduce_bench
```

`NCCL_DEBUG=INFO` 会打印 NCCL 选择的通道、拓扑和通信路径信息，是理解结果的重要证据。

## 6. 项目结构

```text
.
├── CMakeLists.txt
├── README.md
├── src/
│   ├── nccl_allreduce_bench.cu
│   ├── nccl_p2p_demo.cu
│   └── nccl_common.h
├── scripts/
│   └── plot_results.py
└── results/
    └── .gitkeep
```

## 7. 编译

如果系统已经安装 CUDA 和 NCCL：

```bash
cmake -S . -B build
cmake --build build -j
```

如果 NCCL 安装在自定义路径：

```bash
cmake -S . -B build -DNCCL_ROOT=/path/to/nccl
cmake --build build -j
```

如果没有 NCCL，CMake 会正常 configure，但会打印：

```text
NCCL not found. Skipping nccl_intro targets.
```

这台开发机器没有可用 NVIDIA GPU / driver，因此这里只保证代码和构建脚本按 NCCL 环境编写，实际运行需要在有多 GPU 和 NCCL 的机器上完成。

## 8. AllReduce 带宽实验

运行：

```bash
./build/nccl_allreduce_bench
```

单 GPU 运行示例：

```bash
./build/nccl_allreduce_bench results/nccl_allreduce_single_gpu.csv 1 20 5 1024 67108864
```

本机单 GPU 运行结果记录在
`results/nccl_allreduce_single_gpu.csv`；这是 single-rank smoke/latency
evidence，不是多 GPU 通信性能结果。

可选参数：

```bash
./build/nccl_allreduce_bench <output_csv> <num_gpus> <repeats> <warmup> <min_bytes> <max_bytes>
```

示例：

```bash
./build/nccl_allreduce_bench results/nccl_allreduce.csv 4 30 5 1024 268435456
```

CSV 字段：

- `message_bytes`：每个 rank 的消息大小
- `num_gpus`：参与通信的 GPU 数量
- `mean_ms`：多次重复后的平均耗时
- `algbw_gb_s`：按每 rank 数据量计算的算法带宽
- `busbw_gb_s`：按 AllReduce 通信放大系数估算的总线带宽

AllReduce 的 bus bandwidth 估算使用：

```text
busbw = algbw * 2 * (num_gpus - 1) / num_gpus
```

它不是硬件链路的真实峰值，而是一个便于比较不同 GPU 数量的 NCCL 常用指标。

## 9. P2P send / recv 实验

运行：

```bash
./build/nccl_p2p_demo
```

可选参数：

```bash
./build/nccl_p2p_demo <elements>
```

这个 demo 只做一件事：

1. rank 0 在 GPU 0 上生成一段序列
2. rank 0 用 `ncclSend` 发给 rank 1
3. rank 1 用 `ncclRecv` 接收
4. 拷回 CPU 做简单校验

它对应的知识点是：NCCL 不只支持 collective，也支持 GPU 间点对点通信。

## 10. NCCL + CUDA stream overlap

AllReduce benchmark 还提供一个简化 overlap 模式：

```bash
./build/nccl_allreduce_bench results/nccl_overlap.csv 2 20 5 1048576 67108864 --overlap
```

这个模式在同一个 stream 中提交：

```text
kernel -> ncclAllReduce -> kernel
```

它的目的不是追求最快，而是让你观察：

- NCCL 操作和 kernel 一样进入 CUDA stream
- 同一 stream 内保持顺序依赖
- Nsight Systems 里可以看到 kernel 和 NCCL 通信的时间线关系

如果想研究真正的 compute / communication overlap，可以进一步扩展成多个 stream，并让不同数据块分段通信。

## 11. 可视化

安装 Python 依赖：

```bash
python -m pip install pandas matplotlib
```

画图：

```bash
python scripts/plot_results.py --input results/nccl_allreduce.csv --outdir results
```

输出：

- `results/allreduce_algbw.png`
- `results/allreduce_busbw.png`
- `results/nccl_summary.csv`

## 12. 结果怎么解读

典型 AllReduce 曲线通常会有几个阶段：

- 小消息：延迟主导，带宽看起来很低
- 中等消息：带宽快速爬升
- 大消息：逐渐接近当前拓扑和算法能达到的平台

如果带宽明显低于预期，可以优先检查：

- 是否真的用了 NVLink / NVSwitch，而不是绕到 PCIe
- GPU 是否跨 socket 或跨 NUMA
- `NCCL_DEBUG=INFO` 里是否有异常 fallback
- 是否有其他进程占用 GPU 或互联链路
- 消息大小是否太小，导致主要测到 launch / latency

## 13. profiler 证据

建议用 Nsight Systems 看时间线：

```bash
nsys profile -t cuda,nvtx,osrt -o results/nccl_allreduce_nsys \
    ./build/nccl_allreduce_bench results/nccl_allreduce.csv 2 20 5 1048576 67108864
```

重点看：

- 每张 GPU 的 stream 上是否出现 NCCL kernel
- NCCL kernel 和普通 CUDA kernel 的先后关系
- 不同 GPU 上通信任务是否同时开始
- 小消息和大消息的时间线形态是否不同

## 14. 一句话记忆

`NCCL 把多 GPU 通信变成 CUDA stream 里的可排队工作；性能上限主要由消息大小、算法选择和 GPU 互联拓扑共同决定。`
