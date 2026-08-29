# Reduction And Scan Intro：CUDA 归约与前缀和

这个项目实现两个经典并行基元：

- reduction：把数组归约成一个值，例如 sum/max/min
- scan / prefix sum：把数组变成前缀累计结果

它们是 HPC、图计算、排序、压缩、深度学习算子和并行算法里非常常见的构件。

## 1. 本项目包含什么

源码：`src/reduce_scan_bench.cu`

实现：

- shared memory block-level tree reduction
- warp shuffle block reduction
- full array reduction：每个 block 先输出 partial sum，最终在 host 上做第二阶段求和
- 单 block Blelloch exclusive scan
- CPU reference 校验
- CUDA event timing
- CSV 输出和 Python 可视化

## 2. Reduction 的核心思想

串行求和：

```cpp
sum = a[0] + a[1] + ... + a[n-1]
```

并行 reduction 把它变成树：

```text
level 0: a0 a1 a2 a3 a4 a5 a6 a7
level 1: a0+a4, a1+a5, a2+a6, a3+a7
level 2: ...
level k: block sum
```

在 CUDA 里常见做法是：

1. 每个 thread 读 1 到 2 个 global memory 元素
2. 把 partial value 放入 shared memory
3. block 内用树形同步归约
4. 每个 block 输出一个 partial sum
5. 如果 partial sum 数量还很多，再做第二轮；本项目为了清晰，在 host 上做最终归约

## 3. Shared memory tree reduction

核心形态：

```cpp
smem[tid] = x;
__syncthreads();

for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
}
```

它的优点是容易理解，缺点是每一层都需要 block 同步，并且 shared memory 访问模式可能产生 bank conflict。

## 4. Warp shuffle reduction

warp shuffle 用寄存器间交换减少 shared memory 使用：

```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
}
```

优点：

- warp 内不需要 shared memory
- warp 内不需要 `__syncthreads()`
- 常比纯 shared memory reduction 更轻

本项目仍使用少量 shared memory 保存每个 warp 的 sum，然后由 warp0 做最后归约。

## 5. Shared memory reduction 里的 bank conflict

shared memory 通常按 32 个 bank 理解。树形 reduction 的不同阶段会让线程访问：

```text
tid 和 tid + stride
```

如果这些地址映射到同一个 bank，warp 内访问会被拆成多轮，导致下降。

现代写法常见优化：

- 每个 thread 先读两个 global 元素，减少 block 数
- 后半段用 warp shuffle，减少 shared memory 层数
- 避免复杂 stride 访问
- 对特殊数据类型或结构体注意 padding 和对齐

## 6. Exclusive scan 是什么

输入：

```text
[3, 1, 4, 2]
```

exclusive scan 输出：

```text
[0, 3, 4, 8]
```

inclusive scan 输出：

```text
[3, 4, 8, 10]
```

本项目实现单 block Blelloch exclusive scan，适合学习算法结构；完整大数组 scan 还需要 block sums、扫描 block sums、再把偏移加回每个 block。

## 7. Blelloch scan 两阶段

Upsweep：构建树，把总和推到根节点。

Downsweep：把根节点置 0，然后把父节点前缀分发给左右子树。

伪代码结构：

```cpp
for (offset = 1; offset < n; offset <<= 1) upsweep;
temp[n - 1] = 0;
for (offset = n / 2; offset > 0; offset >>= 1) downsweep;
```

## 8. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/reduce_scan_bench
```

可选参数：

```bash
./build/reduce_scan_bench <output_csv> <repeats> <warmup> <elements> <block_size>
# 示例
./build/reduce_scan_bench results/reduce_scan_benchmark.csv 30 5 4194304 256
```

CSV 字段：

- `operation`：`reduction` 或 `scan`
- `variant`：实现版本
- `mean_ms` / `std_ms`：CUDA event 时间
- `effective_gb_s`：按请求字节估算的有效带宽
- `max_abs_error`：对 CPU reference 的最大误差

## 9. 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/reduce_scan_benchmark.csv --outdir results
```

输出：

- `results/runtime_compare.png`
- `results/bandwidth_compare.png`
- `results/summary.csv`

## 10. 怎么理解结果

通常你会看到：

- reduction 很容易受 memory bandwidth 限制，因为每个元素只做少量计算
- warp shuffle 版本可能更快，因为同步和 shared memory 压力更少
- scan 比 reduction 更复杂，因为它不只需要一个最终值，还要给每个元素生成结果
- scan 单 block 版本不是完整工业实现，它的价值是把 upsweep/downsweep 结构讲清楚

## 11. 你应该学到什么

- reduction 和 scan 是并行算法基本功
- shared memory 能减少 global memory 往返，但也有同步和 bank conflict 成本
- warp shuffle 是 warp 内通信的高效工具
- 完整数组 reduction 常是多阶段算法，不是一个 kernel 就自然完成所有规模

一句话记忆：

Reduction 把很多值压成一个值；scan 把“之前所有值的信息”传播到每个位置。
