# Matrix Mul 入门：从二维线程到共享内存复用

这是 A-simply-CUDA-tutorial 在 [Vector Add](../vector_add_intro/README.md) 之后、[CUDA Stream](../cuda_stream_intro/README.md) 之前的一份可选练习。沿用 CPU 输入、H2D、kernel、D2H 和 CPU 校验的数据流，学习二维索引与 shared memory 协作；已熟悉这些概念可跳过。

本章是**学习材料，不是最快或通用的 GEMM 实现**。[src/matrix_mul_demo.cu](src/matrix_mul_demo.cu) 包含朴素版与分块版，各执行一次；不计时、不输出 benchmark CSV，也不接入 profiler。

## 1. C = A × B 在数组里是什么

设 A 为 `M×K`、B 为 `K×N`，输出 C 为 `M×N`。相等的是 A 的列数与 B 的行数，即内维度 K；矩阵乘法不是对应元素相乘：

```text
C[row, col] = Σ A[row, k] * B[k, col]，k 从 0 到 K-1
```

本例全部使用连续的**行主序（row-major）**，同一行的相邻列在内存中相邻：

| 元素 | 一维数组下标 |
| --- | --- |
| A[row, k] | `row * K + k` |
| B[k, col] | `k * N + col` |
| C[row, col] | `row * N + col` |

例如 `M=2, K=3, N=4`，`C[1,2] = A[3]*B[2] + A[4]*B[6] + A[5]*B[10]`，写入 `C[6]`。列主序（column-major）则让同一列的相邻行连续，一个 `R×S` 矩阵的下标为 `col * R + row`；本例没有使用这种布局，也没有做转置，不能混用两套公式。

## 2. 从一维线程到二维线程

每个线程负责 C 的一个元素。`x` 对应列、`y` 对应行；每个 block 为 `16×16` 个线程，负责一个 `16×16` 输出区域：

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
```

朴素版 `matrix_mul_naive` 先检查 `row < M && col < N`，再由该线程独立遍历 K、累加乘积并写回。grid 两个维度都向上取整，多出来的线程不访问数组。K 是每个线程的累加长度，不是 grid 的第三个维度。

## 3. 为什么分块，以及尾块如何处理

朴素版中，同一输出行的线程会读取相同的 A 行，同一输出列的线程会读取相同的 B 列。分块版 `matrix_mul_tiled` 将沿 K 方向的工作每 16 项分为一轮：

1. 每线程向 shared memory 的 `as[ty][tx]` 和 `bs[ty][tx]` 各加载一个值，组成两个 tile。
2. 第一次 `__syncthreads()` 等待整个 block 写完，才能读取其他线程加载的数据。
3. 每线程执行 `sum += as[ty][k] * bs[k][tx]`，累加这一轮的 16 个乘积。完整 tile 内，一个 A 值可供同一输出行的 16 个线程使用，一个 B 值可供同一输出列的 16 个线程使用。
4. 第二次 `__syncthreads()` 等待所有线程用完，才能在下一轮覆盖这两块 shared memory。

`sum` 是每线程私有的累加值，跨轮保留；`as/bs` 是同一 block 内共享的数据，不跨 block 共享。这里展示显式数据复用，不据此推断实际加速比。

默认 `M=37, K=23, N=29`，grid 为 `(2,3)`，沿 K 有两轮：

- **K 尾块**：第二轮覆盖 k=16..31，其中 23..31 无效。A 的列越界或 B 的行越界时加载 `0.0f`，因此额外乘积为零。
- **M/N 尾块**：最后一排 block 只有 5 行输出有效，最后一列 block 只有 13 列输出有效。加载 A 时检查 `row < M && base+tx < K`；加载 B 时检查 `base+ty < K && col < N`。无效位置同样补零。
- **同步与写回**：所有线程都走完两次同步，只有 `row < M && col < N` 的线程写 C。不能把朴素版的提前 `return` 搬到分块版开头：无效输出线程也可能负责加载其他线程需要的 A/B 元素，整个 block 还需要一致地参与同步。

每轮都重新赋值 shared memory，包括补零位置；这样不会残留前一轮数据。无需把 host 输入矩阵实际扩展到 16 的整数倍。

## 4. 构建、运行与 CPU 校验

需要 CUDA Toolkit、兼容驱动、一张 NVIDIA GPU、CMake 3.24+ 和支持 C++17 的 host 编译器。目录内有独立 CMake target `matrix_mul_demo`，不依赖 CUDA Samples 的 helper 头文件或整库构建框架。

以下命令供读者后续验证，**本次集成没有构建、运行程序或测试，不宣称已验证任何 Toolkit 版本兼容性**。从仓库根目录执行；`89` 仅为项目已有 RTX 4060 Laptop 环境的架构示例，其他 GPU 应调整，且须由所用 Toolkit 支持：

```bash
cd matrix_mul_intro
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
./build/matrix_mul_demo           # 默认 M=37, K=23, N=29
./build/matrix_mul_demo 1 1 1     # 比一个 tile 更小
./build/matrix_mul_demo 16 16 16  # 恰好一个完整 tile
./build/matrix_mul_demo 17 19 21  # 三个维度均有尾块
./build/matrix_mul_demo 2 3 4     # 小尺寸矩形输入；程序报告 CPU 校验，不打印矩阵
```

省略架构选项时默认 `native`，配置时需要能访问 GPU；工具链说明见[基础章](../cuda_basics_intro/README.md#2-编译并运行)。参数要么省略，要么完整提供 `M K N`，每项为 1–256 的十进制正整数。上限用于控制教学规模和 CPU 参考计算量，不是通用矩阵尺寸支持声明。

CPU 初始化非恒定、带正负号的 `float` 输入，再用三重循环和 `double` 累加生成参考结果。两个 GPU 版本分别逐元素对照该参考，而不是只互相比较；每次 launch 前先把输出重置为 NaN，避免上一个版本的结果掩盖漏写。校验拒绝 NaN/Inf，并检查：

```text
abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)
```

此容差针对本例输入范围，不能直接用于任意矩阵。预期两个版本各打印一行 `PASS`，全部通过才返回 0；非法参数、CUDA 错误或校验失败返回非零。这是行为约定，不是实测记录；数值校验也不能单独证明没有越界。

沿用 Vector Add 的 `CUDA_CHECK`：launch 后用 `cudaGetLastError()` 检查提交，用 `cudaDeviceSynchronize()` 等待并检查执行错误，再 D2H 回读。短生命周期示例在 API 失败时立即退出，不实现库级错误恢复；不应将这里的调试同步照搬进多 stream 性能实验。

## 5. 接下来学什么

- [Vector Add](../vector_add_intro/README.md)：回顾显式内存与完整数据流。
- [CUDA Stream](../cuda_stream_intro/README.md)：继续原课程的 pinned memory、异步拷贝和多队列。
- [Shared Memory Bank Conflict](../shared_memory_bank_conflict/README.md)：在理解协作加载之后，再学习 shared memory 的访问布局。
- [返回学习导航](../README.md#学习顺序)。

## 6. 来源与改编范围

原始项目为 [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)。本次实际审阅的是用户提供的 fork **WangZW928/cuda-samples** 的固定快照，完整提交为 **`8a9e2c830c8a336b4f48c79e5ed5837031eb8551`**；不将 fork 中的材料归为本课程或 fork 作者原创。

- [该快照的 matrixMul.cu](https://github.com/WangZW928/cuda-samples/blob/8a9e2c830c8a336b4f48c79e5ed5837031eb8551/Samples/0_Introduction/matrixMul/matrixMul.cu) 与 [sample README](https://github.com/WangZW928/cuda-samples/blob/8a9e2c830c8a336b4f48c79e5ed5837031eb8551/Samples/0_Introduction/matrixMul/README.md)：审阅了提供的只读本地副本；源码明确以教学清晰为目标。
- [该快照的完整 LICENSE](https://github.com/WangZW928/cuda-samples/blob/8a9e2c830c8a336b4f48c79e5ed5837031eb8551/LICENSE)：已读取并核对与现有 `vector_add_intro/LICENSE.NVIDIA` 文本一致，本章随附完整副本 [LICENSE.NVIDIA](LICENSE.NVIDIA)，包含原有 CUDA EULA 链接。

实质改编仅限 `MatrixMulCUDA` 的二维线程映射、沿 K 分块、两个 shared 数组和前后两次 block 同步。改用直观的行/列索引，加入非整块边界补零、朴素版、有限小尺寸参数、确定性输入、CPU 参考与输出重置；中文说明和独立 CMake 按本课程风格编写。未移植其计时循环、异步 stream、pinned memory、设备选择、helper 框架或 profiler 调用；也不继承上游 README 的平台、架构或 Toolkit 支持声明。

改编源码保留 **Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.** 和完整原始许可头。分发本章改编源码或二进制时应随附适用许可声明；该声明针对 NVIDIA 衍生材料，不改变本课程其他内容的许可状态。
