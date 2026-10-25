# Vector Add 入门：走通 CPU → GPU → CPU

这是 [CUDA 基础实验](../cuda_basics_intro/README.md) 与 [CUDA Stream](../cuda_stream_intro/README.md) 之间的一份可选小练习。基础章的 `a[i] = i` 不需要 H2D 输入；这里用 NVIDIA CUDA Samples 的经典向量加法补齐**两个输入上传 → GPU 相加 → 结果回读 → CPU 校验**。

本章属于 A-simply-CUDA-tutorial 的学习路径，只改编一个 sample。已理解显式内存流程的读者可以直接进入 Stream 章；这里不重复 Unified Memory、event 计时或 profiler 实验。

## 1. 读懂数据流

[src/vector_add_demo.cu](src/vector_add_demo.cu) 计算 `C[i] = A[i] + B[i]`，默认 50000 个 `float`、每 block 256 个线程。

1. CPU 用普通 `std::vector<float>` 初始化 `h_a/h_b`，用 `cudaMalloc` 分配 `d_a/d_b/d_c`。
2. 两次 `cudaMemcpyHostToDevice` 将 A、B 上传；拷贝大小是 `n * sizeof(float)` 字节。
3. 默认 stream 上启动 kernel，每线程处理一个元素。
4. 等待完成，再用 `cudaMemcpyDeviceToHost` 将 C 拷回 `h_c`。
5. CPU 逐元素对照 `h_a[i] + h_b[i]`，释放设备内存。

`h_*` 与 `d_*` 是不同分配。CPU 不能直接解引用这里的 `cudaMalloc` 指针；普通 host 数组也不能直接当作本例 kernel 的设备输入。这与基础章的 `cudaMallocManaged` 路径不同。

输入确定且数值范围较小，两者之和始终大于零；输出先清零，便于发现漏写。校验拒绝 NaN/Inf，并使用绝对误差 `1e-5`；该阈值针对本例，不是所有浮点算法的通用准则。CUDA API 失败立即报错并以非零状态退出；这个短生命周期示例不实现库级的错误恢复。

## 2. 两个容易混淆的地方

**线程数与元素数不同。** block 数为 `ceil(n / 256)`，例如 `n=257` 会启动 512 个线程，只有前 257 个访问数组。kernel 保留 `i < n` 边界检查；改变输入长度时无需凑成 block 的整数倍。

**提交成功与执行完成不同。** `cudaGetLastError()` 检查 launch 错误但不等待执行；`cudaDeviceSynchronize()` 在本例中显式等待并检查执行错误，便于教学定位。随后的同步 D2H `cudaMemcpy` 本身也会等到结果拷贝完成才返回；同一默认 stream 的这条流程通常不需要额外的设备同步来保证回读正确。不要把这里的调试同步照搬进多 stream 性能实验。

同步语义可对照 [CUDA 12.6 Runtime API 文档](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/api-sync-behavior.html)。

## 3. 构建与运行

需要 CUDA Toolkit、兼容驱动和一张 NVIDIA GPU，以及 CMake 3.24+、支持 C++17 的 host 编译器。不依赖 CUDA Samples 的 `Common/helper_cuda.h`，也不需要下载完整上游仓库。

以下是供后续验证的命令，**本次集成未执行构建或运行**。从仓库根目录进入本章；按项目现有 CUDA 12.6 / RTX 4060 Laptop（sm_89）环境显式指定架构：

```bash
cd vector_add_intro
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
./build/vector_add_demo
./build/vector_add_demo 1
./build/vector_add_demo 255
./build/vector_add_demo 256
./build/vector_add_demo 257
```

省略架构选项时默认使用 `native`，配置阶段需要能访问 GPU。其他 GPU 请使用对应架构；WSL 工具链与驱动说明见[基础章](../cuda_basics_intro/README.md#2-编译并运行)。

唯一可选参数是正整数元素数（不超过 `INT_MAX`，实际还受可用内存限制）。预期成功时打印线程配置和 `PASS`、返回 0；非法参数、CUDA 错误或数值校验失败返回非零。以上描述是程序行为约定，不是已通过的运行记录。

本例没有计时、CSV 或带宽/加速比输出。一次 `PASS` 只说明本次输入的数值校验通过，不证明无越界或性能优越。计时范围继续参考基础章；异步拷贝与 overlap 继续参考 Stream 章。

## 4. 接下来学什么

- [CUDA 基础](../cuda_basics_intro/README.md)：回顾显式内存与 Unified Memory，以及两种计时范围。
- [Matrix Mul](../matrix_mul_intro/README.md)（可选）：沿用本例数据流，学习行主序矩阵乘法、二维线程、尾块边界与共享内存复用，再进入 Stream。
- [CUDA Stream](../cuda_stream_intro/README.md)：将本例的完整拷贝流程扩展为 pinned memory、分块和多队列。
- [Memory Coalescing](../memory_coalescing_intro/README.md)：从相邻线程访问相邻元素，走向 stride/offset 实验。
- [返回学习导航](../README.md#学习顺序)。

## 5. 来源与改编范围

审阅的用户仓库快照为 [WangZW928/cuda-samples @ 8a9e2c8](https://github.com/WangZW928/cuda-samples/tree/8a9e2c830c8a336b4f48c79e5ed5837031eb8551)，其 README 面向 CUDA 12.9；本章只采用基础 Runtime API，不继承整库构建配置，也不宣称已验证 CUDA 12.6 兼容性。

- 用户仓库原文件：[Samples/0_Introduction/vectorAdd/vectorAdd.cu](https://github.com/WangZW928/cuda-samples/blob/8a9e2c830c8a336b4f48c79e5ed5837031eb8551/Samples/0_Introduction/vectorAdd/vectorAdd.cu)。
- NVIDIA 规范来源：[同提交下的 vectorAdd.cu](https://github.com/NVIDIA/cuda-samples/blob/8a9e2c830c8a336b4f48c79e5ed5837031eb8551/Samples/0_Introduction/vectorAdd/vectorAdd.cu)。两仓库该文件的 Git blob 均为 `38b043effa914562a9e9b56abf0c9270b68aaa08`，内容一致。
- 完整许可：[NVIDIA 同提交 LICENSE](https://github.com/NVIDIA/cuda-samples/blob/8a9e2c830c8a336b4f48c79e5ed5837031eb8551/LICENSE)，与用户仓库版本一致，原文保存在 [LICENSE.NVIDIA](LICENSE.NVIDIA)。

改编保留了逐元素加法、默认规模、显式内存拷贝、边界保护与 CPU 校验；改用 `std::vector`、确定性输入和统一错误检查，加入长度参数、输出初始化、有限值检查和用于定位错误的同步。移除了未使用的 helper 头文件和随机初始化；中文说明与独立 CMake 按本项目习惯编写。没有导入上游其他 sample、工具框架或性能结果。

源码保留 **Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.** 及原始许可头；[LICENSE.NVIDIA](LICENSE.NVIDIA) 保留上游完整 BSD 三条款式声明和 CUDA EULA 链接。分发改编源码或二进制时应随附适用声明。此许可说明针对本章 NVIDIA 衍生材料，不改变主仓库其他内容的许可状态。
