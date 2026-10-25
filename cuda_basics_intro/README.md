# CUDA 基础实验：从第一个 kernel 到可信的计时

这是进入本仓库性能实验前的一小段准备。先让程序**算对、报错清楚、计时范围明确**，再去解释 stream、访存和 profiler 的结果。已熟悉这些内容可以直接进入 [CUDA Stream](../cuda_stream_intro/README.md)。

前置知识：C/C++ 数组、指针、循环和基本命令行。运行需要 CUDA Toolkit、兼容驱动和一张可用 NVIDIA GPU；构建使用 CMake 3.24+ 和支持 C++17 的 host 编译器。不依赖 Python、MPI、NCCL，也不需要多 GPU。

## 1. 实验做什么

[basics_bench.cu](src/basics_bench.cu) 只有一个 kernel：`a[i] = i`。用两种内存方式完成相同工作，逐元素检查结果，再输出 CSV：

| 模式 | 分配与结果回读 | 要观察的问题 |
| --- | --- | --- |
| `explicit` | `cudaMalloc` + host 数组 + D2H `cudaMemcpy` | 哪个指针能在 CPU 上解引用？拷贝是否计时？ |
| `managed` | `cudaMallocManaged` + 同步后 CPU 读取 | 共用指针是否意味着没有数据移动成本？ |

这里不需要 H2D 输入拷贝，因为 kernel 直接覆盖所有有效元素。每轮先把结果重置为 `-1`，防止上轮结果掩盖漏写；分配、重置和释放不计入测量。Managed Memory 不支持时会明确输出 `SKIP managed`，只生成 explicit 行。

## 2. 编译并运行

以下命令均从本章节目录执行：

```bash
nvcc --version
nvidia-smi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/basics_bench
```

无参数运行会将本机结果写到 `results/basics_bench_run.csv`，避免覆盖仓库附带的参考数据 `results/basics_benchmark.csv`。

WSL 中如果 `nvidia-smi` 不在 PATH，可以尝试 `/usr/lib/wsl/lib/nvidia-smi`。`nvidia-smi` 的 CUDA Version 表示驱动支持的 CUDA 版本，实际编译器版本看 `nvcc --version`。

默认使用 `CMAKE_CUDA_ARCHITECTURES=native`。如果在没有 GPU 的机器上只做编译，或要指定目标架构，显式设置，例如 RTX 4060 的计算能力为 8.9：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

也可以先绕过 CMake，理解一次完整编译命令（`sm_89` 应换成目标 GPU 的架构，且需要当前 Toolkit 支持）：

```bash
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo -Xcompiler=-Wall,-Wextra \
  src/basics_bench.cu -o build/basics_bench
```

`nvcc` 组织 host/device 代码的编译与链接；`-lineinfo` 保留设备源码行信息，便于错误定位。这里不使用改变优化行为的 `-G` 做性能测量，也不沿用旧课程的 `sm_30`。

参数形式与其他章节一样采用位置参数：

```bash
./build/basics_bench <output_csv> <repeats> <warmup> <elements> <threads>
# 默认：20 次测量、3 次预热、1048579 个 int、每 block 256 个线程
./build/basics_bench results/basics_benchmark.csv 20 3 1048579 256
# 先验证边界：一个元素，以及刚好跨过一个 block 的长度
./build/basics_bench results/one.csv 2 1 1 256
./build/basics_bench results/tail.csv 2 1 257 256
```

程序打印 GPU、架构、driver API/runtime 版本以及 managed memory 能力；保存实验结果时也应记录这些输出、编译命令和输入规模。`PASS` 表示预热和正式测量的每次结果都经过逐元素验证，失败返回非零。原仓库其他章节的历史 CSV 不会被改写。

## 3. 从线程索引理解边界

一维索引是 `blockIdx.x * blockDim.x + threadIdx.x`。block 数向上取整：`(n + threads - 1) / threads`；最后一个 block 可能包含多余线程，因此要检查 `i < n`。

阅读代码后完成这三个小实验：

1. 用 `n=1、255、256、257` 运行，计算各自启动的线程总数，解释为什么都应通过校验。
2. 把 threads 从 128 改到 256，在相同 n 下比较。不要用只有几百个元素的结果推断大规模访存性能。
3. 在临时副本中删掉 `i < n`，用非整倍数 n 和下一节的 Compute Sanitizer 检查越界；修复后再测性能。普通运行没有崩溃不能证明没有越界。

## 4. 错误检查有两个时间点

Runtime API 的返回值都应检查。kernel launch 使用 `<<<...>>>`，因此采用：

```cpp
set_index<<<blocks, threads>>>(device, n);
CUDA_CHECK(cudaGetLastError());       // launch 配置等即时错误，不等待执行
// ... 在同一 stream 记录 stop ...
CUDA_CHECK(cudaEventSynchronize(stop)); // 等待完成，也检查异步执行错误
```

`CUDA_CHECK` 打印调用、源码位置和 CUDA 错误字符串后退出。`cudaGetLastError()` 会读取并清除当前 last-error 状态，但它不是同步操作。调试时可以在 launch 后增加带错误检查的 `cudaDeviceSynchronize()` 帮助定位；进入多 stream 实验后，要依据依赖关系选择同步范围。

可控的错误练习（预期返回非零，且不会写 CSV）：

```bash
./build/basics_bench --invalid-launch
```

它故意使用超过设备 `maxThreadsPerBlock` 的 block，通常会在 `cudaGetLastError()` 报 `invalid configuration argument`。不要靠删除错误检查让程序“继续成功”。

内存检查：

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build/basics_bench results/memcheck.csv 2 1 257 256
```

CSC 旧练习中的 `cuda-memcheck` 已被 Compute Sanitizer 替代，并在 CUDA 12.0 移除；这里使用新的命令。参见 [CUDA 12.0 Release Notes](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/index.html)。Sanitizer 下的耗时只用于调试，不作为性能数据。

## 5. 两把计时尺各自包含什么

两种模式都使用默认 stream，event 顺序为 `start → kernel → stop`，等待 stop 完成后才读取时间。正式测量前先预热，CSV 保留各模式的均值和总体标准差。

| CSV 字段 | 范围 |
| --- | --- |
| `mode,elements,block_size,repeats,warmup` | 实验配置 |
| `mean_ms,std_ms` | CUDA event 区间的均值和标准差；显式内存模式不包含 D2H |
| `host_mean_ms,host_std_ms` | CPU 从提交 start 前，到等待、结果回读和完整 CPU 校验结束的时间 |
| `valid` | 所有轮次通过校验才写入 `1`；失败退出，部分 CSV 不视为完整结果 |

```text
重置数据（不计时）
CPU clock 开始
    record start → kernel → record stop → 等待 stop
    显式内存 D2H / Managed Memory 的 CPU 回读
    CPU 逐元素校验
CPU clock 结束
```

Host 时间包含 event API 和校验成本，但不包含初始化、分配、CSV 输出、释放，因此是本实验定义的端到端区间，不是整个进程运行时间。它与 event 时间不能互换，也不能简单相减得到纯拷贝时间。

Managed 模式的 event 区间可能包含迁移或缺页等成本，也可能有一部分数据移动发生在区间外，取决于平台；不能一概称为“纯 kernel 指令时间”。同一 event 区间也可能受到提交间隔或其他 GPU 工作的影响。参见 [CUDA Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)。

本实验每轮 CPU 重置、GPU 写入、CPU 校验，刻意测量 CPU/GPU 交替访问。预热不代表此后数据始终驻留 GPU；小输入的 launch、等待和 CPU 扫描成本也可能主导结果。它用于理解语义，不是证明哪种分配 API 总是更快。

## 6. Unified Memory 不是免费搬运

`cudaMallocManaged` 简化了地址管理，不代表数据没有迁移、没有同步要求，也不等于 stream 章节里的 pinned host memory。尤其当 `concurrentManagedAccess=0` 时，不能在 GPU 活跃期间随意从 CPU 访问 managed allocation。本实验在 stop 完成后才回读，下一轮重置时 GPU 也已完成。

本节不引入 prefetch 或 CPU/GPU 并发访问的额外条件；先掌握这个可在有限 Unified Memory 平台上运行的流程。平台差异见 [NVIDIA Unified Memory 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)。

## 7. 接下来如何进入已有实验

- [CUDA Stream](../cuda_stream_intro/README.md)：把显式内存流程扩展成 pinned memory + 异步拷贝 + 多队列。多个 stream 的计时必须等待所有相关分支完成，不能直接照搬这里的单 stream event 对。
- [Memory Coalescing](../memory_coalescing_intro/README.md)：本节相邻线程写相邻 `int`；下一步改变 stride/offset，观察访问合并。该章 `mean_ms` 的工作量与本节不同，不能直接横向比数字。
- [Nsight Systems](../nsight_systems_intro/README.md)：观察 API、等待和数据移动的时间位置；先确定测量包含什么，再谈 overlap。
- [Nsight Compute](../nsight_compute_intro/README.md)：继续复用已有访存 benchmark 分析硬件指标。本节不复制 profiler 脚本。

旧课程里的 `nvprof` / `nvvp` 是历史工具，已在 CUDA 13.0 移除；当前学习路径使用 Nsight Systems / Nsight Compute。参见 [CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)。

## 8. 来源与改编范围

本章吸收 CSC Training 的 2017 年 *Introduction to CUDA programming* 中 compiling、first-cuda-program、error-checking、first-unified-memory-program 和 jacobi-timing-events 的教学顺序及基础示例。来源为 [CSC 课程仓库](https://github.com/csc-training/cuda)，本次审阅的 [源仓库快照](https://github.com/WangZW928/csc-CUDA-c/tree/070e77735672516059a1d0fca1173be910e40e5c/exercises) 为 `070e77735672516059a1d0fca1173be910e40e5c`。

数组赋值示例经改编加入参数化边界、每轮校验、两种计时范围和本仓库风格的 CSV 输出；文字为针对本仓库学习路径重新编写。未导入 Jacobi 求解器、旧平台指令、二进制或课件。源材料版权为 **Copyright (c) 2017 CSC Training**，完整 MIT 声明保存在 [LICENSE.CSC](LICENSE.CSC)。该声明对应本章引用和改编的 CSC 材料，不改变主仓库整体的许可状态。

## 9. 本次验证记录（2026-10-25）

环境：WSL、NVIDIA GeForce RTX 4060 Laptop GPU（sm_89）、Windows 驱动 566.36、CUDA Toolkit 12.6.85、CMake 3.28.3、GCC 13.3.0。设备报告 `managedMemory=1`、`concurrentManagedAccess=0`。

[示例 CSV](results/basics_benchmark.csv) 来自 Release 构建，参数为 `20 3 1048579 256`，不是通用性能结论。复现命令（在本章节目录）：

```bash
# 本机 WSL 的库路径；普通 Linux 安装不需要照搬。
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64
cmake -S . -B /tmp/csc-cuda-basics-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-Xcompiler=-Wall,-Wextra
cmake --build /tmp/csc-cuda-basics-build -j2
/tmp/csc-cuda-basics-build/basics_bench /tmp/basics_recheck.csv 20 3 1048579 256
```

验证结果：

- 显式 `89`、全新 `native` CMake 构建及直接 `nvcc` 构建成功；新增代码开启 `-Wall,-Wextra` 未出现警告。
- 两种内存模式的默认规模校验通过；`n=1、255、256、257、1048579` 与 `threads=128、256` 的 10 组组合均通过，另验证了不预热的路径及 CSV 字段。
- `--invalid-launch` 在 `cudaGetLastError()` 报预期配置错误；零重复次数、负预热数、零长度、超限 block、非整数参数及不可写输出路径均返回非零。
- 已有 stream benchmark 以 `3 1 4096 8` 完成小规模运行；原源码的 `#pragma nounroll` 在此 Toolkit 上有警告，本次未改动其实现。
- 部分最初的运行曾报 `CUDA driver version is insufficient for CUDA runtime version`；后续在上述 WSL 库路径环境中完成构建和运行验证，未调整主机驱动。该报错本身不能用于判断 kernel 正确性。
- Compute Sanitizer **未完成检查**：本机报告 `Failed to initialize WDDM debugger interface` / `Device not supported`。被测程序的数值校验通过不等于 sanitizer 通过；本次也未采集 Nsight 硬件报告。
