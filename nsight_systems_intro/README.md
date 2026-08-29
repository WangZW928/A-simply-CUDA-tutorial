# Nsight Systems Intro：用时间线验证 CUDA Stream 重叠

这个项目不新增 CUDA 源码，而是复用 `../cuda_stream_intro/build/stream_bench`，用 Nsight Systems 观察默认 stream 和两个显式 stream 的执行时间线。

核心目标：

- 看清 `cudaMemcpy`、`cudaMemcpyAsync`、kernel launch 在时间轴上的位置
- 验证 H2D / kernel / D2H 是否真的发生 overlap
- 把“CSV 显示 two_streams 更快”推进到“时间线证明它为什么更快”

## 1. Nsight Systems 看什么

Nsight Systems 适合看系统级调度：

- CPU 线程什么时候调用 CUDA Runtime API
- GPU 上 kernel 和 memcpy 什么时候开始、什么时候结束
- 多个 stream 是否并发执行
- CPU launch 间隔、同步点、空洞是否明显

它不是单 kernel 指标工具。寄存器数、warp stall、访存事务等细节要交给 Nsight Compute。

## 2. 构建前置 stream benchmark

脚本会自动构建 `cuda_stream_intro`，也可以手动执行：

```bash
cd ../cuda_stream_intro
cmake -S . -B build
cmake --build build -j
```

## 3. 运行 nsys profile

```bash
cd nsight_systems_intro
bash scripts/run_nsys.sh
```

可用环境变量调整 workload：

```bash
REPEATS=10 WARMUP=2 CHUNK_ELEMS=1048576 ITERS=512 OUT_NAME=stream_overlap bash scripts/run_nsys.sh
```

脚本会生成：

- `results/stream_overlap.nsys-rep`
- `results/stream_profile_input.csv`

打开 GUI：

```bash
nsys-ui results/stream_overlap.nsys-rep
```

如果只是想保留老命令名，也可以运行：

```bash
bash scripts/profile_streams.sh
```

## 4. 时间线应该怎么看

打开报告后重点看 CUDA 行：

- CUDA API：CPU 端调用，比如 `cudaMemcpyAsync`、kernel launch、`cudaEventRecord`
- CUDA GPU Kernels：真正跑在 GPU 上的 kernel
- CUDA Memory：H2D / D2H 拷贝
- CUDA Streams：不同 stream 的队列关系

对默认 stream：

- chunk0 的 H2D、kernel、D2H 基本串行
- chunk1 要等 chunk0 完成后才继续
- 时间线呈现一条长串排队

对 two streams：

- stream0 和 stream1 上各自有一组 H2D -> kernel -> D2H
- 如果硬件和 workload 允许，会看到不同 stream 的拷贝和 kernel 在时间上重叠
- GPU timeline 上的空洞应减少，总 wall time 应下降

## 5. 怎么验证 copy/compute overlap

不要只看 CSV 的 speedup。更可靠的判断顺序是：

1. CSV 中 `two_streams` 的 `mean_ms` 小于 `default`
2. nsys 时间线中 H2D/D2H 和 kernel 在不同 stream 上有横向重叠
3. CPU 侧没有明显同步把异步队列提前阻塞
4. `cudaDeviceProp.asyncEngineCount` 不为 0，说明设备有 copy engine 支持

如果 CSV 更快但时间线没有重叠，可能是测量噪声或 workload 变化造成的。  
如果时间线有重叠但加速不明显，可能是 kernel 太重、拷贝占比太小，或 PCIe/内存瓶颈限制了收益。

## 6. 可视化 CSV

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/stream_profile_input.csv --outdir results
```

输出：

- `results/stream_profile_timing.png`
- `results/stream_profile_summary.csv`
- `results/summary.txt`

## 7. 常见 nsys 命令解释

脚本核心命令：

```bash
nsys profile \
  --trace=cuda,osrt,nvtx \
  --sample=none \
  --force-overwrite=true \
  --output=results/stream_overlap \
  ../cuda_stream_intro/build/stream_bench results/stream_profile_input.csv 10 2 1048576 512
```

参数含义：

- `--trace=cuda,osrt,nvtx`：记录 CUDA API/GPU 活动、OS runtime 和 NVTX
- `--sample=none`：关闭 CPU 采样，减少报告噪声
- `--force-overwrite=true`：允许覆盖旧报告
- `--output`：输出 `.nsys-rep` 报告前缀

## 8. 你应该学到什么

- Stream 是组织 GPU 工作的队列，不是单 kernel 加速开关
- `cudaMemcpyAsync` 只是异步提交，是否重叠必须用 timeline 验证
- Nsight Systems 回答的是“工作如何排队、是否并发、哪里等待”
- 性能实验要同时看数值结果和时间线，否则容易误判

一句话记忆：

Nsight Systems 是 CUDA 程序的时间线显微镜；它让 stream overlap 从猜测变成证据。
