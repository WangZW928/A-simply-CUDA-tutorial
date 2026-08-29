# register_Occupancy：寄存器与 Occupancy 实验

本项目用于学习：**寄存器压力如何限制并发（occupancy），并影响性能**。

> 本项目是寄存器与 Occupancy 主题的基础版实验。更完整的资源约束、occupancy 阶梯、cliff 和 register spilling 分析，请继续学习 `occupancy_vs_registers`。

核心观点：

- `registers per thread` 越高，单个 SM 可同时驻留的线程/warp 往往越少
- 并发下降后，隐藏延迟能力变弱，kernel 可能变慢
- 所以“做了更多计算”不一定更快，资源瓶颈可能先出现

## 1. 实验设计

两个 kernel 版本：

- 版本 A（低寄存器）：

```cpp
float x = a[i];
```

- 版本 B（高寄存器）：

```cpp
float tmp[HIGH_REG_TMP_SIZE];
#pragma unroll
for (int k = 0; k < HIGH_REG_TMP_SIZE; k++) tmp[k] = x;
```

其中版本 B 通过局部数组 + 展开循环显著提升寄存器压力。数组长度用宏 `HIGH_REG_TMP_SIZE` 统一控制。

## 2. 你将观察什么

输出中重点关注：

- `regs_per_thread`（每线程寄存器数）
- `theoretical_occupancy`（理论 occupancy）
- `avg_ms` / `std_ms`（kernel 平均时间与波动）


## 2.1 Stat 字段说明

`src/reg_occ_bench.cu` 里的 `Stat` 结构体用于记录一次实验结果。字段含义如下：

- `kernel`：kernel 名称（`low_reg` / `high_reg`）
- `high_reg_tmp_size`：高寄存器版本临时数组长度（`HIGH_REG_TMP_SIZE`）
- `threads_per_block`：每个 block 的线程数（`blockDim.x`）
- `blocks`：grid 的 block 数（`gridDim.x`）
- `elements`：本次处理元素总数（通常是 `blocks * threads_per_block`）
- `repeats`：正式计时重复次数
- `warmup`：预热轮数（不计入统计）
- `iters`：kernel 内部循环次数（控制计算强度）
- `regs_per_thread`：每线程寄存器数（编译器分配结果）
- `shmem_static_bytes`：静态 shared memory 大小（字节）
- `max_active_blocks_per_sm`：每个 SM 理论最多可驻留 block 数
- `active_warps_per_sm`：每个 SM 当前活跃 warp 数
- `max_warps_per_sm`：硬件每个 SM 的 warp 上限
- `theoretical_occupancy`：理论占有率（`active_warps_per_sm / max_warps_per_sm`）
- `avg_ms`：平均执行时间（毫秒）
- `std_ms`：执行时间标准差（毫秒）
- `elems_per_ms`：吞吐量（每毫秒处理元素数，`elements / avg_ms`）
## 3. 项目结构

```text
.
├── CMakeLists.txt
├── src/
│   └── reg_occ_bench.cu
├── scripts/
│   └── plot_results.py
├── results/
├── requirements.txt
└── README.md
```

## 4. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/reg_occ_bench
```

自定义参数：

```bash
./build/reg_occ_bench <output_csv> <repeats> <warmup> <iters>
# 示例
./build/reg_occ_bench results/reg_occ_benchmark.csv 30 8 256
```

## 5. Python 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/reg_occ_benchmark.csv --outdir results
```

会生成：

- `results/registers_per_thread.png`
- `results/occupancy_compare.png`
- `results/runtime_compare.png`
- `results/summary_compare.csv`

## 6. 如何思考“为什么多做计算反而变慢”

可以从这条链路分析：

1. 局部变量增多 -> `regs_per_thread` 上升
2. 每个 SM 可驻留 warp 数下降 -> occupancy 下降
3. 可用于隐藏访存/流水线延迟的 warp 变少
4. 尽管算术操作变多，但总吞吐不一定提升，甚至下降

注意：如果 kernel 完全 compute-bound 且之前 occupancy 远超“够用阈值”，增加寄存器不一定显著变慢。最终要用实测数据判断。

## 7. 可扩展练习

- 修改 `tmp[64]` 为 `tmp[16]`, `tmp[32]`, `tmp[96]`，画出寄存器-occupancy-时间曲线
- 加入 `-Xptxas -v`（或查看 Nsight Compute）对比编译器寄存器分配
- 将 block size 改成 128/256/512，观察 occupancy 与性能是否一致变化


## 8. 批量 Sweep（不同寄存器压力）

项目提供脚本：`scripts/sweep_registers.sh`，用于自动完成：

- 以不同 `HIGH_REG_TMP_SIZE` 重新编译
- 运行 benchmark 并写每轮日志
- 汇总到总 CSV
- 自动调用 Python 画趋势图

### 8.1 一条命令运行

```bash
cd register_Occupancy
bash scripts/sweep_registers.sh
```

### 8.2 可选参数（环境变量）

```bash
TMP_SIZES="8 16 24 32 48 64 80 96 128" REPEATS=20 WARMUP=5 ITERS=256 bash scripts/sweep_registers.sh
```

### 8.3 结果输出

- 汇总 CSV：`results/reg_occ_sweep.csv`
- 每轮日志：`results/logs/build_run_tmp_<size>.log`
- 趋势图：
  - `results/sweep_occupancy_vs_regs.png`
  - `results/sweep_runtime_vs_regs.png`
  - `results/sweep_throughput_vs_regs.png`
- 汇总表：`results/sweep_summary.csv`

说明：趋势图默认使用 `high_reg` 行，横轴是 `regs_per_thread`，更直接反映“寄存器压力 -> occupancy/性能”的关系。

