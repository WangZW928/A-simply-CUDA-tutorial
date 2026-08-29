#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSms = 24;
constexpr int kMaxWarpsPerSm = 48;
constexpr int kWarpSize = 32;
constexpr int kWavesPerSm = 96;
constexpr int kDefaultRepeats = 20;
constexpr int kDefaultWarmup = 5;
constexpr int kIters = 256;
constexpr size_t kElements = 1u << 20;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <int REG_TMP_SIZE>
__global__ void sweep_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             float* __restrict__ sink,
                             size_t n,
                             int iters) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        float x = in[i];
        float s = x * 1.000001f + 0.25f;

        if constexpr (REG_TMP_SIZE > 0) {
            volatile float tmp[REG_TMP_SIZE];
#pragma unroll
            for (int k = 0; k < REG_TMP_SIZE; ++k) {
                tmp[k] = s + static_cast<float>(k) * 0.03125f;
            }

            for (int iter = 0; iter < iters; ++iter) {
                s = fmaf(s, 1.000000119f, tmp[0] * 0.0000009536743164f);
                s = fmaf(s, 0.999999881f,
                         tmp[(REG_TMP_SIZE - 1) / 2] * 0.0000004768371582f);
                s += tmp[REG_TMP_SIZE - 1] * 0.0000002384185791f;
            }

#pragma unroll
            for (int k = 0; k < REG_TMP_SIZE; ++k) {
                s += tmp[k] * 0.0000152587890625f;
            }
        } else {
            for (int iter = 0; iter < iters; ++iter) {
                s = fmaf(s, 1.000000119f, 0.000001f * static_cast<float>(iter & 7));
                s = s - 0.0000005f * x;
            }
        }

        out[i] = s;

        // Data-dependent and practically never taken; keeps the dummy work live.
        if (s == sink[0] + 1234567.0f) {
            sink[threadIdx.x & 31] = s;
        }
    }
}

template <int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(256, MIN_BLOCKS_PER_SM)
void launch_bounds_kernel(const float* __restrict__ in,
                          float* __restrict__ out,
                          float* __restrict__ sink,
                          size_t n,
                          int iters) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    // 48 个活跃浮点 -> 自然寄存器需求 ~100+，让 __launch_bounds__ 的
    // minnctapersm 约束真正“咬合”（编译器被迫压寄存器/溢出）。
    for (size_t i = tid; i < n; i += stride) {
        float x = in[i];
        float a[48];
#pragma unroll
        for (int k = 0; k < 48; ++k) {
            a[k] = x * (0.01f + 0.001f * static_cast<float>(k)) + 0.25f;
        }

        for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
            for (int k = 0; k < 48; ++k) {
                a[k] = fmaf(a[k], 1.0000001f,
                            a[(k + 1) & 47] * 0.000001f);
            }
        }

        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 48; ++k) {
            s += a[k];
        }
        out[i] = s;
        if (s == sink[0] - 987654.0f) {
            sink[threadIdx.x & 31] = s;
        }
    }
}

struct Result {
    std::string experiment;
    int reg_tmp_size;
    int launch_bounds_min_blocks;
    int block_size;
    int regs_per_thread;
    int active_blocks_per_sm;
    double theoretical_occupancy;
    int grid_blocks;
    size_t elements;
    int iters;
    int repeats;
    int warmup;
    float avg_ms;
    double throughput_gel_s;
};

template <typename Kernel>
Result run_one(const std::string& experiment,
               int reg_tmp_size,
               int launch_bounds_min_blocks,
               int block_size,
               Kernel kernel,
               const float* d_in,
               float* d_out,
               float* d_sink,
               size_t n,
               int iters,
               int repeats,
               int warmup) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));

    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, block_size, 0));

    const int active_warps = active_blocks * (block_size / kWarpSize);
    const double occupancy = static_cast<double>(active_warps) /
                             static_cast<double>(kMaxWarpsPerSm);
    const int grid_blocks = kSms * kWavesPerSm;

    // 当 regs/thread * blockDim 超过每 SM 64K 寄存器堆时，该配置无法启动：
    // occupancy API 返回 0 个 active block，直接跳过计时（避免 launch 报错）。
    if (active_blocks == 0) {
        std::cout << "[skip] " << experiment << " reg_tmp=" << reg_tmp_size
                  << " block=" << block_size
                  << " regs/thread=" << attr.numRegs
                  << " (regs x threads exceed 64K register file; cannot launch)"
                  << std::endl;
        return Result{experiment,
                      reg_tmp_size,
                      launch_bounds_min_blocks,
                      block_size,
                      attr.numRegs,
                      0,
                      0.0,
                      grid_blocks,
                      n,
                      iters,
                      repeats,
                      warmup,
                      -1.0f,  // sentinel: skipped, no timing
                      0.0};
    }

    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid_blocks, block_size>>>(d_in, d_out, d_sink, n, iters);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        kernel<<<grid_blocks, block_size>>>(d_in, d_out, d_sink, n, iters);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    const float avg_ms = total_ms / static_cast<float>(repeats);
    const double throughput = (static_cast<double>(n) / (avg_ms * 1.0e-3)) / 1.0e9;

    return Result{experiment,
                  reg_tmp_size,
                  launch_bounds_min_blocks,
                  block_size,
                  attr.numRegs,
                  active_blocks,
                  occupancy,
                  grid_blocks,
                  n,
                  iters,
                  repeats,
                  warmup,
                  avg_ms,
                  throughput};
}

template <int REG_TMP_SIZE>
void run_sweep_variant(std::ofstream& csv,
                       const std::vector<int>& block_sizes,
                       const float* d_in,
                       float* d_out,
                       float* d_sink,
                       int repeats,
                       int warmup) {
    for (int block_size : block_sizes) {
        Result r = run_one("sweep",
                           REG_TMP_SIZE,
                           0,
                           block_size,
                           sweep_kernel<REG_TMP_SIZE>,
                           d_in,
                           d_out,
                           d_sink,
                           kElements,
                           kIters,
                           repeats,
                           warmup);
        if (r.avg_ms < 0.0f) continue;  // skipped (un-launchable config)
        csv << r.experiment << ',' << r.reg_tmp_size << ','
            << r.launch_bounds_min_blocks << ',' << r.block_size << ','
            << r.regs_per_thread << ',' << r.active_blocks_per_sm << ','
            << std::setprecision(8) << r.theoretical_occupancy << ','
            << r.grid_blocks << ',' << r.elements << ',' << r.iters << ','
            << r.repeats << ',' << r.warmup << ',' << std::fixed
            << std::setprecision(6) << r.avg_ms << ',' << std::setprecision(9)
            << r.throughput_gel_s << '\n';
        csv.flush();

        std::cout << "[sweep] tmp=" << std::setw(3) << REG_TMP_SIZE
                  << " block=" << std::setw(4) << block_size
                  << " regs/thread=" << std::setw(4) << r.regs_per_thread
                  << " active_blocks/SM=" << std::setw(2) << r.active_blocks_per_sm
                  << " occ=" << std::fixed << std::setprecision(3)
                  << r.theoretical_occupancy
                  << " avg_ms=" << std::setprecision(4) << r.avg_ms
                  << " throughput=" << std::setprecision(3) << r.throughput_gel_s
                  << " GE/s" << std::endl;
    }
}

template <int MIN_BLOCKS_PER_SM>
void run_launch_bounds_variant(std::ofstream& csv,
                               const float* d_in,
                               float* d_out,
                               float* d_sink,
                               int repeats,
                               int warmup) {
    constexpr int block_size = 256;
    // launch_bounds 内核计算强度更高（48 路 FMA 链），用更少的 iters 保持时长可比。
    const int iters = kIters / 4;
    Result r = run_one("launch_bounds",
                       0,
                       MIN_BLOCKS_PER_SM,
                       block_size,
                       launch_bounds_kernel<MIN_BLOCKS_PER_SM>,
                       d_in,
                       d_out,
                       d_sink,
                       kElements,
                       iters,
                       repeats,
                       warmup);
    if (r.avg_ms < 0.0f) return;  // skipped (un-launchable config)
    csv << r.experiment << ',' << r.reg_tmp_size << ','
        << r.launch_bounds_min_blocks << ',' << r.block_size << ','
        << r.regs_per_thread << ',' << r.active_blocks_per_sm << ','
        << std::setprecision(8) << r.theoretical_occupancy << ','
        << r.grid_blocks << ',' << r.elements << ',' << r.iters << ','
        << r.repeats << ',' << r.warmup << ',' << std::fixed
        << std::setprecision(6) << r.avg_ms << ',' << std::setprecision(9)
        << r.throughput_gel_s << '\n';
    csv.flush();

    std::cout << "[launch_bounds] min_blocks=" << std::setw(2)
              << MIN_BLOCKS_PER_SM
              << " regs/thread=" << std::setw(4) << r.regs_per_thread
              << " active_blocks/SM=" << std::setw(2) << r.active_blocks_per_sm
              << " occ=" << std::fixed << std::setprecision(3)
              << r.theoretical_occupancy
              << " avg_ms=" << std::setprecision(4) << r.avg_ms
              << " throughput=" << std::setprecision(3) << r.throughput_gel_s
              << " GE/s" << std::endl;
}

void initialize_input(float* d_in, float* d_out, float* d_sink) {
    std::vector<float> h_in(kElements);
    for (size_t i = 0; i < h_in.size(); ++i) {
        h_in[i] = static_cast<float>((i % 1024) + 1) * 0.001f;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, h_in.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sink, 0, 32 * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <output.csv> [repeats=" << kDefaultRepeats
                  << "] [warmup=" << kDefaultWarmup << "]\n";
        return EXIT_FAILURE;
    }

    const std::string output_path = argv[1];
    const int repeats = (argc >= 3) ? std::atoi(argv[2]) : kDefaultRepeats;
    const int warmup = (argc >= 4) ? std::atoi(argv[3]) : kDefaultWarmup;
    if (repeats <= 0 || warmup < 0) {
        std::cerr << "repeats must be positive and warmup must be non-negative\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device " << device << ": " << prop.name
              << ", SMs=" << prop.multiProcessorCount
              << ", cc=" << prop.major << "." << prop.minor << std::endl;

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, kElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, kElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sink, 32 * sizeof(float)));
    initialize_input(d_in, d_out, d_sink);

    std::ofstream csv(output_path);
    if (!csv) {
        std::cerr << "Failed to open output CSV: " << output_path << "\n";
        return EXIT_FAILURE;
    }
    csv << "experiment,reg_tmp_size,launch_bounds_min_blocks,block_size,"
           "regs_per_thread,active_blocks_per_sm,theoretical_occupancy,"
           "grid_blocks,elements,iters,repeats,warmup,avg_ms,throughput_gel_s\n";

    const std::vector<int> block_sizes{128, 256, 512, 1024};
    run_sweep_variant<0>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<8>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<16>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<24>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<32>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<40>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<48>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<56>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<64>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<80>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<96>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<128>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<160>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<192>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);
    run_sweep_variant<256>(csv, block_sizes, d_in, d_out, d_sink, repeats, warmup);

    run_launch_bounds_variant<1>(csv, d_in, d_out, d_sink, repeats, warmup);
    run_launch_bounds_variant<2>(csv, d_in, d_out, d_sink, repeats, warmup);
    run_launch_bounds_variant<4>(csv, d_in, d_out, d_sink, repeats, warmup);
    run_launch_bounds_variant<6>(csv, d_in, d_out, d_sink, repeats, warmup);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaDeviceReset());

    std::cout << "Wrote " << output_path << std::endl;
    return EXIT_SUCCESS;
}
