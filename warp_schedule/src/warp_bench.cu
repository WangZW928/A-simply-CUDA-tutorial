#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                      << " -> " << cudaGetErrorString(err) << std::endl;            \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

__global__ void warp_stress_kernel(float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = static_cast<float>(idx) * 1e-4f;
#pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        x = x * 1.000001f + 0.00001f;
    }
    out[idx] = x;
}

struct RunStat {
    int blocks_per_sm;
    int blocks;
    int warps_per_block;
    int threads_per_block;
    int total_warps;
    int elements;
    int iterations;
    double avg_ms;
    double std_ms;
};

static double mean(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double stdev(const std::vector<float>& v, double m) {
    double acc = 0.0;
    for (float x : v) {
        double d = static_cast<double>(x) - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(v.size()));
}

int main(int argc, char** argv) {
    std::string output_csv = "results/warp_benchmark.csv";
    int repeats = 20;
    int warmup = 5;
    int iters = 2048;

    if (argc >= 2) output_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) iters = std::max(1, std::atoi(argv[4]));

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n"
              << "SM count: " << prop.multiProcessorCount << "\n"
              << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";

    const std::vector<int> blocks_per_sm_cases = {1, 2, 4, 8, 16, 24, 32, 48, 64};

    std::vector<RunStat> all_stats;

    for (int blocks_per_sm : blocks_per_sm_cases) {
        int blocks = prop.multiProcessorCount * blocks_per_sm;

        for (int warps_per_block = 1; warps_per_block <= 32; ++warps_per_block) {
            int threads_per_block = warps_per_block * 32;
            if (threads_per_block > prop.maxThreadsPerBlock) continue;

            int elements = blocks * threads_per_block;
            float* d_out = nullptr;
            CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(elements) * sizeof(float)));

            for (int i = 0; i < warmup; ++i) {
                warp_stress_kernel<<<blocks, threads_per_block>>>(d_out, elements, iters);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> times;
            times.reserve(repeats);

            for (int r = 0; r < repeats; ++r) {
                cudaEvent_t start, stop;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));

                CUDA_CHECK(cudaEventRecord(start));
                warp_stress_kernel<<<blocks, threads_per_block>>>(d_out, elements, iters);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));

                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                times.push_back(ms);

                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(stop));
            }

            double avg = mean(times);
            double sd = stdev(times, avg);

            all_stats.push_back(RunStat{
                blocks_per_sm,
                blocks,
                warps_per_block,
                threads_per_block,
                blocks * warps_per_block,
                elements,
                iters,
                avg,
                sd,
            });

            std::cout << "[blocks/SM=" << std::setw(2) << blocks_per_sm
                      << "] warps/block=" << std::setw(2) << warps_per_block
                      << "  avg=" << std::fixed << std::setprecision(4) << avg
                      << " ms  std=" << sd << " ms\n";

            CUDA_CHECK(cudaFree(d_out));
        }
    }

    std::ofstream ofs(output_csv);
    if (!ofs) {
        std::cerr << "Failed to open output CSV: " << output_csv << std::endl;
        return 1;
    }

    ofs << "device_name,sm_count,blocks_per_sm,blocks,warps_per_block,threads_per_block,total_warps,elements,iterations,repeats,warmup,avg_ms,std_ms\n";
    for (const auto& s : all_stats) {
        ofs << '"' << prop.name << '"' << ','
            << prop.multiProcessorCount << ','
            << s.blocks_per_sm << ','
            << s.blocks << ','
            << s.warps_per_block << ','
            << s.threads_per_block << ','
            << s.total_warps << ','
            << s.elements << ','
            << s.iterations << ','
            << repeats << ','
            << warmup << ','
            << std::fixed << std::setprecision(6) << s.avg_ms << ','
            << s.std_ms << '\n';
    }

    std::cout << "\nSaved benchmark CSV to: " << output_csv << std::endl;
    return 0;
}

