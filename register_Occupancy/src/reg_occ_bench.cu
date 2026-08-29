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

#ifndef HIGH_REG_TMP_SIZE
#define HIGH_REG_TMP_SIZE 64
#endif

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                      << " -> " << cudaGetErrorString(err) << std::endl;            \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

__global__ void kernel_low_reg(const float* a, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = a[i];
#pragma unroll 4
    for (int t = 0; t < iters; ++t) {
        x = x * 1.00001f + 0.0001f;
    }
    out[i] = x;
}

__global__ void kernel_high_reg(const float* a, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = a[i];
    float tmp[HIGH_REG_TMP_SIZE];
#pragma unroll
    for (int k = 0; k < HIGH_REG_TMP_SIZE; ++k) {
        tmp[k] = x + static_cast<float>(k) * 1e-6f;
    }

#pragma unroll 4
    for (int t = 0; t < iters; ++t) {
#pragma unroll
        for (int k = 0; k < HIGH_REG_TMP_SIZE; ++k) {
            tmp[k] = tmp[k] * 1.000001f + 0.000001f;
        }
    }

    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < HIGH_REG_TMP_SIZE; ++k) {
        acc += tmp[k];
    }
    out[i] = acc;
}

struct Stat {
    std::string kernel;
    int high_reg_tmp_size;
    int threads_per_block;
    int blocks;
    int elements;
    int repeats;
    int warmup;
    int iters;
    int regs_per_thread;
    size_t shmem_static_bytes;
    int max_active_blocks_per_sm;
    int active_warps_per_sm;
    int max_warps_per_sm;
    double theoretical_occupancy;
    double avg_ms;
    double std_ms;
    double elems_per_ms;
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

template <typename KernelT>
Stat run_one(const std::string& name,
             KernelT kernel,
             const float* d_in,
             float* d_out,
             int n,
             int threads_per_block,
             int blocks,
             int repeats,
             int warmup,
             int iters,
             int max_warps_per_sm) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));

    int active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, kernel, threads_per_block, 0));

    int warps_per_block = threads_per_block / 32;
    int active_warps_per_sm = active_blocks_per_sm * warps_per_block;
    double occupancy = static_cast<double>(active_warps_per_sm) / static_cast<double>(max_warps_per_sm);

    for (int i = 0; i < warmup; ++i) {
        kernel<<<blocks, threads_per_block>>>(d_in, d_out, n, iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<blocks, threads_per_block>>>(d_in, d_out, n, iters);
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
    double elems_per_ms = static_cast<double>(n) / avg;

    return Stat{
        name,
        HIGH_REG_TMP_SIZE,
        threads_per_block,
        blocks,
        n,
        repeats,
        warmup,
        iters,
        attr.numRegs,
        static_cast<size_t>(attr.sharedSizeBytes),
        active_blocks_per_sm,
        active_warps_per_sm,
        max_warps_per_sm,
        occupancy,
        avg,
        sd,
        elems_per_ms,
    };
}

int main(int argc, char** argv) {
    std::string output_csv = "results/reg_occ_benchmark.csv";
    int repeats = 20;
    int warmup = 5;
    int iters = 256;

    if (argc >= 2) output_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) iters = std::max(1, std::atoi(argv[4]));

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int threads_per_block = 256;
    int blocks = prop.multiProcessorCount * 8;
    int n = blocks * threads_per_block;

    std::vector<float> h_in(n, 1.0f);
    float* d_in = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max warps/SM: " << prop.maxThreadsPerMultiProcessor / 32 << "\n";
    std::cout << "HIGH_REG_TMP_SIZE: " << HIGH_REG_TMP_SIZE << "\n\n";

    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;

    Stat low = run_one("low_reg", kernel_low_reg, d_in, d_out, n, threads_per_block, blocks,
                       repeats, warmup, iters, max_warps_per_sm);
    Stat high = run_one("high_reg", kernel_high_reg, d_in, d_out, n, threads_per_block, blocks,
                        repeats, warmup, iters, max_warps_per_sm);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    auto print_stat = [](const Stat& s) {
        std::cout << "[" << s.kernel << "] "
                  << "regs/thread=" << s.regs_per_thread
                  << ", occ=" << std::fixed << std::setprecision(2) << s.theoretical_occupancy * 100.0 << "%"
                  << ", avg=" << std::setprecision(4) << s.avg_ms << " ms"
                  << ", elems/ms=" << s.elems_per_ms
                  << ", std=" << s.std_ms << " ms\n";
    };

    print_stat(low);
    print_stat(high);

    std::ofstream ofs(output_csv);
    if (!ofs) {
        std::cerr << "Failed to open output CSV: " << output_csv << std::endl;
        return 1;
    }

    ofs << "device_name,sm_count,kernel,high_reg_tmp_size,threads_per_block,blocks,elements,repeats,warmup,iters,regs_per_thread,shmem_static_bytes,max_active_blocks_per_sm,active_warps_per_sm,max_warps_per_sm,theoretical_occupancy,avg_ms,std_ms,elems_per_ms\n";

    auto dump = [&](const Stat& s) {
        ofs << '"' << prop.name << '"' << ','
            << prop.multiProcessorCount << ','
            << s.kernel << ','
            << s.high_reg_tmp_size << ','
            << s.threads_per_block << ','
            << s.blocks << ','
            << s.elements << ','
            << s.repeats << ','
            << s.warmup << ','
            << s.iters << ','
            << s.regs_per_thread << ','
            << s.shmem_static_bytes << ','
            << s.max_active_blocks_per_sm << ','
            << s.active_warps_per_sm << ','
            << s.max_warps_per_sm << ','
            << std::fixed << std::setprecision(6) << s.theoretical_occupancy << ','
            << s.avg_ms << ','
            << s.std_ms << ','
            << s.elems_per_ms << '\n';
    };

    dump(low);
    dump(high);

    std::cout << "\nSaved CSV: " << output_csv << std::endl;
    return 0;
}
