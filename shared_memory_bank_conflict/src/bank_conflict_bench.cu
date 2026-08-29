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
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                      << " -> " << cudaGetErrorString(err) << std::endl;            \
            std::exit(EXIT_FAILURE);                                                 \
        }                                                                            \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kSharedBanks = 32;
constexpr int kSharedElems = 2048;

__global__ void shared_stride_kernel(float* out, int stride, int iterations) {
    __shared__ float smem[kSharedElems];

    int tid = threadIdx.x;
    int lane = tid & (kWarpSize - 1);
    int index = lane * stride;

    for (int i = tid; i < kSharedElems; i += blockDim.x) {
        smem[i] = static_cast<float>((i % 127) + 1);
    }
    __syncthreads();

    float acc = 0.0f;
    volatile float* vsmem = smem;
#pragma unroll 4
    for (int i = 0; i < iterations; ++i) {
        // Volatile keeps the repeated shared-memory load visible to the benchmark.
        acc += vsmem[index];
    }

    int global = blockIdx.x * blockDim.x + tid;
    out[global] = acc;
}

struct Result {
    int stride;
    int estimated_conflict_degree;
    int blocks;
    int threads_per_block;
    int iterations;
    int repeats;
    int warmup;
    double mean_ms;
    double std_ms;
    double shared_loads;
    double shared_loads_per_ms;
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

static int gcd_int(int a, int b) {
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int estimate_conflict_degree(int stride) {
    return gcd_int(stride, kSharedBanks);
}

static Result benchmark_stride(float* d_out,
                               int stride,
                               int blocks,
                               int threads_per_block,
                               int iterations,
                               int repeats,
                               int warmup) {
    for (int i = 0; i < warmup; ++i) {
        shared_stride_kernel<<<blocks, threads_per_block>>>(d_out, stride, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        shared_stride_kernel<<<blocks, threads_per_block>>>(d_out, stride, iterations);
        CUDA_CHECK(cudaGetLastError());
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
    double loads = static_cast<double>(blocks) * threads_per_block * iterations;

    return Result{
        stride,
        estimate_conflict_degree(stride),
        blocks,
        threads_per_block,
        iterations,
        repeats,
        warmup,
        avg,
        sd,
        loads,
        loads / avg,
    };
}

int main(int argc, char** argv) {
    std::string out_csv = "results/bank_conflict.csv";
    int repeats = 50;
    int warmup = 10;
    int iterations = 4096;
    int blocks_per_sm = 8;

    if (argc >= 2) out_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) iterations = std::max(1, std::atoi(argv[4]));
    if (argc >= 6) blocks_per_sm = std::max(1, std::atoi(argv[5]));

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int threads_per_block = kWarpSize;
    const int blocks = prop.multiProcessorCount * blocks_per_sm;
    const size_t output_bytes = static_cast<size_t>(blocks) * threads_per_block * sizeof(float);

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, output_bytes));

    std::vector<int> strides = {1, 2, 3, 4, 5, 8, 16, 32};
    std::vector<Result> results;
    results.reserve(strides.size());

    for (int stride : strides) {
        results.push_back(benchmark_stride(
            d_out, stride, blocks, threads_per_block, iterations, repeats, warmup));
    }

    CUDA_CHECK(cudaFree(d_out));

    std::ofstream ofs(out_csv);
    if (!ofs) {
        std::cerr << "Failed to open output CSV: " << out_csv << std::endl;
        return 1;
    }

    ofs << "device_name,sm_count,stride,estimated_conflict_degree,blocks,threads_per_block,"
           "iterations,repeats,warmup,mean_ms,std_ms,shared_loads,shared_loads_per_ms\n";
    for (const auto& r : results) {
        ofs << '"' << prop.name << '"' << ','
            << prop.multiProcessorCount << ','
            << r.stride << ','
            << r.estimated_conflict_degree << ','
            << r.blocks << ','
            << r.threads_per_block << ','
            << r.iterations << ','
            << r.repeats << ','
            << r.warmup << ','
            << std::fixed << std::setprecision(6)
            << r.mean_ms << ','
            << r.std_ms << ','
            << r.shared_loads << ','
            << r.shared_loads_per_ms << '\n';
    }

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "iterations=" << iterations << ", repeats=" << repeats
              << ", warmup=" << warmup << ", blocks_per_sm=" << blocks_per_sm << "\n";
    std::cout << "Saved CSV: " << out_csv << "\n";
    return 0;
}
