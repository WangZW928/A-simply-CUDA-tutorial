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

__global__ void block_reduce_shared_kernel(const float* in, float* block_sums, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float x = 0.0f;
    if (i < n) x += in[i];
    if (i + blockDim.x < n) x += in[i + blockDim.x];
    smem[tid] = x;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = smem[0];
}

__inline__ __device__ float warp_reduce_sum(float x) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(mask, x, offset);
    }
    return x;
}

__global__ void block_reduce_shuffle_kernel(const float* in, float* block_sums, int n) {
    extern __shared__ float warp_sums[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float x = 0.0f;
    if (i < n) x += in[i];
    if (i + blockDim.x < n) x += in[i + blockDim.x];

    x = warp_reduce_sum(x);
    if (lane == 0) warp_sums[warp] = x;
    __syncthreads();

    float block_sum = 0.0f;
    int warp_count = (blockDim.x + 31) / 32;
    if (tid < warp_count) block_sum = warp_sums[tid];
    if (warp == 0) block_sum = warp_reduce_sum(block_sum);
    if (tid == 0) block_sums[blockIdx.x] = block_sum;
}

__global__ void exclusive_scan_blelloch_kernel(const float* in, float* out, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int ai = tid;
    int bi = tid + blockDim.x;
    int elems = blockDim.x * 2;

    temp[ai] = (ai < n) ? in[ai] : 0.0f;
    temp[bi] = (bi < n) ? in[bi] : 0.0f;

    for (int offset = 1; offset < elems; offset <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < elems) temp[idx] += temp[idx - offset];
    }

    if (tid == 0) temp[elems - 1] = 0.0f;

    for (int offset = elems >> 1; offset > 0; offset >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < elems) {
            float t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
    }
    __syncthreads();

    if (ai < n) out[ai] = temp[ai];
    if (bi < n) out[bi] = temp[bi];
}

struct Result {
    std::string operation;
    std::string variant;
    int elements;
    int block_size;
    int blocks;
    int repeats;
    int warmup;
    double mean_ms;
    double std_ms;
    double effective_gb_s;
    double max_abs_error;
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

template <typename Launcher>
static Result benchmark(const std::string& operation,
                        const std::string& variant,
                        int elements,
                        int block_size,
                        int blocks,
                        int repeats,
                        int warmup,
                        double bytes,
                        double max_abs_error,
                        Launcher launch) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        launch();
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
    return {operation, variant, elements, block_size, blocks, repeats, warmup,
            avg, sd, bytes / (avg * 1.0e6), max_abs_error};
}

int main(int argc, char** argv) {
    std::string out_csv = "results/reduce_scan_benchmark.csv";
    int repeats = 30;
    int warmup = 5;
    int elements = 1 << 22;
    int block_size = 256;

    if (argc >= 2) out_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) elements = std::max(1024, std::atoi(argv[4]));
    if (argc >= 6) block_size = std::max(32, std::atoi(argv[5]));
    block_size = std::min(1024, ((block_size + 31) / 32) * 32);

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<float> h_in(elements);
    for (int i = 0; i < elements; ++i) h_in[i] = static_cast<float>((i % 17) + 1) * 0.25f;
    double cpu_sum = std::accumulate(h_in.begin(), h_in.end(), 0.0);

    float* d_in = nullptr;
    float* d_partial = nullptr;
    float* d_scan = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(elements) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), static_cast<size_t>(elements) * sizeof(float),
                          cudaMemcpyHostToDevice));

    int reduce_blocks = (elements + block_size * 2 - 1) / (block_size * 2);
    CUDA_CHECK(cudaMalloc(&d_partial, static_cast<size_t>(reduce_blocks) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan, static_cast<size_t>(std::min(elements, block_size * 2)) * sizeof(float)));

    std::vector<Result> results;
    std::vector<float> h_partial(reduce_blocks);

    auto shared_launch = [&]() {
        block_reduce_shared_kernel<<<reduce_blocks, block_size, block_size * sizeof(float)>>>(
            d_in, d_partial, elements);
    };
    shared_launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, static_cast<size_t>(reduce_blocks) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double shared_sum = std::accumulate(h_partial.begin(), h_partial.end(), 0.0);
    results.push_back(benchmark("reduction", "shared_memory_tree", elements, block_size, reduce_blocks,
                                repeats, warmup, static_cast<double>(elements) * sizeof(float),
                                std::abs(shared_sum - cpu_sum), shared_launch));

    int warp_smem_bytes = ((block_size + 31) / 32) * static_cast<int>(sizeof(float));
    auto shuffle_launch = [&]() {
        block_reduce_shuffle_kernel<<<reduce_blocks, block_size, warp_smem_bytes>>>(
            d_in, d_partial, elements);
    };
    shuffle_launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, static_cast<size_t>(reduce_blocks) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double shuffle_sum = std::accumulate(h_partial.begin(), h_partial.end(), 0.0);
    results.push_back(benchmark("reduction", "warp_shuffle", elements, block_size, reduce_blocks,
                                repeats, warmup, static_cast<double>(elements) * sizeof(float),
                                std::abs(shuffle_sum - cpu_sum), shuffle_launch));

    int scan_elements = std::min(elements, block_size * 2);
    std::vector<float> h_scan(scan_elements);
    std::vector<float> h_scan_ref(scan_elements);
    float running = 0.0f;
    for (int i = 0; i < scan_elements; ++i) {
        h_scan_ref[i] = running;
        running += h_in[i];
    }

    auto scan_launch = [&]() {
        exclusive_scan_blelloch_kernel<<<1, block_size, block_size * 2 * sizeof(float)>>>(
            d_in, d_scan, scan_elements);
    };
    scan_launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_scan.data(), d_scan, static_cast<size_t>(scan_elements) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double scan_error = 0.0;
    for (int i = 0; i < scan_elements; ++i) {
        scan_error = std::max(scan_error, std::abs(static_cast<double>(h_scan[i] - h_scan_ref[i])));
    }
    results.push_back(benchmark("scan", "blelloch_exclusive_single_block", scan_elements, block_size, 1,
                                repeats, warmup, static_cast<double>(scan_elements) * sizeof(float) * 2.0,
                                scan_error, scan_launch));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_scan));

    std::ofstream ofs(out_csv);
    if (!ofs) {
        std::cerr << "Failed to open output CSV: " << out_csv << std::endl;
        return 1;
    }
    ofs << "device_name,sm_count,operation,variant,elements,block_size,blocks,repeats,warmup,"
           "mean_ms,std_ms,effective_gb_s,max_abs_error\n";
    for (const auto& r : results) {
        ofs << '"' << prop.name << '"' << ',' << prop.multiProcessorCount << ','
            << r.operation << ',' << r.variant << ',' << r.elements << ',' << r.block_size << ','
            << r.blocks << ',' << r.repeats << ',' << r.warmup << ','
            << std::fixed << std::setprecision(6) << r.mean_ms << ',' << r.std_ms << ','
            << r.effective_gb_s << ',' << r.max_abs_error << '\n';
    }

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "elements=" << elements << ", block_size=" << block_size
              << ", reduce_blocks=" << reduce_blocks << "\n";
    std::cout << "CPU sum=" << std::fixed << std::setprecision(3) << cpu_sum
              << ", shared_error=" << std::abs(shared_sum - cpu_sum)
              << ", shuffle_error=" << std::abs(shuffle_sum - cpu_sum)
              << ", scan_error=" << scan_error << "\n";
    std::cout << "Saved CSV: " << out_csv << "\n";
    return 0;
}
