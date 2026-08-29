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

__global__ void compute_bound_kernel(const float* a, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float y = x * 0.5f + 1.0f;
#pragma unroll 4
    for (int t = 0; t < iters; ++t) {
        x = fmaf(x, 1.000001f, y);
        y = fmaf(y, 0.999999f, x * 0.000001f);
    }
    out[i] = x + y;
}

__global__ void memory_bound_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + 1.7f * b[i];
}

__global__ void latency_bound_kernel(const int* next, float* out, int n, int steps) {
    int lane = threadIdx.x & 31;
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int warps_total = (gridDim.x * blockDim.x) >> 5;
    int idx = (warp_global * 131 + lane) & (n - 1);
    float acc = 0.0f;
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
        idx = next[idx];
        acc += static_cast<float>(idx & 255) * 0.001f;
    }
    if (lane == 0 && warp_global < warps_total) out[warp_global] = acc;
}

__global__ void launch_overhead_kernel(float* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] += 1.0f;
}

struct Result {
    std::string experiment;
    std::string kernel_type;
    std::string mode;
    int elements;
    int block_size;
    int blocks;
    int repeats;
    int warmup;
    int iters_or_launches;
    int streams;
    double mean_ms;
    double std_ms;
    double throughput_units_per_ms;
    double effective_gb_s;
    double theoretical_occupancy;
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
static Result benchmark(const std::string& experiment,
                        const std::string& kernel_type,
                        const std::string& mode,
                        int elements,
                        int block_size,
                        int blocks,
                        int repeats,
                        int warmup,
                        int iters_or_launches,
                        int streams,
                        double work_units,
                        double bytes,
                        double occupancy,
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
    return {experiment, kernel_type, mode, elements, block_size, blocks, repeats, warmup,
            iters_or_launches, streams, avg, sd, work_units / avg, bytes / (avg * 1.0e6), occupancy};
}

template <typename KernelT>
static double occupancy_for(KernelT kernel, int block_size, const cudaDeviceProp& prop) {
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, block_size, 0));
    int active_warps = active_blocks * ((block_size + 31) / 32);
    return static_cast<double>(active_warps) / static_cast<double>(prop.maxThreadsPerMultiProcessor / 32);
}

int main(int argc, char** argv) {
    std::string out_csv = "results/kernel_type_benchmark.csv";
    int repeats = 25;
    int warmup = 5;
    int elements = 1 << 22;
    int compute_iters = 1024;
    int latency_steps = 512;
    int tiny_launches = 1000;

    if (argc >= 2) out_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) elements = std::max(1024, std::atoi(argv[4]));
    if (argc >= 6) compute_iters = std::max(1, std::atoi(argv[5]));
    if (argc >= 7) latency_steps = std::max(1, std::atoi(argv[6]));
    if (argc >= 8) tiny_launches = std::max(1, std::atoi(argv[7]));

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int pow2_elements = 1;
    while (pow2_elements < elements) pow2_elements <<= 1;
    elements = pow2_elements;

    std::vector<float> h_a(elements, 1.0f), h_b(elements, 2.0f);
    std::vector<int> h_next(elements);
    for (int i = 0; i < elements; ++i) h_next[i] = (i * 1664525 + 1013904223) & (elements - 1);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    int* d_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(elements) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(elements) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(elements) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next, static_cast<size_t>(elements) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), static_cast<size_t>(elements) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), static_cast<size_t>(elements) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next, h_next.data(), static_cast<size_t>(elements) * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<Result> results;
    std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
    for (int block_size : block_sizes) {
        if (block_size > prop.maxThreadsPerBlock) continue;
        int blocks = (elements + block_size - 1) / block_size;

        results.push_back(benchmark("block_size_sweep", "compute_bound", "single_stream", elements,
                                    block_size, blocks, repeats, warmup, compute_iters, 1,
                                    static_cast<double>(elements) * compute_iters * 4.0,
                                    static_cast<double>(elements) * sizeof(float) * 2.0,
                                    occupancy_for(compute_bound_kernel, block_size, prop), [&]() {
                                        compute_bound_kernel<<<blocks, block_size>>>(d_a, d_c, elements, compute_iters);
                                    }));
        results.push_back(benchmark("block_size_sweep", "memory_bound", "single_stream", elements,
                                    block_size, blocks, repeats, warmup, 1, 1,
                                    static_cast<double>(elements),
                                    static_cast<double>(elements) * sizeof(float) * 3.0,
                                    occupancy_for(memory_bound_kernel, block_size, prop), [&]() {
                                        memory_bound_kernel<<<blocks, block_size>>>(d_a, d_b, d_c, elements);
                                    }));
        results.push_back(benchmark("block_size_sweep", "latency_bound", "single_stream", elements,
                                    block_size, blocks, repeats, warmup, latency_steps, 1,
                                    static_cast<double>(blocks * block_size / 32) * latency_steps,
                                    static_cast<double>(blocks * block_size / 32) * latency_steps * sizeof(int),
                                    occupancy_for(latency_bound_kernel, block_size, prop), [&]() {
                                        latency_bound_kernel<<<blocks, block_size>>>(d_next, d_c, elements, latency_steps);
                                    }));
    }

    std::vector<int> blocks_per_sm_cases = {1, 2, 4, 8, 16};
    int occ_block = 256;
    for (int bpsm : blocks_per_sm_cases) {
        int blocks = prop.multiProcessorCount * bpsm;
        int n = std::min(elements, blocks * occ_block);
        results.push_back(benchmark("occupancy_sweep", "compute_bound", "blocks_per_sm", n,
                                    occ_block, blocks, repeats, warmup, compute_iters, 1,
                                    static_cast<double>(n) * compute_iters * 4.0,
                                    static_cast<double>(n) * sizeof(float) * 2.0,
                                    occupancy_for(compute_bound_kernel, occ_block, prop), [&]() {
                                        compute_bound_kernel<<<blocks, occ_block>>>(d_a, d_c, n, compute_iters);
                                    }));
        results.push_back(benchmark("occupancy_sweep", "memory_bound", "blocks_per_sm", n,
                                    occ_block, blocks, repeats, warmup, 1, 1,
                                    static_cast<double>(n),
                                    static_cast<double>(n) * sizeof(float) * 3.0,
                                    occupancy_for(memory_bound_kernel, occ_block, prop), [&]() {
                                        memory_bound_kernel<<<blocks, occ_block>>>(d_a, d_b, d_c, n);
                                    }));
        results.push_back(benchmark("occupancy_sweep", "latency_bound", "blocks_per_sm", n,
                                    occ_block, blocks, repeats, warmup, latency_steps, 1,
                                    static_cast<double>(blocks * occ_block / 32) * latency_steps,
                                    static_cast<double>(blocks * occ_block / 32) * latency_steps * sizeof(int),
                                    occupancy_for(latency_bound_kernel, occ_block, prop), [&]() {
                                        latency_bound_kernel<<<blocks, occ_block>>>(d_next, d_c, elements, latency_steps);
                                    }));
    }

    cudaStream_t s0, s1;
    CUDA_CHECK(cudaStreamCreate(&s0));
    CUDA_CHECK(cudaStreamCreate(&s1));
    int half = elements / 2;
    int stream_block = 256;
    int stream_blocks = (half + stream_block - 1) / stream_block;
    results.push_back(benchmark("stream_compare", "memory_bound", "single_stream_two_chunks", elements,
                                stream_block, stream_blocks * 2, repeats, warmup, 1, 1,
                                static_cast<double>(elements),
                                static_cast<double>(elements) * sizeof(float) * 3.0,
                                occupancy_for(memory_bound_kernel, stream_block, prop), [&]() {
                                    memory_bound_kernel<<<stream_blocks, stream_block>>>(d_a, d_b, d_c, half);
                                    memory_bound_kernel<<<stream_blocks, stream_block>>>(d_a + half, d_b + half, d_c + half, half);
                                }));
    results.push_back(benchmark("stream_compare", "memory_bound", "two_streams_two_chunks", elements,
                                stream_block, stream_blocks * 2, repeats, warmup, 1, 2,
                                static_cast<double>(elements),
                                static_cast<double>(elements) * sizeof(float) * 3.0,
                                occupancy_for(memory_bound_kernel, stream_block, prop), [&]() {
                                    memory_bound_kernel<<<stream_blocks, stream_block, 0, s0>>>(d_a, d_b, d_c, half);
                                    memory_bound_kernel<<<stream_blocks, stream_block, 0, s1>>>(d_a + half, d_b + half, d_c + half, half);
                                    CUDA_CHECK(cudaStreamSynchronize(s0));
                                    CUDA_CHECK(cudaStreamSynchronize(s1));
                                }));
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaStreamDestroy(s1));

    float* d_tiny = d_c;
    int graph_block = 1;
    int graph_blocks = 1;
    auto normal_launches = [&]() {
        for (int i = 0; i < tiny_launches; ++i) launch_overhead_kernel<<<1, 1>>>(d_tiny);
    };
    results.push_back(benchmark("graph_compare", "launch_overhead", "normal_many_launches", 1,
                                graph_block, graph_blocks, repeats, warmup, tiny_launches, 1,
                                static_cast<double>(tiny_launches), sizeof(float),
                                occupancy_for(launch_overhead_kernel, graph_block, prop), normal_launches));

    cudaStream_t gs;
    CUDA_CHECK(cudaStreamCreate(&gs));
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < tiny_launches; ++i) launch_overhead_kernel<<<1, 1, 0, gs>>>(d_tiny);
    CUDA_CHECK(cudaStreamEndCapture(gs, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    auto graph_launch = [&]() {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, gs));
        CUDA_CHECK(cudaStreamSynchronize(gs));
    };
    results.push_back(benchmark("graph_compare", "launch_overhead", "cuda_graph_replay", 1,
                                graph_block, graph_blocks, repeats, warmup, tiny_launches, 1,
                                static_cast<double>(tiny_launches), sizeof(float),
                                occupancy_for(launch_overhead_kernel, graph_block, prop), graph_launch));
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(gs));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_next));

    std::ofstream ofs(out_csv);
    if (!ofs) {
        std::cerr << "Failed to open output CSV: " << out_csv << std::endl;
        return 1;
    }
    ofs << "device_name,sm_count,experiment,kernel_type,mode,elements,block_size,blocks,repeats,warmup,"
           "iters_or_launches,streams,mean_ms,std_ms,throughput_units_per_ms,effective_gb_s,theoretical_occupancy\n";
    for (const auto& r : results) {
        ofs << '"' << prop.name << '"' << ',' << prop.multiProcessorCount << ','
            << r.experiment << ',' << r.kernel_type << ',' << r.mode << ','
            << r.elements << ',' << r.block_size << ',' << r.blocks << ',' << r.repeats << ','
            << r.warmup << ',' << r.iters_or_launches << ',' << r.streams << ','
            << std::fixed << std::setprecision(6) << r.mean_ms << ',' << r.std_ms << ','
            << r.throughput_units_per_ms << ',' << r.effective_gb_s << ',' << r.theoretical_occupancy << '\n';
    }

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "elements=" << elements << ", compute_iters=" << compute_iters
              << ", latency_steps=" << latency_steps << ", tiny_launches=" << tiny_launches << "\n";
    std::cout << "Saved CSV: " << out_csv << "\n";
    return 0;
}
