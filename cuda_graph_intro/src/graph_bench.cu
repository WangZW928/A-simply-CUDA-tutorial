#include <cuda_runtime.h>

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

__global__ void add_bias(float* x, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += b;
}

__global__ void scale(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = 0.0f;
}

static double mean(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

int main(int argc, char** argv) {
    std::string out_csv = "results/graph_benchmark.csv";
    int repeats = 2000;
    int warmup = 200;
    int n = 1 << 20;

    if (argc >= 2) out_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) n = std::max(1024, std::atoi(argv[4]));

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> h_x(n, 1.0f);
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(float)));

    auto reset_input = [&]() {
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    };

    // Baseline: normal launches (3 kernels per iteration)
    reset_input();
    for (int i = 0; i < warmup; ++i) {
        add_bias<<<blocks, threads>>>(d_x, 0.1f, n);
        scale<<<blocks, threads>>>(d_x, 1.01f, n);
        relu<<<blocks, threads>>>(d_x, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> t_normal;
    t_normal.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        cudaEvent_t st, ed;
        CUDA_CHECK(cudaEventCreate(&st));
        CUDA_CHECK(cudaEventCreate(&ed));
        CUDA_CHECK(cudaEventRecord(st));
        add_bias<<<blocks, threads>>>(d_x, 0.1f, n);
        scale<<<blocks, threads>>>(d_x, 1.01f, n);
        relu<<<blocks, threads>>>(d_x, n);
        CUDA_CHECK(cudaEventRecord(ed));
        CUDA_CHECK(cudaEventSynchronize(ed));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed));
        t_normal.push_back(ms);
        CUDA_CHECK(cudaEventDestroy(st));
        CUDA_CHECK(cudaEventDestroy(ed));
    }

    // Graph path: capture once, replay many times
    reset_input();
    
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    add_bias<<<blocks, threads, 0, s>>>(d_x, 0.1f, n);
    scale<<<blocks, threads, 0, s>>>(d_x, 1.01f, n);
    relu<<<blocks, threads, 0, s>>>(d_x, n);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<float> t_graph;
    t_graph.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        cudaEvent_t st, ed;
        CUDA_CHECK(cudaEventCreate(&st));
        CUDA_CHECK(cudaEventCreate(&ed));
        CUDA_CHECK(cudaEventRecord(st, s));
        CUDA_CHECK(cudaGraphLaunch(graph_exec, s));
        CUDA_CHECK(cudaEventRecord(ed, s));
        CUDA_CHECK(cudaEventSynchronize(ed));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed));
        t_graph.push_back(ms);
        CUDA_CHECK(cudaEventDestroy(st));
        CUDA_CHECK(cudaEventDestroy(ed));
    }

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(d_x));

    double normal_mean = mean(t_normal);
    double graph_mean = mean(t_graph);
    double speedup = normal_mean / graph_mean;

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "n=" << n << ", repeats=" << repeats << ", warmup=" << warmup << "\n";
    std::cout << std::fixed << std::setprecision(6)
              << "normal_mean_ms=" << normal_mean << "\n"
              << "graph_mean_ms=" << graph_mean << "\n"
              << "speedup(normal/graph)=" << speedup << "\n";

    std::ofstream ofs(out_csv);
    ofs << "device_name,n,repeats,warmup,mode,mean_ms\n";
    ofs << '"' << prop.name << "\"," << n << "," << repeats << "," << warmup << ",normal," << normal_mean << "\n";
    ofs << '"' << prop.name << "\"," << n << "," << repeats << "," << warmup << ",graph," << graph_mean << "\n";
    return 0;
}
