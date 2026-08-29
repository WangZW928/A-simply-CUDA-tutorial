#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void vector_add(const float* a, const float* b, float* c, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = a[idx];
    float y = b[idx];
#pragma nounroll
    for (int i = 0; i < iters; ++i) {
        x = x * 1.000001f + y * 0.999999f;
        y = y * 1.0000001f + 0.000001f;
    }
    c[idx] = x + y;
}

struct HostBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* out = nullptr;
};

struct DeviceBuffers {
    float* a0 = nullptr;
    float* b0 = nullptr;
    float* c0 = nullptr;
    float* a1 = nullptr;
    float* b1 = nullptr;
    float* c1 = nullptr;
};

static double mean(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static void run_default_once(const HostBuffers& host,
                             const DeviceBuffers& dev,
                             size_t chunk_bytes,
                             int chunk_elems,
                             int blocks,
                             int threads,
                             int iters)
{
    CUDA_CHECK(cudaMemcpy(dev.a0, host.a, chunk_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.b0, host.b, chunk_bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(dev.a0, dev.b0, dev.c0, chunk_elems, iters);
    CUDA_CHECK(cudaMemcpy(host.out, dev.c0, chunk_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(dev.a1, host.a + chunk_elems, chunk_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.b1, host.b + chunk_elems, chunk_bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(dev.a1, dev.b1, dev.c1, chunk_elems, iters);
    CUDA_CHECK(cudaMemcpy(host.out + chunk_elems, dev.c1, chunk_bytes, cudaMemcpyDeviceToHost));
}

static void run_two_streams_once(const HostBuffers& host,
                                 const DeviceBuffers& dev,
                                 size_t chunk_bytes,
                                 int chunk_elems,
                                 int blocks,
                                 int threads,
                                 int iters,
                                 cudaStream_t s0,
                                 cudaStream_t s1)
{
    CUDA_CHECK(cudaMemcpyAsync(dev.a0, host.a, chunk_bytes, cudaMemcpyHostToDevice, s0));
    CUDA_CHECK(cudaMemcpyAsync(dev.b0, host.b, chunk_bytes, cudaMemcpyHostToDevice, s0));
    vector_add<<<blocks, threads, 0, s0>>>(dev.a0, dev.b0, dev.c0, chunk_elems, iters);
    CUDA_CHECK(cudaMemcpyAsync(host.out, dev.c0, chunk_bytes, cudaMemcpyDeviceToHost, s0));

    CUDA_CHECK(cudaMemcpyAsync(dev.a1, host.a + chunk_elems, chunk_bytes, cudaMemcpyHostToDevice, s1));
    CUDA_CHECK(cudaMemcpyAsync(dev.b1, host.b + chunk_elems, chunk_bytes, cudaMemcpyHostToDevice, s1));
    vector_add<<<blocks, threads, 0, s1>>>(dev.a1, dev.b1, dev.c1, chunk_elems, iters);
    CUDA_CHECK(cudaMemcpyAsync(host.out + chunk_elems, dev.c1, chunk_bytes, cudaMemcpyDeviceToHost, s1));
}

int main(int argc, char** argv) {
    std::string out_csv = "../results/stream_benchmark.csv";
    int repeats = 30;
    int warmup = 5;
    int chunk_elems = 1 << 20;
    int iters = 512;

    if (argc >= 2) out_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) chunk_elems = std::max(1024, std::atoi(argv[4]));
    if (argc >= 6) iters = std::max(1, std::atoi(argv[5]));

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int chunks = 2;
    const int total_elems = chunks * chunk_elems;
    const size_t chunk_bytes = static_cast<size_t>(chunk_elems) * sizeof(float);
    const size_t total_bytes = static_cast<size_t>(total_elems) * sizeof(float);
    const int threads = 256;
    const int blocks = (chunk_elems + threads - 1) / threads;

    HostBuffers host;
    CUDA_CHECK(cudaMallocHost(&host.a, total_bytes));
    CUDA_CHECK(cudaMallocHost(&host.b, total_bytes));
    CUDA_CHECK(cudaMallocHost(&host.out, total_bytes));
    std::fill_n(host.a, total_elems, 1.0f);
    std::fill_n(host.b, total_elems, 2.0f);
    std::fill_n(host.out, total_elems, 0.0f);

    DeviceBuffers dev;
    CUDA_CHECK(cudaMalloc(&dev.a0, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&dev.b0, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&dev.c0, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&dev.a1, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&dev.b1, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&dev.c1, chunk_bytes));

    cudaStream_t s0, s1, timing_stream;
    CUDA_CHECK(cudaStreamCreate(&s0));
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&timing_stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t done0 = nullptr;
    cudaEvent_t done1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&done0));
    CUDA_CHECK(cudaEventCreate(&done1));

    std::vector<float> default_times;
    std::vector<float> stream_times;
    default_times.reserve(repeats);
    stream_times.reserve(repeats);

    for (int i = 0; i < warmup; ++i) {
        run_default_once(host, dev, chunk_bytes, chunk_elems, blocks, threads, iters);
        CUDA_CHECK(cudaDeviceSynchronize());

        run_two_streams_once(host, dev, chunk_bytes, chunk_elems, blocks, threads, iters, s0, s1);
        CUDA_CHECK(cudaStreamSynchronize(s0));
        CUDA_CHECK(cudaStreamSynchronize(s1));
    }

    // one stream
    for (int r = 0; r < repeats; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        run_default_once(host, dev, chunk_bytes, chunk_elems, blocks, threads, iters);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        default_times.push_back(ms);
    }

    // two streams
    for (int r = 0; r < repeats; ++r) {
        CUDA_CHECK(cudaStreamSynchronize(s0));
        CUDA_CHECK(cudaStreamSynchronize(s1));
        CUDA_CHECK(cudaStreamSynchronize(timing_stream));

        CUDA_CHECK(cudaEventRecord(start, timing_stream));
        run_two_streams_once(host, dev, chunk_bytes, chunk_elems, blocks, threads, iters, s0, s1);
        CUDA_CHECK(cudaEventRecord(done0, s0));
        CUDA_CHECK(cudaEventRecord(done1, s1));
        CUDA_CHECK(cudaStreamWaitEvent(timing_stream, done0, 0));
        CUDA_CHECK(cudaStreamWaitEvent(timing_stream, done1, 0));
        CUDA_CHECK(cudaEventRecord(stop, timing_stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        stream_times.push_back(ms);
    }

    const double default_ms = mean(default_times);
    const double stream_ms = mean(stream_times);

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "chunk_elems=" << chunk_elems << ", repeats=" << repeats << ", iters=" << iters << "\n";
    std::cout << "asyncEngineCount=" << prop.asyncEngineCount << "\n";
    std::cout << std::fixed << std::setprecision(6)
              << "default_stream_mean_ms=" << default_ms << "\n"
              << "two_stream_mean_ms=" << stream_ms << "\n"
              << "speedup(default/two_stream)=" << (default_ms / stream_ms) << "\n";

    std::ofstream ofs(out_csv);
    ofs << "device_name,chunk_elems,repeats,warmup,iters,async_engine_count,mode,mean_ms\n";
    ofs << '"' << prop.name << "\"," << chunk_elems << "," << repeats << "," << warmup << "," << iters
        << "," << prop.asyncEngineCount << ",default," << default_ms << "\n";
    ofs << '"' << prop.name << "\"," << chunk_elems << "," << repeats << "," << warmup << "," << iters
        << "," << prop.asyncEngineCount << ",two_streams," << stream_ms << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(done0));
    CUDA_CHECK(cudaEventDestroy(done1));
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(timing_stream));
    CUDA_CHECK(cudaFree(dev.a0));
    CUDA_CHECK(cudaFree(dev.b0));
    CUDA_CHECK(cudaFree(dev.c0));
    CUDA_CHECK(cudaFree(dev.a1));
    CUDA_CHECK(cudaFree(dev.b1));
    CUDA_CHECK(cudaFree(dev.c1));
    CUDA_CHECK(cudaFreeHost(host.a));
    CUDA_CHECK(cudaFreeHost(host.b));
    CUDA_CHECK(cudaFreeHost(host.out));
    return 0;
}
