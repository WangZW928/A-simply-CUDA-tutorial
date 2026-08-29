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

__global__ void stride_read_kernel(const float* in, float* out, int elements, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    out[idx] = in[idx * stride] * 1.000001f + 1.0f;
}

__global__ void offset_read_kernel(const float* in, float* out, int elements, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    out[idx] = in[idx + offset] * 1.000001f + 1.0f;
}

struct Result {
    std::string experiment;
    std::string param_name;
    int param_value;
    int elements;
    int repeats;
    int warmup;
    double mean_ms;
    double std_ms;
    double requested_bytes;
    double requested_bandwidth_gb_s;
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
Result benchmark_case(const std::string& experiment,
                      const std::string& param_name,
                      int param_value,
                      int elements,
                      int repeats,
                      int warmup,
                      double requested_bytes,
                      Launcher launch) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
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
    double bandwidth = requested_bytes / (avg * 1.0e6);

    return Result{
        experiment,
        param_name,
        param_value,
        elements,
        repeats,
        warmup,
        avg,
        sd,
        requested_bytes,
        bandwidth,
    };
}

int main(int argc, char** argv) {
    std::string out_csv = "results/memory_coalescing.csv";
    int repeats = 50;
    int warmup = 10;
    int elements = 1 << 20;
    int max_stride = 32;

    if (argc >= 2) out_csv = argv[1];
    if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) warmup = std::max(0, std::atoi(argv[3]));
    if (argc >= 5) elements = std::max(1024, std::atoi(argv[4]));
    if (argc >= 6) max_stride = std::max(1, std::atoi(argv[5]));

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;
    const int max_offset = 32;
    const int input_elements = elements * max_stride + max_offset + 1;// Protect Data load
    const size_t input_bytes = static_cast<size_t>(input_elements) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(elements) * sizeof(float);
    const double requested_bytes = static_cast<double>(elements) * sizeof(float) * 2.0; // load and store

    std::vector<float> h_input(input_elements, 1.0f);
    for (int i = 0; i < input_elements; ++i) {
        h_input[i] = static_cast<float>(i % 251) * 0.01f;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    std::vector<Result> results;

    std::vector<int> strides;
    for (int s = 1; s <= max_stride; s *= 2) {
        strides.push_back(s); // 1,2,4,8,16,32~2^5
    }
    if (strides.back() != max_stride) {
        strides.push_back(max_stride);
    }

    for (int stride : strides) {
        auto launch = [&]() {
            stride_read_kernel<<<blocks, threads>>>(d_input, d_output, elements, stride);
        };
        results.push_back(benchmark_case(
            "stride_sweep", "stride", stride, elements, repeats, warmup, requested_bytes, launch));
    }

    for (int offset = 0; offset <= max_offset; ++offset) {
        auto launch = [&]() {
            offset_read_kernel<<<blocks, threads>>>(d_input, d_output, elements, offset);
        };
        results.push_back(benchmark_case(
            "offset_sweep", "offset", offset, elements, repeats, warmup, requested_bytes, launch));
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::ofstream ofs(out_csv);
    if (!ofs) {
        std::cerr << "Failed to open output CSV: " << out_csv << std::endl;
        return 1;
    }

    ofs << "device_name,sm_count,experiment,param_name,param_value,elements,repeats,warmup,"
           "mean_ms,std_ms,requested_bytes,requested_bandwidth_gb_s\n";
    for (const auto& r : results) {
        ofs << '"' << prop.name << '"' << ','
            << prop.multiProcessorCount << ','
            << r.experiment << ','
            << r.param_name << ','
            << r.param_value << ','
            << r.elements << ','
            << r.repeats << ','
            << r.warmup << ','
            << std::fixed << std::setprecision(6)
            << r.mean_ms << ','
            << r.std_ms << ','
            << r.requested_bytes << ','
            << r.requested_bandwidth_gb_s << '\n';
    }

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "elements=" << elements << ", repeats=" << repeats
              << ", warmup=" << warmup << ", max_stride=" << max_stride << "\n";
    std::cout << "Saved CSV: " << out_csv << "\n";
    return 0;
}
