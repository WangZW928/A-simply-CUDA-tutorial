// Adapted from CSC Training's 2017 CUDA exercises: first-cuda-program,
// error-checking, first-unified-memory-program, and jacobi-timing-events.
// Copyright (c) 2017 CSC Training
// The upstream MIT notice is preserved in ../LICENSE.CSC.
// This version adds validation, explicit timing scopes, and benchmark CSV output.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                               \
            std::cerr << __FILE__ << ':' << __LINE__ << " " << #call << ": "     \
                      << cudaGetErrorString(err) << '\n';                       \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void set_index(int* a, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) a[i] = static_cast<int>(i);
}

struct Stats {
    double mean;
    double stddev;
};

static Stats summarize(const std::vector<double>& samples) {
    double mean = 0.0;
    for (double x : samples) mean += x;
    mean /= samples.size();
    double variance = 0.0;
    for (double x : samples) variance += (x - mean) * (x - mean);
    return {mean, std::sqrt(variance / samples.size())};
}

static int parse_int(const char* text, int minimum, const char* name) {
    size_t end = 0;
    const long long value = std::stoll(text, &end);
    if (text[end] != '\0' || value < minimum ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " 超出有效整数范围");
    }
    return static_cast<int>(value);
}

static void run_mode(bool managed, int n, int threads, int repeats, int warmup,
                     std::ostream& csv) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<size_t>(n) + threads - 1) / threads);
    std::vector<int> host(managed ? 0 : n);
    int* device = nullptr;
    if (managed) {
        CUDA_CHECK(cudaMallocManaged(&device, bytes));
    } else {
        CUDA_CHECK(cudaMalloc(&device, bytes));
    }
    int* result = managed ? device : host.data();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> event_times, host_times;

    // 每轮 CPU 都会读结果；测的是 CPU/GPU 交替访问，不是常驻 GPU 数据。
    for (long long r = -static_cast<long long>(warmup); r < repeats; ++r) {
        // 重置为 -1，避免上轮正确值掩盖漏写。初始化和分配不计时。
        // 上轮 stop 已完成；此时 host 才能安全访问 managed allocation。
        std::fill_n(result, n, -1);
        if (!managed) {
            CUDA_CHECK(cudaMemset(device, 0xff, bytes));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        const auto host_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(start));
        set_index<<<blocks, threads>>>(device, n);
        CUDA_CHECK(cudaGetLastError());  // 检查 launch；不等待 kernel 完成。
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));  // 等待执行并检查异步错误。

        if (!managed) {
            CUDA_CHECK(cudaMemcpy(result, device, bytes, cudaMemcpyDeviceToHost));
        }
        // managed 模式也必须在同步后由 CPU 读取，不能省略上面的等待。
        bool valid = true;
        for (int i = 0; i < n; ++i) {
            if (result[i] != i) {
                std::cerr << "校验失败: i=" << i << " actual=" << result[i] << '\n';
                valid = false;
                break;
            }
        }
        const auto host_stop = std::chrono::steady_clock::now();
        if (!valid) std::exit(EXIT_FAILURE);

        float event_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&event_ms, start, stop));
        if (r >= 0) {
            event_times.push_back(event_ms);
            host_times.push_back(
                std::chrono::duration<double, std::milli>(host_stop - host_start).count());
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device));
    const auto event_stats = summarize(event_times);
    const auto host_stats = summarize(host_times);
    const char* mode = managed ? "managed" : "explicit";
    csv << mode << ',' << n << ',' << threads << ',' << repeats << ',' << warmup
        << ',' << event_stats.mean << ',' << event_stats.stddev
        << ',' << host_stats.mean << ',' << host_stats.stddev << ",1\n";
    std::cout << mode << ": PASS, event_mean_ms=" << event_stats.mean
              << ", host_mean_ms=" << host_stats.mean << '\n';
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "basics_bench [output_csv [repeats [warmup [elements [threads]]]]]\n"
                      << "basics_bench --invalid-launch  # 故意触发 launch 错误，返回非零\n";
            return 0;
        }
        const bool invalid_launch = argc == 2 && std::string(argv[1]) == "--invalid-launch";
        if (argc > 6) throw std::invalid_argument("参数过多；使用 --help 查看用法");
        const std::string path = argc > 1 ? argv[1] : "results/basics_bench_run.csv";
        const int repeats = argc > 2 ? parse_int(argv[2], 1, "repeats") : 20;
        const int warmup = argc > 3 ? parse_int(argv[3], 0, "warmup") : 3;
        const int n = argc > 4 ? parse_int(argv[4], 1, "elements") : 1048579;
        const int threads = argc > 5 ? parse_int(argv[5], 1, "threads") : 256;

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::cout << "GPU=" << prop.name << " sm_" << prop.major << prop.minor
                  << " driver_api=" << driver << " runtime=" << runtime
                  << " managedMemory=" << prop.managedMemory
                  << " concurrentManagedAccess=" << prop.concurrentManagedAccess << '\n';

        if (invalid_launch) {
            // n=0 保证即便配置意外被接受也不会访问空指针。
            set_index<<<1, prop.maxThreadsPerBlock + 1>>>(nullptr, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cerr << "预期的 launch 错误没有出现\n";
            return EXIT_FAILURE;
        }
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0]) {
            throw std::invalid_argument("threads 超过设备的 block 限制");
        }
        const size_t blocks = (static_cast<size_t>(n) + threads - 1) / threads;
        if (blocks > static_cast<size_t>(prop.maxGridSize[0])) {
            throw std::invalid_argument("elements / threads 超过设备的 grid 限制");
        }

        const auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("无法打开 CSV: " + path);
        csv << "mode,elements,block_size,repeats,warmup,mean_ms,std_ms,host_mean_ms,host_std_ms,valid\n"
            << std::fixed << std::setprecision(6);
        run_mode(false, n, threads, repeats, warmup, csv);
        if (prop.managedMemory) {
            run_mode(true, n, threads, repeats, warmup, csv);
        } else {
            std::cout << "SKIP managed: 设备不支持 managed memory\n";
        }
        csv.close();
        if (!csv) throw std::runtime_error("写入 CSV 失败: " + path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
