#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(status));                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CHECK_NCCL(call)                                                       \
    do {                                                                       \
        ncclResult_t status = (call);                                          \
        if (status != ncclSuccess) {                                           \
            std::fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__,          \
                         __LINE__, ncclGetErrorString(status));                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

inline std::string get_hostname() {
    char name[256];
    if (gethostname(name, sizeof(name)) != 0) {
        return "unknown";
    }
    name[sizeof(name) - 1] = '\0';
    return std::string(name);
}

inline int get_device_count_or_exit() {
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    if (count <= 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        std::exit(EXIT_FAILURE);
    }
    return count;
}

inline std::string device_name(int device) {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    return std::string(prop.name);
}

inline int parse_int_arg(int argc, char** argv, int index, int default_value) {
    if (argc <= index) {
        return default_value;
    }
    char* end = nullptr;
    long value = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || *end != '\0' || value <= 0) {
        throw std::runtime_error("Invalid positive integer argument: " +
                                 std::string(argv[index]));
    }
    return static_cast<int>(value);
}

inline size_t parse_size_arg(int argc, char** argv, int index,
                             size_t default_value) {
    if (argc <= index) {
        return default_value;
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(argv[index], &end, 10);
    if (end == argv[index] || *end != '\0' || value == 0) {
        throw std::runtime_error("Invalid positive size argument: " +
                                 std::string(argv[index]));
    }
    return static_cast<size_t>(value);
}

class CpuTimer {
public:
    void start() { begin_ = clock::now(); }

    double elapsed_ms() const {
        const auto end = clock::now();
        return std::chrono::duration<double, std::milli>(end - begin_).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point begin_{clock::now()};
};
