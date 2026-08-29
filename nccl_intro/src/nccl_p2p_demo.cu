#include "nccl_common.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

__global__ void fill_sequence(float* data, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = static_cast<float>(i);
    }
}

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " [elements]\n\n"
              << "Example:\n"
              << "  " << program << " 1048576\n";
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        const size_t elements = parse_size_arg(argc, argv, 1, 1 << 20);
        const size_t bytes = elements * sizeof(float);
        const int visible_gpus = get_device_count_or_exit();
        if (visible_gpus < 2) {
            std::cerr << "NCCL P2P demo needs at least 2 GPUs.\n";
            return 1;
        }

        std::cout << "Host: " << get_hostname() << "\n";
        std::cout << "rank 0 -> " << device_name(0) << "\n";
        std::cout << "rank 1 -> " << device_name(1) << "\n";

        ncclUniqueId id;
        ncclComm_t comms[2];
        cudaStream_t streams[2];
        CHECK_NCCL(ncclGetUniqueId(&id));

        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < 2; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            CHECK_NCCL(ncclCommInitRank(&comms[rank], 2, id, rank));
        }
        CHECK_NCCL(ncclGroupEnd());

        for (int rank = 0; rank < 2; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            CHECK_CUDA(cudaStreamCreateWithFlags(&streams[rank],
                                                 cudaStreamNonBlocking));
        }

        float* send0 = nullptr;
        float* recv1 = nullptr;
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMalloc(&send0, bytes));
        const int threads = 256;
        const int blocks = static_cast<int>((elements + threads - 1) / threads);
        fill_sequence<<<blocks, threads, 0, streams[0]>>>(send0, elements);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaMalloc(&recv1, bytes));
        CHECK_CUDA(cudaMemsetAsync(recv1, 0, bytes, streams[1]));

        CHECK_NCCL(ncclGroupStart());
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_NCCL(ncclSend(send0, elements, ncclFloat, 1, comms[0],
                            streams[0]));
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_NCCL(ncclRecv(recv1, elements, ncclFloat, 0, comms[1],
                            streams[1]));
        CHECK_NCCL(ncclGroupEnd());

        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaStreamSynchronize(streams[0]));
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaStreamSynchronize(streams[1]));

        std::vector<float> host(elements);
        CHECK_CUDA(cudaMemcpy(host.data(), recv1, bytes, cudaMemcpyDeviceToHost));
        size_t mismatches = 0;
        for (size_t i = 0; i < elements; ++i) {
            if (std::fabs(host[i] - static_cast<float>(i)) > 0.0f) {
                ++mismatches;
                if (mismatches <= 5) {
                    std::cerr << "Mismatch at " << i << ": got " << host[i]
                              << " expected " << static_cast<float>(i)
                              << "\n";
                }
            }
        }

        if (mismatches == 0) {
            std::cout << "P2P send/recv passed, bytes=" << bytes << "\n";
        } else {
            std::cerr << "P2P send/recv failed, mismatches=" << mismatches
                      << "\n";
        }

        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaFree(send0));
        CHECK_CUDA(cudaStreamDestroy(streams[0]));
        CHECK_NCCL(ncclCommDestroy(comms[0]));
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaFree(recv1));
        CHECK_CUDA(cudaStreamDestroy(streams[1]));
        CHECK_NCCL(ncclCommDestroy(comms[1]));

        return mismatches == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
}
