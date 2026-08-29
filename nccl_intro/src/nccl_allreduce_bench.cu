#include "nccl_common.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

__global__ void fill_kernel(float* data, size_t n, float value) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}

__global__ void scale_kernel(float* data, size_t n, float scale) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * scale + 1.0f;
    }
}

struct Args {
    std::string output_csv = "results/nccl_allreduce.csv";
    int requested_gpus = 0;
    int repeats = 20;
    int warmup = 5;
    size_t min_bytes = 1 << 10;
    size_t max_bytes = 1 << 26;
    bool overlap_demo = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    if (argc > 1) {
        args.output_csv = argv[1];
    }
    args.requested_gpus = parse_int_arg(argc, argv, 2, 0);
    args.repeats = parse_int_arg(argc, argv, 3, args.repeats);
    args.warmup = parse_int_arg(argc, argv, 4, args.warmup);
    args.min_bytes = parse_size_arg(argc, argv, 5, args.min_bytes);
    args.max_bytes = parse_size_arg(argc, argv, 6, args.max_bytes);
    if (argc > 7) {
        args.overlap_demo = std::string(argv[7]) == "--overlap";
    }
    if (args.min_bytes > args.max_bytes) {
        throw std::runtime_error("min_bytes must be <= max_bytes");
    }
    return args;
}

void print_usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " [output_csv] [num_gpus] [repeats] [warmup] [min_bytes] [max_bytes] [--overlap]\n\n"
        << "Example:\n"
        << "  " << program << " results/nccl_allreduce.csv 4 30 5 1024 268435456\n"
        << "  " << program << " results/nccl_overlap.csv 2 20 5 1048576 67108864 --overlap\n";
}

double mean_ms(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

void initialize_buffers(const std::vector<float*>& sendbuff,
                        const std::vector<float*>& recvbuff, size_t elements,
                        const std::vector<cudaStream_t>& streams) {
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    for (size_t rank = 0; rank < sendbuff.size(); ++rank) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(rank)));
        fill_kernel<<<blocks, threads, 0, streams[rank]>>>(
            sendbuff[rank], elements, static_cast<float>(rank + 1));
        fill_kernel<<<blocks, threads, 0, streams[rank]>>>(recvbuff[rank],
                                                           elements, 0.0f);
        CHECK_CUDA(cudaGetLastError());
    }
}

void run_one_allreduce(const std::vector<ncclComm_t>& comms,
                       const std::vector<float*>& sendbuff,
                       const std::vector<float*>& recvbuff, size_t elements,
                       const std::vector<cudaStream_t>& streams,
                       bool overlap_demo) {
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);

    if (overlap_demo) {
        for (size_t rank = 0; rank < comms.size(); ++rank) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(rank)));
            scale_kernel<<<blocks, threads, 0, streams[rank]>>>(
                sendbuff[rank], elements, 1.00001f);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    CHECK_NCCL(ncclGroupStart());
    for (size_t rank = 0; rank < comms.size(); ++rank) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(rank)));
        CHECK_NCCL(ncclAllReduce(sendbuff[rank], recvbuff[rank], elements,
                                 ncclFloat, ncclSum, comms[rank],
                                 streams[rank]));
    }
    CHECK_NCCL(ncclGroupEnd());

    if (overlap_demo) {
        for (size_t rank = 0; rank < comms.size(); ++rank) {
            CHECK_CUDA(cudaSetDevice(static_cast<int>(rank)));
            scale_kernel<<<blocks, threads, 0, streams[rank]>>>(
                recvbuff[rank], elements, 0.5f);
            CHECK_CUDA(cudaGetLastError());
        }
    }
}

void synchronize_all(const std::vector<cudaStream_t>& streams) {
    for (size_t rank = 0; rank < streams.size(); ++rank) {
        CHECK_CUDA(cudaSetDevice(static_cast<int>(rank)));
        CHECK_CUDA(cudaStreamSynchronize(streams[rank]));
    }
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        const Args args = parse_args(argc, argv);
        const int visible_gpus = get_device_count_or_exit();
        const int num_gpus = args.requested_gpus > 0
                                 ? std::min(args.requested_gpus, visible_gpus)
                                 : visible_gpus;
        if (num_gpus < 1) {
            std::cerr << "NCCL AllReduce demo needs at least 1 GPU.\n";
            return 1;
        }

        std::cout << "Host: " << get_hostname() << "\n";
        std::cout << "Using " << num_gpus << " GPU(s)\n";
        for (int rank = 0; rank < num_gpus; ++rank) {
            std::cout << "  rank " << rank << " -> " << device_name(rank)
                      << "\n";
        }

        std::vector<ncclComm_t> comms(num_gpus);
        std::vector<cudaStream_t> streams(num_gpus);
        ncclUniqueId id;
        CHECK_NCCL(ncclGetUniqueId(&id));

        CHECK_NCCL(ncclGroupStart());
        for (int rank = 0; rank < num_gpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            CHECK_NCCL(ncclCommInitRank(&comms[rank], num_gpus, id, rank));
        }
        CHECK_NCCL(ncclGroupEnd());

        for (int rank = 0; rank < num_gpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            CHECK_CUDA(cudaStreamCreateWithFlags(&streams[rank],
                                                 cudaStreamNonBlocking));
        }

        std::ofstream csv(args.output_csv);
        if (!csv) {
            throw std::runtime_error("Cannot open output CSV: " +
                                     args.output_csv);
        }
        csv << "message_bytes,elements,num_gpus,mode,mean_ms,algbw_gb_s,busbw_gb_s\n";

        for (size_t bytes = args.min_bytes; bytes <= args.max_bytes;
             bytes *= 2) {
            const size_t elements = std::max<size_t>(1, bytes / sizeof(float));
            const size_t actual_bytes = elements * sizeof(float);

            std::vector<float*> sendbuff(num_gpus, nullptr);
            std::vector<float*> recvbuff(num_gpus, nullptr);
            for (int rank = 0; rank < num_gpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(rank));
                CHECK_CUDA(cudaMalloc(&sendbuff[rank], actual_bytes));
                CHECK_CUDA(cudaMalloc(&recvbuff[rank], actual_bytes));
            }
            initialize_buffers(sendbuff, recvbuff, elements, streams);
            synchronize_all(streams);

            for (int i = 0; i < args.warmup; ++i) {
                run_one_allreduce(comms, sendbuff, recvbuff, elements, streams,
                                  args.overlap_demo);
            }
            synchronize_all(streams);

            std::vector<double> samples;
            samples.reserve(args.repeats);
            for (int i = 0; i < args.repeats; ++i) {
                CpuTimer timer;
                timer.start();
                run_one_allreduce(comms, sendbuff, recvbuff, elements, streams,
                                  args.overlap_demo);
                synchronize_all(streams);
                samples.push_back(timer.elapsed_ms());
            }

            const double ms = mean_ms(samples);
            const double seconds = ms / 1.0e3;
            const double algbw = static_cast<double>(actual_bytes) / seconds /
                                 1.0e9;
            const double busbw =
                algbw * (2.0 * static_cast<double>(num_gpus - 1) /
                         static_cast<double>(num_gpus));
            const char* mode = args.overlap_demo ? "stream_overlap"
                                                 : "allreduce";
            csv << actual_bytes << "," << elements << "," << num_gpus << ","
                << mode << "," << std::fixed << std::setprecision(6) << ms
                << "," << algbw << "," << busbw << "\n";

            std::cout << "bytes=" << actual_bytes << " mean_ms=" << ms
                      << " algbw=" << algbw << " GB/s busbw=" << busbw
                      << " GB/s\n";

            for (int rank = 0; rank < num_gpus; ++rank) {
                CHECK_CUDA(cudaSetDevice(rank));
                CHECK_CUDA(cudaFree(sendbuff[rank]));
                CHECK_CUDA(cudaFree(recvbuff[rank]));
            }
        }

        for (int rank = 0; rank < num_gpus; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            CHECK_CUDA(cudaStreamDestroy(streams[rank]));
            CHECK_NCCL(ncclCommDestroy(comms[rank]));
        }
        std::cout << "Saved CSV: " << args.output_csv << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
}
