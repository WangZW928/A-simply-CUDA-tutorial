/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Adapted from NVIDIA CUDA Samples vectorAdd, via WangZW928/cuda-samples.
// Source snapshot: 8a9e2c830c8a336b4f48c79e5ed5837031eb8551
// 本仓库改编：确定性输入、可选长度、统一报错及显式同步；移除 helper_cuda 依赖。
// 完整上游仓库许可见 ../LICENSE.NVIDIA；本示例不测量性能。
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
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

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    try {
        int n = 50000;  // 沿用上游默认长度；不是 256 的整数倍。
        if (argc > 2) {
            throw std::invalid_argument("用法：vector_add_demo [elements]");
        }
        if (argc == 2) {
            if (argv[1][0] < '0' || argv[1][0] > '9') {
                throw std::invalid_argument(
                    "elements 必须是 1..INT_MAX 的纯十进制正整数");
            }
            size_t end = 0;
            long long value = 0;
            try {
                value = std::stoll(argv[1], &end);
            } catch (const std::exception&) {
                throw std::invalid_argument(
                    "elements 必须是 1..INT_MAX 的纯十进制正整数");
            }
            if (argv[1][end] != '\0' || value < 1 ||
                value > std::numeric_limits<int>::max()) {
                throw std::invalid_argument(
                    "elements 必须是 1..INT_MAX 的纯十进制正整数");
            }
            n = static_cast<int>(value);
        }

        // h_* 是 CPU 普通内存；d_* 是 cudaMalloc 分配的 GPU 内存。
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        std::vector<float> h_a(n), h_b(n), h_c(n);
        for (int i = 0; i < n; ++i) {
            h_a[i] = static_cast<float>(i % 1024) * 0.25f;
            h_b[i] = 1.0f + static_cast<float>(i % 17) * 0.125f;
        }

        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        // 正确结果均大于零；清零输出可让漏写更容易被校验发现。
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));

        // 1. H2D：CPU 准备的两个输入都必须传给 GPU。
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

        // 2. GPU：每个线程负责一个元素，尾部多余线程由 i < n 挡住。
        constexpr int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>(
            (static_cast<size_t>(n) + threads - 1) / threads);
        std::cout << "elements=" << n << ", blocks=" << blocks
                  << ", threads=" << threads << '\n';
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaGetLastError());       // 检查提交，不等待执行完成。
        CUDA_CHECK(cudaDeviceSynchronize());  // 教学用：在这里等待并检查执行错误。

        // 3. D2H：拷回结果后，CPU 才读取 h_c 做逐元素校验。
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        bool valid = true;
        for (int i = 0; i < n; ++i) {
            const float expected = h_a[i] + h_b[i];
            if (!std::isfinite(h_c[i]) || std::fabs(h_c[i] - expected) > 1e-5f) {
                std::cerr << "FAIL: i=" << i << ", expected=" << expected
                          << ", actual=" << h_c[i] << '\n';
                valid = false;
                break;
            }
        }

        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_a));
        if (!valid) return EXIT_FAILURE;
        std::cout << "PASS: 全部元素通过 CPU 校验（教学示例，无性能计时）\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
