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

// Tiling adapted from NVIDIA CUDA Samples matrixMul, via the fork
// WangZW928/cuda-samples @ 8a9e2c830c8a336b4f48c79e5ed5837031eb8551.
// 本仓库改编：朴素版对照、行主序矩形索引、尾块补零、CPU 参考校验。
// 完整许可见 ../LICENSE.NVIDIA；这是学习材料，不是通用高性能 GEMM。
#include <cuda_runtime.h>

#include <algorithm>
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

constexpr int TILE = 16;

// A: M×K，B: K×N，C: M×N；全部是连续行主序，每线程计算一个 C 元素。
__global__ void matrix_mul_naive(const float* a, const float* b, float* c,
                                 int m, int k_size, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;  // 本 kernel 没有 block 同步。
    float sum = 0.0f;
    for (int k = 0; k < k_size; ++k) {
        sum += a[row * k_size + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

// 必须以 (TILE, TILE) 启动；同一 block 协作加载两个 tile，再复用数据。
__global__ void matrix_mul_tiled(const float* a, const float* b, float* c,
                                 int m, int k_size, int n) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    // 即使 row/col 越界也不提前返回：这些线程仍需加载数据并参与同步。
    for (int base = 0; base < k_size; base += TILE) {
        const int a_col = base + tx;
        const int b_row = base + ty;
        as[ty][tx] = (row < m && a_col < k_size) ? a[row * k_size + a_col] : 0.0f;
        bs[ty][tx] = (b_row < k_size && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();  // 整个 block 加载完成后，才能读取其他线程写入的值。

        for (int k = 0; k < TILE; ++k) {
            sum += as[ty][k] * bs[k][tx];
        }
        __syncthreads();  // 所有线程用完本轮 tile，才允许覆盖下一轮数据。
    }
    if (row < m && col < n) c[row * n + col] = sum;
}

static int parse_dimension(const char* text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("M、K、N 必须是 1..256 的十进制整数");
    }
    int dimension = 0;
    try {
        dimension = std::stoi(value);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("M、K、N 必须在 1..256 内（教学规模）");
    }
    if (dimension < 1 || dimension > 256) {
        throw std::invalid_argument("M、K、N 必须在 1..256 内（教学规模）");
    }
    return dimension;
}

static bool validate(const char* name, const std::vector<float>& actual,
                     const std::vector<double>& expected, int n) {
    for (size_t i = 0; i < actual.size(); ++i) {
        const double tolerance = 1e-5 + 1e-5 * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
            std::cerr << "FAIL " << name << ": row=" << i / n << ", col=" << i % n
                      << ", expected=" << expected[i] << ", actual=" << actual[i] << '\n';
            return false;
        }
    }
    std::cout << "PASS " << name << ": 全部元素通过 CPU 校验\n";
    return true;
}

int main(int argc, char** argv) {
    try {
        int m = 37, k_size = 23, n = 29;  // 三个维度都不是 TILE 的整数倍。
        if (argc != 1 && argc != 4) {
            throw std::invalid_argument("用法：matrix_mul_demo [M K N]（各为 1..256）");
        }
        if (argc == 4) {
            m = parse_dimension(argv[1]);
            k_size = parse_dimension(argv[2]);
            n = parse_dimension(argv[3]);
        }

        std::vector<float> h_a(m * k_size), h_b(k_size * n), h_c(m * n);
        std::vector<double> reference(m * n, 0.0);
        // 非常量、带正负号的确定性输入，便于暴露行列/步长错误。
        for (int row = 0; row < m; ++row) {
            for (int k = 0; k < k_size; ++k) {
                h_a[row * k_size + k] = ((row * 7 + k * 3) % 19 - 9) * 0.125f;
            }
        }
        for (int k = 0; k < k_size; ++k) {
            for (int col = 0; col < n; ++col) {
                h_b[k * n + col] = ((k * 5 + col * 11) % 23 - 11) * 0.0625f;
            }
        }
        // CPU 直接按定义计算，用 double 累加作为两个 GPU 版本的独立参考。
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                for (int k = 0; k < k_size; ++k) {
                    reference[row * n + col] +=
                        static_cast<double>(h_a[row * k_size + k]) * h_b[k * n + col];
                }
            }
        }

        const size_t a_bytes = h_a.size() * sizeof(float);
        const size_t b_bytes = h_b.size() * sizeof(float);
        const size_t c_bytes = h_c.size() * sizeof(float);
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
        CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
        CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice));

        const dim3 block(TILE, TILE);
        const dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
        std::cout << "M=" << m << ", K=" << k_size << ", N=" << n
                  << ", block=(" << block.x << ',' << block.y << "), grid=("
                  << grid.x << ',' << grid.y << ")（教学示例，无性能计时）\n";
        bool valid = true;
        for (int version = 0; version < 2; ++version) {
            // 每个版本开始前重置输出；漏写不能被上个版本的正确结果掩盖。
            std::fill(h_c.begin(), h_c.end(), std::numeric_limits<float>::quiet_NaN());
            CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), c_bytes, cudaMemcpyHostToDevice));
            if (version == 0) {
                matrix_mul_naive<<<grid, block>>>(d_a, d_b, d_c, m, k_size, n);
            } else {
                matrix_mul_tiled<<<grid, block>>>(d_a, d_b, d_c, m, k_size, n);
            }
            CUDA_CHECK(cudaGetLastError());       // 检查 launch，不等待执行。
            CUDA_CHECK(cudaDeviceSynchronize());  // 教学用：等待并定位执行错误。
            CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));
            if (!validate(version == 0 ? "naive" : "tiled", h_c, reference, n)) valid = false;
        }

        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_a));
        return valid ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
