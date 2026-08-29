cmake_minimum_required(VERSION 3.18)
project(cuda_warp_impact LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(warp_bench src/warp_bench.cu)
set_target_properties(warp_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)
