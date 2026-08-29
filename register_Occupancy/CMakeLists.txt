cmake_minimum_required(VERSION 3.18)
project(register_occupancy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(reg_occ_bench src/reg_occ_bench.cu)
set_target_properties(reg_occ_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)
