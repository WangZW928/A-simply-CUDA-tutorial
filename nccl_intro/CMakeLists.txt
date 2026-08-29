cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures")
endif()

project(nccl_intro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(NCCL QUIET CONFIG)

if(NCCL_FOUND AND TARGET NCCL::nccl AND NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL ALIAS NCCL::nccl)
endif()

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
    set(NCCL_FOUND FALSE)
endif()

if(NOT NCCL_FOUND)
    find_path(NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT}
        PATH_SUFFIXES include
    )
    find_library(NCCL_LIBRARY
        NAMES nccl
        HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT}
        PATH_SUFFIXES lib lib64
    )

    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        add_library(NCCL::NCCL UNKNOWN IMPORTED)
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
        set(NCCL_FOUND TRUE)
    endif()
endif()

if(NOT NCCL_FOUND)
    message(STATUS "NCCL not found. Skipping nccl_intro targets.")
    message(STATUS "Install NCCL or configure with -DNCCL_ROOT=/path/to/nccl.")
    return()
endif()

message(STATUS "NCCL found. Building nccl_intro examples.")

add_executable(nccl_allreduce_bench src/nccl_allreduce_bench.cu)
target_link_libraries(nccl_allreduce_bench PRIVATE NCCL::NCCL)
set_target_properties(nccl_allreduce_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(nccl_p2p_demo src/nccl_p2p_demo.cu)
target_link_libraries(nccl_p2p_demo PRIVATE NCCL::NCCL)
set_target_properties(nccl_p2p_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
