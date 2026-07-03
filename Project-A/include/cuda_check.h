#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)
