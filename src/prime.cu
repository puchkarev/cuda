#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>

#include "flag.h"

// A function that fills the array v that starts at a number [start] and of size [size] with 1's
// for any value that is devisible by a value val.
// __device__ designator means it can be called from GPU functions (like kernels)
// __host__ designator means ti can be called from the CPU functions
__device__ __host__ void basic_prime_search(int *v, uint64_t start, uint64_t size, uint64_t val) {
    if (val <= 1) return;

    for (int64_t offset = (start / val) * val; offset < start + size; offset += val) {
      if (offset < start) continue;
      if (offset == val) continue;
      v[offset - start] = 1;
    }
}

// This is the kernel to be used on the GPU.
// It is only compiled for the GPU, and requires a special semantics to call.
__global__ void prime_search(int *v, uint64_t start, uint64_t size) {
    // Convert from the block id and thread id (within the block) into a unique (and sequential number).
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    basic_prime_search(v, start, size, /*val=*/tid);
}

// This function accepts
int main(int argc, char** argv) {
    const bool use_cuda = GetIntFlagOrDefault("use_cuda", 0) == 1;
    const int thread_num = GetIntFlagOrDefault("threads", 1);
    const int64_t start = GetIntFlagOrDefault("start", 1);
    const int64_t size = GetIntFlagOrDefault("size", 10);
    const int block_size = GetIntFlagOrDefault("block_size", 256);

    fprintf(stderr, "Computing primes between %ld and %ld [%s cuda] threads=%d\n",
            start, start+size, use_cuda ? "with" : "without", thread_num);

    // Allocate host memory
    int *o = (int*)malloc(sizeof(int) * size);

    // Initialize host array
    for(int i = 0; i < size; i++){
        o[i] = 0;
    }

    if (use_cuda) {
      const int grid_size = ((start + size + block_size) / block_size);

      // Allocate device memory
      int *d_o;
      cudaMalloc((void**)&d_o, sizeof(int) * size);

      // Transfer data from host to device memory
      cudaMemcpy(d_o, o, sizeof(int) * size, cudaMemcpyHostToDevice);

      // Executing kernel
      prime_search<<<grid_size,block_size>>>(d_o, start, size);

      // Transfer data back to host memory
      cudaMemcpy(o, d_o, sizeof(float) * size, cudaMemcpyDeviceToHost);

      // Deallocate device memory
      cudaFree(d_o);
    } else if (thread_num > 2) {
      const int grid_size = ((start + size + thread_num) / thread_num);

      // Start the threads
      std::thread threads[thread_num];
      for (int tid = 0; tid < thread_num; ++tid) {
        threads[tid] = std::thread([&](int from, int to) {
          for (int val = from; val < to; ++val) {
            basic_prime_search(o, start, size, val);
          }
        }, tid * grid_size, (tid + 1) * grid_size);
      }

      // Join the threads
      for (int tid = 0; tid < thread_num; ++tid) {
        if (threads[tid].joinable()) threads[tid].join();
      }
    } else {
      for (int val = 0; val < start + size; ++val) {
        basic_prime_search(o, start, size, val);
      }
    }

    // Output
    for(int i = 0; i < size; i++){
        if (o[i] == 0) {
          printf("%" PRId64 "\n", i + start);
        }
    }

    // Deallocate host memory
    free(o);
}
