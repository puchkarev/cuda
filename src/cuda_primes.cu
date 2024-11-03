#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>

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

// Function sets up the use of prime search using cuda.
// It uplolads the initialization array, calls the kernel and then copies results back and frees the memory usage.
void cuda_prime_search(int *o, uint64_t start, uint64_t size, int block_size) {
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
}
