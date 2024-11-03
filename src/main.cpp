#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "flag.h"
#include "cuda_primes.h"

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
      cuda_prime_search(o, start, size, block_size);
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
