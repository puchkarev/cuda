#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "flag.h"
#include "cuda_primes.h"
#include "parallel_primes.h"

// This function accepts
int main(int argc, char** argv) {
    const bool use_cuda = base::GetIntFlagOrDefault("use_cuda", 0) == 1;
    const int thread_num = base::GetIntFlagOrDefault("threads", 1);
    const int64_t start = base::GetIntFlagOrDefault("start", 1);
    const int64_t size = base::GetIntFlagOrDefault("size", 10);
    const int block_size = base::GetIntFlagOrDefault("block_size", 256);

    fprintf(stderr, "Computing primes between %ld and %ld [%s cuda] threads=%d\n",
            start, start+size, use_cuda ? "with" : "without", thread_num);

    // Allocate host memory
    int *o = (int*)malloc(sizeof(int) * size);

    // Initialize host array
    for(int i = 0; i < size; i++){
        o[i] = 0;
    }

    if (use_cuda) {
      cuda_primes::cuda_prime_search(o, start, size, block_size);
    } else if (thread_num > 2) {
      parallel_primes::parallel_prime_search(o, start, size, thread_num);
    } else {
      for (int val = 0; val < start + size; ++val) {
        cuda_primes::basic_prime_search(o, start, size, val);
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
