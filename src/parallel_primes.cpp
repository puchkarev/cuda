#include <inttypes.h>
#include <thread>

#include "cuda_primes.h"

namespace parallel_primes {

void parallel_prime_search(int *o, uint64_t start, uint64_t size, int thread_num) {
    const int grid_size = ((start + size + thread_num) / thread_num);

    // Start the threads
    std::thread threads[thread_num];
    for (int tid = 0; tid < thread_num; ++tid) {
      threads[tid] = std::thread([&](int from, int to) {
        for (int val = from; val < to; ++val) {
          cuda_primes::basic_prime_search(o, start, size, val);
        }
      }, tid * grid_size, (tid + 1) * grid_size);
    }

    // Join the threads
    for (int tid = 0; tid < thread_num; ++tid) {
      if (threads[tid].joinable()) threads[tid].join();
    }
}

}  // namespace parallel_primes

