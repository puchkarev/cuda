#ifndef PARALLEL_PRIMES_H
#define PARALLE_PRIMES_H

#include <inttypes.h>

namespace parallel_primes {

// Function sets up the prime search using threads
void parallel_prime_search(int *o, uint64_t start, uint64_t size, int thread_num);

}  // namespace parallel_primes

#endif // PARALLEL_PRIMES_H
