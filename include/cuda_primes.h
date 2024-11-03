#ifndef CUDA_PRIMES_H
#define CUDA_PRIMES_H

#include <inttypes.h>

// A function that fills the array v that starts at a number [start] and of size [size] with 1's
// for any value that is devisible by a value val.
void basic_prime_search(int *v, uint64_t start, uint64_t size, uint64_t val);

// Function sets up the use of prime search using cuda
void cuda_prime_search(int *o, uint64_t start, uint64_t size, int block_size);

#endif // CUDA_PRIMES_H
