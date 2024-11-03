# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Project name and specify CUDA as a required language
project(CudaPrimeSearch LANGUAGES CUDA CXX)

# Specify the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories (for header files)
include_directories(include)

# Specify the CUDA source file
set(SRC_FILES src/cuda_primes.cu src/parallel_primes.cpp src/main.cpp)

# Add the executable target
add_executable(prime ${SRC_FILES})

# Set the output directory for the build files
set_target_properties(prime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Optional: Set additional CUDA compilation flags if needed
# target_compile_options(my_cuda_program PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
