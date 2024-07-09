There are 3 different sqrt functions here, one is a basic implementation using cmath, one is with basic SIMD and the last and most efficient is an optimized SIMD version. I have already written a benchmark template so you can see the stark difference in each implementation.

To run the following code, past the following line into the terminal

g++ -o simd_benchmark main.cpp -std=c++23 -O3 -mavx && ./simd_benchmark
