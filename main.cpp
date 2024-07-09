#include <iostream>
#include <immintrin.h> // For SIMD intrinsics
#include <chrono> // For timing
#include <cmath> // For standard sqrt

// compute square root using basic SIMD
double sqrt_simd(double value) {
    // loads the value into a __m256d register
    __m256d input = _mm256_set1_pd(value);

    // performs the square root operation
    __m256d result = _mm256_sqrt_pd(input);

    // extracts the result back to a double
    double output;
    _mm_store_sd(&output, _mm256_castpd256_pd128(result)); // only need the first element

    return output;
}

// computes the square root using SIMD, with an optimized approach similar to the exp_ps function
double sqrt_simd_optimized(double value) {
    __m256d x = _mm256_set1_pd(value);

    // Newton-Raphson iterations for better precision
    __m256d half = _mm256_set1_pd(0.5);

    __m256d y = _mm256_set1_pd(_mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_load_sd(&value)))); // initial estimate
    for (int i = 0; i < 5; ++i) {
        y = _mm256_mul_pd(half, _mm256_add_pd(y, _mm256_div_pd(x, y)));
    }

    double output {};
    _mm_store_sd(&output, _mm256_castpd256_pd128(y)); // only need the first element

    return output;
}

// fallback implementation for environments where AVX is not available (older computers, etc)
double sqrt_standard(double value) {
    return std::sqrt(value);
}

// benchmark function
template <typename Func>
double benchmark(Func func, const std::string& name, double value, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    double result = 0;
    for (int i = 0; i < iterations; ++i) {
        result = func(value);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << name << " - Result: " << result << ", Time taken: " << elapsed.count() << " seconds for " << iterations << " iterations" << std::endl;
    return elapsed.count();
}

int main() {
    double value = 42.0;
    int iterations = 1000000;

    // benchmarks
    double time_standard = benchmark(sqrt_standard, "Standard sqrt", value, iterations);

    double time_basic_simd = benchmark(sqrt_simd, "Basic SIMD", value, iterations);

    double time_optimized_simd = benchmark(sqrt_simd_optimized, "Optimized SIMD", value, iterations);

    double diff_basic_vs_standard = ((time_basic_simd - time_standard) / time_standard) * 100;
    double diff_optimized_vs_standard = ((time_optimized_simd - time_standard) / time_standard) * 100;
    double diff_optimized_vs_basic = ((time_optimized_simd - time_basic_simd) / time_basic_simd) * 100;

    std::cout << "Percentage difference between Basic SIMD and Standard: " << diff_basic_vs_standard << "%" << std::endl;
    std::cout << "Percentage difference between Optimized SIMD and Standard: " << diff_optimized_vs_standard << "%" << std::endl;
    std::cout << "Percentage difference between Optimized SIMD and Basic SIMD: " << diff_optimized_vs_basic << "%" << std::endl;

    return 0;
}
