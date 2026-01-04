/*
 * Naive GEMM Implementation
 * 
 * Purpose: Baseline matrix multiplication kernel for correctness and benchmarking
 * Expected Performance: 100-300 GFLOPS (~0.5-1.5% of A100 FP32 peak)
 * 
 * Compilation:
 *   nvcc -O3 -arch=sm_80 01_naive_gemm.cu -o naive_gemm
 * 
 * Usage:
 *   ./naive_gemm [M] [N] [K]
 *   Example: ./naive_gemm 1024 1024 1024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// KERNEL: Naive GEMM
// ============================================================================

__global__ void gemm_naive(
    int M, int N, int K,
    const float* __restrict__ A,  // M × K
    const float* __restrict__ B,  // K × N
    float* __restrict__ C         // M × N
) {
    // Compute the row and column of C this thread is responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Write result
        C[row * N + col] = sum;
    }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

void gemm_cpu(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

// Initialize matrix with random values
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Verify GPU result against CPU reference
bool verify_result(const float* gpu_result, const float* cpu_result, int size) {
    float max_error = 0.0f;
    int error_count = 0;
    const float tolerance = 1e-3f;
    
    for (int i = 0; i < size; i++) {
        float error = fabsf(gpu_result[i] - cpu_result[i]);
        max_error = fmaxf(max_error, error);
        
        if (error > tolerance) {
            error_count++;
            if (error_count <= 5) {  // Print first 5 errors
                printf("  Error at index %d: GPU=%.6f, CPU=%.6f, diff=%.6e\n",
                       i, gpu_result[i], cpu_result[i], error);
            }
        }
    }
    
    printf("Max error: %.6e\n", max_error);
    printf("Errors (>%.0e): %d / %d (%.2f%%)\n", 
           tolerance, error_count, size, 100.0f * error_count / size);
    
    return max_error < tolerance;
}

// ============================================================================
// Benchmarking
// ============================================================================

float benchmark_kernel(
    int M, int N, int K,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int num_iterations
) {
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    // Warm-up run
    gemm_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        gemm_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds / num_iterations;  // Average time per iteration
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Matrix dimensions (default to 1024x1024x1024)
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    
    printf("=============================================================\n");
    printf("Naive GEMM Benchmark\n");
    printf("=============================================================\n");
    printf("Matrix dimensions: C(%d × %d) = A(%d × %d) × B(%d × %d)\n",
           M, N, M, K, K, N);
    
    // Calculate total FLOPs
    double flops = 2.0 * M * N * K;
    printf("Total FLOPs: %.2e\n", flops);
    
    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);
    
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize matrices
    printf("\nInitializing matrices...\n");
    srand(42);  // Fixed seed for reproducibility
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Run GPU kernel
    printf("Running GPU kernel...\n");
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    printf("  Grid: (%d, %d), Block: (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("  Total threads: %d\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);
    
    gemm_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Verify correctness (only for small matrices)
    if (M * N <= 1024 * 1024) {
        printf("\nRunning CPU reference...\n");
        gemm_cpu(M, N, K, h_A, h_B, h_C_cpu);
        
        printf("Verifying result...\n");
        bool correct = verify_result(h_C_gpu, h_C_cpu, M * N);
        
        if (correct) {
            printf("✓ Verification PASSED!\n");
        } else {
            printf("✗ Verification FAILED!\n");
            return EXIT_FAILURE;
        }
    } else {
        printf("\nSkipping verification (matrix too large)\n");
    }
    
    // Benchmark
    printf("\n=============================================================\n");
    printf("Benchmarking...\n");
    printf("=============================================================\n");
    
    int num_iterations = 10;
    float avg_time_ms = benchmark_kernel(M, N, K, d_A, d_B, d_C, num_iterations);
    
    // Get GPU information
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Determine peak TFLOPS based on GPU
    double peak_tflops;
    const char* gpu_name = deviceProp.name;
    
    if (strstr(gpu_name, "3060 Ti") || strstr(gpu_name, "RTX 3060 Ti")) {
        peak_tflops = 16.2;  // RTX 3060 Ti FP32 peak
    } else if (strstr(gpu_name, "A5000")) {
        peak_tflops = 27.8;  // A5000 FP32 peak
    } else if (strstr(gpu_name, "A100")) {
        peak_tflops = 19.5;  // A100 FP32 peak
    } else if (strstr(gpu_name, "4090") || strstr(gpu_name, "RTX 4090")) {
        peak_tflops = 82.6;  // RTX 4090 FP32 peak
    } else if (strstr(gpu_name, "3090") || strstr(gpu_name, "RTX 3090")) {
        peak_tflops = 35.6;  // RTX 3090 FP32 peak
    } else if (strstr(gpu_name, "V100")) {
        peak_tflops = 15.7;  // V100 FP32 peak
    } else {
        // Generic estimate based on CUDA cores
        peak_tflops = (deviceProp.multiProcessorCount * 128 * 2 * deviceProp.clockRate) / 1e9;
    }
    
    // Calculate performance metrics
    double gflops = (flops / 1e9) / (avg_time_ms / 1000.0);
    double tflops = gflops / 1000.0;
    
    printf("GPU:          %s\n", gpu_name);
    printf("Average time: %.3f ms\n", avg_time_ms);
    printf("Performance:  %.2f GFLOPS (%.4f TFLOPS)\n", gflops, tflops);
    
    // Calculate percentage of peak
    double efficiency = (tflops / peak_tflops) * 100.0;
    printf("Efficiency:   %.2f%% of FP32 peak (%.1f TFLOPS)\n",
           efficiency, peak_tflops);
    
    // Memory bandwidth
    double bytes_accessed = (size_A + size_B + size_C);  // Naive estimate
    double bandwidth_gb_s = (bytes_accessed / 1e9) / (avg_time_ms / 1000.0);
    printf("Bandwidth:    %.2f GB/s\n", bandwidth_gb_s);
    
    printf("=============================================================\n");
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return EXIT_SUCCESS;
}
