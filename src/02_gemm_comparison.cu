// Row-major FP32 GEMM: naive -> shared-memory tiling -> strict FP32 cuBLAS.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t code = (call); if (code != cudaSuccess) \
    throw std::runtime_error(cudaGetErrorString(code)); } while (0)
#define BLAS(call) do { cublasStatus_t code = (call); if (code != CUBLAS_STATUS_SUCCESS) \
    throw std::runtime_error("cuBLAS status " + std::to_string(code)); } while (0)

__global__ void naive(int m, int n, int k, const float* a, const float* b, float* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0;
        for (int inner = 0; inner < k; ++inner)
            sum = fmaf(a[size_t(row) * k + inner], b[size_t(inner) * n + col], sum);
        c[size_t(row) * n + col] = sum;
    }
}

template<int Tile>
__global__ void tiled(int m, int n, int k, const float* a, const float* b, float* c) {
    __shared__ float as[Tile][Tile], bs[Tile][Tile];
    int row = blockIdx.y * Tile + threadIdx.y;
    int col = blockIdx.x * Tile + threadIdx.x;
    float sum = 0;
    for (int base = 0; base < k; base += Tile) {
        int ak = base + threadIdx.x, bk = base + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = row < m && ak < k ? a[size_t(row) * k + ak] : 0;
        bs[threadIdx.y][threadIdx.x] = bk < k && col < n ? b[size_t(bk) * n + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int inner = 0; inner < Tile; ++inner)
            sum = fmaf(as[threadIdx.y][inner], bs[inner][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < m && col < n) c[size_t(row) * n + col] = sum;
}

void fill(std::vector<float>& values, uint32_t& state) {
    for (float& value : values) {
        state = 1664525u * state + 1013904223u;
        value = float(state >> 8) / float(1u << 24) - 0.5f;
    }
}

double scaled_error(float actual, double expected) {
    if (!std::isfinite(actual)) return INFINITY;
    return std::abs(double(actual) - expected) / (1e-4 + 1e-4 * std::abs(expected));
}

int number(const char* text, int limit) {
    std::string value(text);
    size_t used = 0;
    int result = std::stoi(value, &used);
    if (used != value.size() || result <= 0 || result > limit)
        throw std::runtime_error("Arguments must be positive integers within supported limits");
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6) throw std::runtime_error("Usage: 02_gemm_comparison M N K [iterations=20] [warmup=5]; dimensions <= 8192");
        int m = number(argv[1], 8192), n = number(argv[2], 8192), k = number(argv[3], 8192);
        int iterations = argc > 4 ? number(argv[4], 10000) : 20;
        int warmup = argc > 5 ? number(argv[5], 1000) : 5;
        // Always select visible ordinal zero. Choose a physical GPU using CUDA_VISIBLE_DEVICES.
        CUDA(cudaSetDevice(0));
        cudaDeviceProp properties;
        CUDA(cudaGetDeviceProperties(&properties, 0));
        int runtime, driver, blas_version;
        CUDA(cudaRuntimeGetVersion(&runtime));
        CUDA(cudaDriverGetVersion(&driver));
        cublasHandle_t handle;
        BLAS(cublasCreate(&handle));
        BLAS(cublasGetVersion(handle, &blas_version));
        // Identical FP32 arithmetic policy: no TF32 / Tensor Core acceleration.
        BLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        std::vector<float> a(size_t(m) * k), b(size_t(k) * n), reference(size_t(m) * n), output(reference.size());
        uint32_t seed = 42;
        fill(a, seed); fill(b, seed);
        float *da, *db, *dc;
        CUDA(cudaMalloc(&da, a.size() * sizeof(float)));
        CUDA(cudaMalloc(&db, b.size() * sizeof(float)));
        CUDA(cudaMalloc(&dc, output.size() * sizeof(float)));
        CUDA(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
        float alpha = 1, beta = 0;
        auto blas = [&]() {
            // Column-major API: C^T = B^T A^T gives row-major C = A B.
            BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, db, n, da, k, &beta, dc, n));
        };
        blas();
        CUDA(cudaMemcpy(reference.data(), dc, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        // Full CPU reference for small cases; deterministic samples for larger ones.
        size_t samples = output.size() <= 65536 && k <= 256 ? output.size() : std::min<size_t>(128, output.size());
        double cpu_max_error = 0;
        for (size_t sample = 0; sample < samples; ++sample) {
            size_t index = samples == output.size() ? sample : sample * (output.size() - 1) / (samples - 1);
            int row = index / n, col = index % n;
            double sum = 0;
            for (int inner = 0; inner < k; ++inner)
                sum += double(a[size_t(row) * k + inner]) * b[size_t(inner) * n + col];
            cpu_max_error = std::max(cpu_max_error, scaled_error(reference[index], sum));
        }
        if (cpu_max_error > 1) throw std::runtime_error("cuBLAS failed the independent double-precision CPU reference");
        dim3 block16(16, 16), grid16((n + 15) / 16, (m + 15) / 16);
        dim3 block32(32, 32), grid32((n + 31) / 32, (m + 31) / 32);
        std::vector<std::pair<std::string, std::function<void()>>> kernels = {
            {"naive", [&]() { naive<<<grid16, block16>>>(m, n, k, da, db, dc); }},
            {"tiled16", [&]() { tiled<16><<<grid16, block16>>>(m, n, k, da, db, dc); }},
            {"tiled32", [&]() { tiled<32><<<grid32, block32>>>(m, n, k, da, db, dc); }},
            {"cublas_fp32", blas},
        };
        cudaEvent_t start, stop;
        CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
        std::cout << std::setprecision(9);
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        // Metadata strings originate from the runtime or numeric/UUID device masks.
        std::cout << "{\"M\":" << m << ",\"N\":" << n << ",\"K\":" << k
                  << ",\"gpu\":\"" << properties.name << "\",\"compute_capability\":\"sm_"
                  << properties.major << properties.minor << "\",\"visible_device\":0,\"cuda_visible_devices\":\""
                  << (visible ? visible : "") << "\",\"runtime_version\":" << runtime
                  << ",\"driver_version\":" << driver << ",\"cublas_version\":" << blas_version
                  << ",\"seed\":42,\"warmup\":" << warmup << ",\"iterations\":" << iterations
                  << ",\"math_mode\":\"FP32_PEDANTIC_NO_TF32\",\"rtol\":0.0001,\"atol\":0.0001"
                  << ",\"cpu_reference_samples\":" << samples << ",\"cpu_max_scaled_error\":" << cpu_max_error
                  << ",\"results\":[";
        bool first = true;
        for (const auto& kernel : kernels) {
            CUDA(cudaMemset(dc, 0xff, output.size() * sizeof(float)));
            kernel.second(); CUDA(cudaGetLastError());
            CUDA(cudaMemcpy(output.data(), dc, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double max_absolute = 0, max_scaled = 0;
            for (size_t i = 0; i < output.size(); ++i) {
                max_scaled = std::max(max_scaled, scaled_error(output[i], reference[i]));
                max_absolute = std::max(max_absolute, std::abs(double(output[i]) - reference[i]));
            }
            if (max_scaled > 1) throw std::runtime_error(kernel.first + " failed full-output correctness check");
            for (int i = 0; i < warmup; ++i) kernel.second();
            CUDA(cudaDeviceSynchronize());
            std::vector<float> latency;
            for (int i = 0; i < iterations; ++i) {
                CUDA(cudaEventRecord(start)); kernel.second(); CUDA(cudaGetLastError());
                CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
                float ms; CUDA(cudaEventElapsedTime(&ms, start, stop));
                if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("Invalid timing measurement");
                latency.push_back(ms);
            }
            std::vector<float> sorted = latency;
            std::sort(sorted.begin(), sorted.end());
            double p50 = sorted[sorted.size() / 2];
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"kernel\":\"" << kernel.first << "\",\"valid\":true,\"max_abs_error\":" << max_absolute
                      << ",\"max_scaled_error\":" << max_scaled << ",\"p50_ms\":" << p50
                      << ",\"gflops\":" << (2.0 * m * n * k / (p50 * 1e6)) << ",\"latencies_ms\":[";
            for (size_t i = 0; i < latency.size(); ++i) std::cout << (i ? "," : "") << latency[i];
            std::cout << "]}";
        }
        std::cout << "]}\n";
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
        BLAS(cublasDestroy(handle));
        CUDA(cudaFree(da)); CUDA(cudaFree(db)); CUDA(cudaFree(dc));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GEMM comparison: " << error.what() << '\n';
        return 1;
    }
}
