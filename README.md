# Optimized Matrix Multiplication (GEMM) in CUDA

## Goal
Implement a high-performance FP16 Matrix Multiplication kernel from scratch, optimizing it step-by-step to approach cuBLAS performance.

## Roadmap
- [ ] **v0_naive.cu**: Simple `C[row][col] = sum(A[row][k] * B[k][col])` implementation.
- [ ] **v1_coalesced.cu**: Optimizing global memory access patterns.
- [ ] **v2_shared_mem_tiling.cu**: Using shared memory blocks to minimize global bandwidth.
- [ ] **v3_double_buffering.cu**: Prefetching data to hide memory latency.
- [ ] **v4_tensor_cores.cu**: Using `nvcuda::wmma` intrinsics for FP16 acceleration.
- [ ] **benchmark.py**: Python script to plot GFLOPS comparison.

## References
*   [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
*   [Cutlass Documentation](https://github.com/NVIDIA/cutlass)
