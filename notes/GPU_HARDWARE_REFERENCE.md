# GPU Hardware Reference

This document contains specifications and expected performance for the GPUs used in this project.

## Your Hardware

### RTX 3060 Ti (Personal PC)

| Specification | Value |
|--------------|-------|
| **Compute Capability** | 8.6 (sm_86) |
| **CUDA Cores** | 4864 |
| **SMs** | 38 |
| **FP32 Peak** | 16.2 TFLOPS |
| **FP16 (Tensor Core)** | 130 TFLOPS |
| **Memory** | 8 GB GDDR6 |
| **Memory Bandwidth** | 448 GB/s |
| **TDP** | 200W |

**Performance Expectations:**

| Kernel Version | Expected TFLOPS | % of Peak | Notes |
|----------------|----------------|-----------|-------|
| Naive GEMM | 0.05 - 0.15 | 0.3-0.9% | Baseline, unoptimized |
| Coalesced | 0.15 - 0.30 | 0.9-1.8% | Fixed memory patterns |
| Shared Memory Tiling | 1.5 - 3.0 | 9-18% | Major speedup |
| Register Blocking | 4.0 - 6.0 | 25-37% | Good compute utilization |
| Tensor Cores (FP16) | 30 - 60 | 23-46% | Hardware accelerated |

---

### A5000 (Server)

| Specification | Value |
|--------------|-------|
| **Compute Capability** | 8.6 (sm_86) |
| **CUDA Cores** | 8192 |
| **SMs** | 64 |
| **FP32 Peak** | 27.8 TFLOPS |
| **FP16 (Tensor Core)** | 222 TFLOPS |
| **Memory** | 24 GB GDDR6 |
| **Memory Bandwidth** | 768 GB/s |
| **TDP** | 230W |

**Performance Expectations:**

| Kernel Version | Expected TFLOPS | % of Peak | Notes |
|----------------|----------------|-----------|-------|
| Naive GEMM | 0.10 - 0.25 | 0.4-0.9% | Baseline, unoptimized |
| Coalesced | 0.25 - 0.50 | 0.9-1.8% | Fixed memory patterns |
| Shared Memory Tiling | 2.5 - 5.0 | 9-18% | Major speedup |
| Register Blocking | 7.0 - 10.0 | 25-36% | Good compute utilization |
| Tensor Cores (FP16) | 50 - 100 | 23-45% | Hardware accelerated |

---

## Why A5000 Will Be Faster

For the same kernel, the A5000 should be approximately **1.7×** faster than the 3060 Ti because:

1. **More SMs**: 64 vs 38 (1.68× more)
2. **More CUDA Cores**: 8192 vs 4864 (1.68× more)
3. **Higher Memory BW**: 768 GB/s vs 448 GB/s (1.71× faster)
4. **More Memory**: 24 GB vs 8 GB (allows larger matrices without paging)

**Example**: If 3060 Ti achieves 0.10 TFLOPS on Naive GEMM:
```
A5000 should achieve: 0.10 × 1.7 ≈ 0.17 TFLOPS
```

---

## Compilation

Both GPUs use the same architecture (sm_86), so the Makefile is already configured correctly:

```makefile
ARCH = sm_86
```

You can compile once and run on either GPU without recompilation.

---

## Roofline Analysis

### RTX 3060 Ti

For 1024×1024×1024 GEMM:
```
AI = 1024/6 ≈ 170 FLOPs/Byte

Memory-bound limit = 448 GB/s × 170 = 76.2 TFLOPS
Compute-bound limit = 16.2 TFLOPS

→ Compute-bound at 16.2 TFLOPS
```

### A5000

For 1024×1024×1024 GEMM:
```
AI = 1024/6 ≈ 170 FLOPs/Byte

Memory-bound limit = 768 GB/s × 170 = 130.6 TFLOPS
Compute-bound limit = 27.8 TFLOPS

→ Compute-bound at 27.8 TFLOPS
```

**Conclusion**: For typical GEMM sizes (N ≥ 512), both GPUs are **compute-bound**, so optimizations should focus on:
1. Maximizing instruction throughput (occupancy)
2. Increasing ILP (register blocking)
3. Using Tensor Cores for FP16

---

## Testing Strategy

**Recommended Workflow**:

1. **Develop on 3060 Ti** (local PC):
   - Quick iteration
   - Sufficient for kernel development and debugging
   - All optimizations will transfer to A5000

2. **Benchmark on A5000** (server):
   - Final performance numbers
   - Larger matrix sizes (use that 24GB!)
   - Production-scale testing

**Both GPUs will show the same relative performance improvements** (e.g., tiling gives 10-20× speedup on both).

---

## Next Steps

Run the naive GEMM kernel on your 3060 Ti to establish your baseline:

```bash
make
make run_naive
```

Expected output:
```
GPU:          NVIDIA GeForce RTX 3060 Ti
Performance:  ~0.05 - 0.15 GFLOPS
Efficiency:   ~0.3-0.9% of FP32 peak (16.2 TFLOPS)
```

Then you can SSH to your server and run on the A5000 for comparison!
