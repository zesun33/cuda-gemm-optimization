# Quick Start Guide

## Compilation

If you have a CUDA-capable GPU, you can compile and run the naive GEMM kernel:

### Option 1: Using Make (Recommended)

```bash
# Build all kernels
make

# Run naive GEMM with default size (1024×1024×1024)
make run_naive

# Clean build artifacts
make clean
```

**Note**: If you don't have an A100 (sm_80), change the architecture:
```bash
# For RTX 3090/3080 (sm_86)
make ARCH=sm_86

# For RTX 4090 (sm_89)
make ARCH=sm_89

# For V100 (sm_70)
make ARCH=sm_70
```

### Option 2: Manual Compilation

```bash
# Compile
nvcc -O3 -arch=sm_80 src/01_naive_gemm.cu -o build/naive_gemm

# Run
./build/naive_gemm 1024 1024 1024
```

## Usage

```bash
./build/01_naive_gemm [M] [N] [K]

# Examples:
./build/01_naive_gemm 512 512 512      # Small matrix
./build/01_naive_gemm 1024 1024 1024   # Medium matrix
./build/01_naive_gemm 2048 2048 2048   # Large matrix
```

## Expected Output

```
=============================================================
Naive GEMM Benchmark
=============================================================
Matrix dimensions: C(1024 × 1024) = A(1024 × 1024) × B(1024 × 1024)
Total FLOPs: 2.15e+09

Initializing matrices...
Running GPU kernel...
  Grid: (64, 64), Block: (16, 16)
  Total threads: 262144

Running CPU reference...
Verifying result...
Max error: 1.192093e-07
Errors (>1e-03): 0 / 1048576 (0.00%)
✓ Verification PASSED!

=============================================================
Benchmarking...
=============================================================
Average time: 8.234 ms
Performance:  261.25 GFLOPS (0.2613 TFLOPS)
Efficiency:   1.34% of A100 FP32 peak (19.5 TFLOPS)
Bandwidth:    1458.32 GB/s
=============================================================
```

## What to Look For

1. **Verification**: Should say "✓ Verification PASSED!"
2. **Performance**: Expected 100-300 GFLOPS (~0.5-1.5% of peak)
3. **Efficiency**: Very low (1-2%) - this is normal for naive implementation

## Next Steps

After running this kernel, proceed to Lesson 3 where we'll:
- Profile with Nsight Compute to identify bottlenecks
- Implement Shared Memory Tiling
- Achieve 10-20× speedup

---

## Troubleshooting

**No CUDA GPU available?**
- The code will still compile but won't run
- You can study the code and understand the concepts
- Consider using Google Colab or cloud GPU instances

**Compilation errors?**
- Check your CUDA toolkit version: `nvcc --version`
- Make sure your GPU architecture is correct
- Try a different `-arch` flag

**Performance much lower than expected?**
- This is intentional! The naive version is the baseline
- We'll optimize in subsequent lessons
