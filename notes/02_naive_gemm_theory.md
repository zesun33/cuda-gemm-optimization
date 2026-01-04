# 02. Naive GEMM: Theory & Implementation Strategy

## Objective

Implement the simplest possible GEMM kernel to establish a **baseline** for optimization. This version prioritizes **correctness** over performance.

**Expected Performance**: 100-300 GFLOPS (~0.5-1.5% of A100 FP32 peak)

---

## The Algorithm

### Mathematical Operation

```
C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
```

For each output element `C[i][j]`:
1. Iterate over the shared dimension `k`
2. Accumulate the dot product of row `i` of A and column `j` of B

### Thread-to-Work Mapping

**Strategy**: Each thread computes **one element** of the output matrix C.

```
Thread (tx, ty) in Block (bx, by) computes C[row][col]
where:
  row = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x
```

**Visualization**:
```
Matrix C (M × N):
┌─────────────────────────────┐
│ [T0,0] [T0,1] ... [T0,N-1]  │  ← Thread (0, *, *) computes row 0
│ [T1,0] [T1,1] ... [T1,N-1]  │  ← Thread (1, *, *) computes row 1
│  ...                         │
│ [TM-1,0]       ... [TM-1,N-1]│
└─────────────────────────────┘

Each thread computes ONE element by:
  sum = 0
  for k in 0..K-1:
    sum += A[row][k] * B[k][col]
  C[row][col] = sum
```

---

## Memory Access Pattern Analysis

### Your GPU Hardware Specs

**RTX 3060 Ti** (Personal PC):
- **DRAM Bandwidth**: 448 GB/s
- **FP32 Peak**: 16.2 TFLOPS
- **Shared Memory per SM**: 100 KB
- **L2 Cache**: 4 MB

**A5000** (Server):
- **DRAM Bandwidth**: 768 GB/s
- **FP32 Peak**: 27.8 TFLOPS
- **Shared Memory per SM**: 100 KB
- **L2 Cache**: 6 MB

### Access Patterns in Naive GEMM

For thread computing `C[i][j]`:

**Reads from A**:
```cpp
for (int k = 0; k < K; k++) {
    sum += A[i * K + k] * ...;  // Access A[i][0], A[i][1], ..., A[i][K-1]
}
```
- **Pattern**: Sequential within a row
- **Coalescing**: **GOOD** if threads in a warp compute adjacent columns (same row)
- **Reuse**: Each row of A is read **N times** (once per output column)

**Reads from B**:
```cpp
for (int k = 0; k < K; k++) {
    sum += ... * B[k * N + j];  // Access B[0][j], B[1][j], ..., B[K-1][j]
}
```
- **Pattern**: Strided access (column-major reading from row-major storage)
- **Coalescing**: **BAD** - threads in a warp access elements `N` apart
- **Reuse**: Each column of B is read **M times** (once per output row)

---

### Deep Dive: Memory Coalescing

Memory coalescing is critical for GPU performance. Let's understand **exactly** why one pattern is GOOD and the other is BAD.

#### What is Memory Coalescing?

When 32 threads in a warp access memory simultaneously, the GPU tries to **combine** their requests into as few memory transactions as possible.

**GPU Memory Transaction**: Each transaction fetches **128 bytes** (32 consecutive floats).

**Best case**: 32 threads access consecutive addresses → **1 transaction** (100% bandwidth)  
**Worst case**: 32 threads access scattered addresses → **32 transactions** (3% bandwidth, 32× slower!)

#### Understanding Our Thread Layout

With `dim3 blockDim(16, 16)`, threads are organized as:

```
Block layout (16×16 = 256 threads):
Row 0:  (0,0)  (0,1)  (0,2)  ... (0,15)
Row 1:  (1,0)  (1,1)  (1,2)  ... (1,15)
...
Row 15: (15,0) (15,1) (15,2) ... (15,15)
```

**CUDA linearizes these for warp assignment** (row-major order):
- **Warp 0**: threads 0-31 = (0,0) through (1,15) [rows 0-1]
- **Warp 1**: threads 32-63 = (2,0) through (3,15) [rows 2-3]

Within Warp 0:
- Threads 0-15: compute C[blockIdx.y*16 + 0][blockIdx.x*16 + 0..15]
- Threads 16-31: compute C[blockIdx.y*16 + 1][blockIdx.x*16 + 0..15]

---

#### GOOD Pattern: Reading from A (Broadcast)

**What each thread accesses**:
```cpp
// Thread computing C[i][j] reads:
A[i * K + k]  // Element A[row i, column k]
```

**Example: Warp 0 at iteration k=5**:

```
Thread 0  (C[0][0]): reads A[0 * K + 5] = A[0][5]
Thread 1  (C[0][1]): reads A[0 * K + 5] = A[0][5]  ← SAME
Thread 2  (C[0][2]): reads A[0 * K + 5] = A[0][5]  ← SAME
...
Thread 15 (C[0][15]): reads A[0 * K + 5] = A[0][5] ← SAME

Thread 16 (C[1][0]): reads A[1 * K + 5] = A[1][5]
Thread 17 (C[1][1]): reads A[1 * K + 5] = A[1][5]  ← SAME
...
Thread 31 (C[1][15]): reads A[1 * K + 5] = A[1][5] ← SAME
```

**All threads with the same row index read the SAME address!**

This is called **broadcast** or **uniform access**:
- GPU fetches `A[0][5]` once and distributes to threads 0-15
- GPU fetches `A[1][5]` once and distributes to threads 16-31

**Result**: **2 transactions** for 32 threads instead of 32 → **16× more efficient!**

**Why this helps**:
- L1 cache serves the first request, then immediately serves 15 more from cache
- Or the memory controller broadcasts the value directly
- Minimal DRAM traffic

---

#### BAD Pattern: Reading from B (Strided Access)

**What each thread accesses**:
```cpp
// Thread computing C[i][j] reads:
B[k * N + j]  // Element B[row k, column j]
```

**Example: Warp 0 at iteration k=5**:

```
Thread 0  (C[0][0]): reads B[5 * N + 0]  ← Address: base + 5*N*4 + 0*4
Thread 1  (C[0][1]): reads B[5 * N + 1]  ← Address: base + 5*N*4 + 4
Thread 2  (C[0][2]): reads B[5 * N + 2]  ← Address: base + 5*N*4 + 8
...
Thread 15 (C[0][15]): reads B[5 * N + 15] ← Address: base + 5*N*4 + 60

Thread 16 (C[1][0]): reads B[5 * N + 0]  ← SAME as Thread 0
Thread 17 (C[1][1]): reads B[5 * N + 1]  ← SAME as Thread 1
...
```

**Within a single iteration**: Threads 0-15 access **consecutive** addresses (B[5*N+0] through B[5*N+15]). This IS coalesced! ✅

**But the real problem is ACROSS iterations**:

```
k=0: Warp reads B[0*N + j] (row 0)
k=1: Warp reads B[1*N + j] (row 1) ← N*4 bytes away from row 0!
k=2: Warp reads B[2*N + j] (row 2) ← N*4 bytes away from row 1!
...
```

**For N=1024**: Each row is **4096 bytes** (1024 floats × 4 bytes) away from the next.

A typical cache line is **128 bytes**, so row k and row k+1 are **32 cache lines apart**!

**The cache cannot hold all these rows**, so we get:
- Load B[0][j] → evicts from cache before we need it again
- Load B[1][j] → evicts B[0][j]
- Load B[2][j] → evicts B[1][j]
- ...never reuse cached data!

This is called **cache thrashing** or **poor spatial locality**.

---

#### Visual Comparison

**Memory access pattern for A (GOOD)**:
```
Iteration k=0: All threads in warp → A[row][0]  (2 addresses)
Iteration k=1: All threads in warp → A[row][1]  (2 addresses)
Iteration k=2: All threads in warp → A[row][2]  (2 addresses)
                ↓ consecutive elements, good cache reuse
```

**Memory access pattern for B (BAD)**:
```
Warp accesses column j across all rows:
k=0: B[0][j]   ← Load cache line
k=1: B[1][j]   ← 4KB away, new cache line
k=2: B[2][j]   ← 4KB away, new cache line 
k=3: B[3][j]   ← 4KB away, new cache line
     ↓ No reuse! Each access loads a new cache line
```

---

#### Quantitative Performance Impact

**For 1024×1024 GEMM on your 3060 Ti**:

**Ideal case (perfect coalescing)**:
- Bandwidth: 448 GB/s
- Can sustain: ~112 billion float loads/sec

**Reading from A (GOOD)**:
- Broadcast pattern → L1 cache hit rate ~95%
- Effective bandwidth utilization: ~80-90%
- Minimal DRAM traffic

**Reading from B (BAD)**:
- Strided access → L1 cache hit rate ~10-20%
- Effective bandwidth utilization: ~20-30%
- High DRAM traffic, poor cache reuse

**Overall Naive GEMM**:
- Achieves only ~25-30% of peak memory bandwidth
- Bottlenecked by the bad B access pattern
- This is why we see ~0.1-0.15 TFLOPS instead of 16.2 TFLOPS!

---

#### The Fix (Coming in Lesson 3: Shared Memory Tiling)

We'll fix the B access pattern by:
1. Loading a **tile** of B into Shared Memory
2. All threads in the block reuse this tile many times
3. Amortize the cost of the strided access over many operations

**Expected improvement**: 10-20× speedup! (From ~0.15 to ~2-3 TFLOPS)

---

### Bottleneck Identification

**For N = 1024, M = K = 1024**:

**Arithmetic Intensity (from Lesson 1)**:
```
AI = N/6 = 1024/6 ≈ 170 FLOPs/Byte
```

**Expected Regime**: Compute-bound (with perfect memory)

**But in reality**:
1. **Uncoalesced B accesses** reduce effective bandwidth by ~4-8×
2. **No cache reuse** - each element is read from DRAM every time it's needed
3. **Low ILP** - each iteration depends on the previous accumulation

**Predicted Performance**:
```
Effective Bandwidth ≈ 1555 GB/s / 4 = 388 GB/s (due to B strides)
Effective AI ≈ 170 / 4 = 42 FLOPs/Byte

Attainable = 388 GB/s × 42 = 16.3 TFLOPS (theoretical)

But with low ILP and stalls: ~0.1 - 0.3 TFLOPS (0.5-1.5% of peak)
```

---

## Implementation Plan

### Kernel Signature

```cpp
__global__ void gemm_naive(
    int M, int N, int K,
    const float* A,  // M × K
    const float* B,  // K × N
    float* C         // M × N
);
```

### Kernel Logic (Pseudocode)

```cpp
__global__ void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    // 1. Compute which element this thread is responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Bounds check
    if (row < M && col < N) {
        // 3. Compute the dot product
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // 4. Write result
        C[row * N + col] = sum;
    }
}
```

### Launch Configuration

```cpp
dim3 blockDim(16, 16);  // 256 threads per block (16×16 tile)
dim3 gridDim(
    (N + blockDim.x - 1) / blockDim.x,  // Ceiling division
    (M + blockDim.y - 1) / blockDim.y
);

gemm_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
```

**Why 16×16 blocks?**
- 256 threads per block = 8 warps (good occupancy)
- Square blocks have balanced memory access patterns
- Common choice for 2D grid problems

---

## Host Code Structure

### Memory Management

```cpp
// 1. Allocate device memory
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, M * K * sizeof(float));
cudaMalloc(&d_B, K * N * sizeof(float));
cudaMalloc(&d_C, M * N * sizeof(float));

// 2. Copy input matrices to device
cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

// 3. Launch kernel
gemm_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);

// 4. Copy result back to host
cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

// 5. Free device memory
cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
```

### Benchmarking

```cpp
// Use CUDA events for accurate timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
gemm_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

// Calculate TFLOPS
float flops = 2.0f * M * N * K;  // 2 ops per multiply-add
float tflops = (flops / milliseconds) / 1e9;  // Convert to TFLOPS
printf("Performance: %.2f TFLOPS\n", tflops);
```

---

## Correctness Verification

**Always verify correctness before benchmarking!**

### Reference Implementation (CPU)

```cpp
void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
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
```

### Verification

```cpp
// Compare GPU result to CPU reference
float max_error = 0.0f;
for (int i = 0; i < M * N; i++) {
    float error = fabs(h_C_gpu[i] - h_C_cpu[i]);
    max_error = fmax(max_error, error);
}

printf("Max error: %e\n", max_error);
if (max_error < 1e-3) {
    printf("✓ Verification passed!\n");
} else {
    printf("✗ Verification failed!\n");
}
```

---

## Expected Results

### Performance Prediction

| Matrix Size | FLOPs | Expected TFLOPS | Time (ms) |
|-------------|-------|-----------------|-----------|
| 512×512×512 | 268M | 0.1 - 0.2 | 1-3 ms |
| 1024×1024×1024 | 2.15B | 0.2 - 0.3 | 7-10 ms |
| 2048×2048×2048 | 17.2B | 0.2 - 0.3 | 57-86 ms |

**Why constant TFLOPS?** Because we're **not** utilizing the hardware efficiently:
- Memory bandwidth is underutilized (uncoalesced B)
- Compute is underutilized (low ILP, stalls)
- No data reuse (everything from DRAM)

---

## What We'll Learn

After implementing and profiling this kernel, you will understand:

1. **How to write a basic CUDA kernel** (thread indexing, memory access)
2. **How to benchmark GPU code** (CUDA events, TFLOPS calculation)
3. **Why naive approaches are slow** (uncoalesced access, no reuse)
4. **What to optimize next** (Shared Memory Tiling in Lesson 3)

---

## Key Takeaways

✅ **Correctness First**: Always verify against a reference before optimizing  
✅ **Baseline Performance**: ~0.1-0.3 TFLOPS is expected (don't be discouraged!)  
✅ **Identify Bottlenecks**: Profiling will show the problems (uncoalesced access, low occupancy)  
✅ **Iterative Optimization**: This is step 1 of 5 in our roadmap

---

## Next Steps

In the next section, we'll:
1. Write the complete CUDA code (`01_naive_gemm.cu`)
2. Create a Makefile for compilation
3. Run the kernel and verify correctness
4. Benchmark and profile with Nsight Compute
5. Analyze the results and identify specific bottlenecks

**Question to Consider**: Looking at the pseudocode, can you identify which line will cause the most memory traffic? Why?
