# 00. Project Learning Strategy

## The First-Principles Philosophy
This project is not just about writing code; it is about mastering the underlying hardware. Every step of the implementation will be accompanied by a corresponding note in this directory.

## Workflow
## Workflow

### 1. Theory & First Principles
Before writing code, I must establish the **Physics of the Problem**.
*   **Mathematical Formulation**: Define the operation strictly (e.g., $C = \alpha A \times B + \beta C$).
*   **Roofline Analysis**: Calculate the theoretical limits.
    *   What is the Arithmetic Intensity (FLOPs / Byte)?
    *   Are we Compute Bound or Memory Bound on the target GPU?
*   **The Hardware Bottleneck**: explicit statement of what resource we are optimizing for (e.g., "Latency is hidden, but DRAM bandwidth is saturated").

### 2. Implementation Strategy
*   **Baseline**: Start with a functionally correct, naive implementation.
*   **Iterative Optimization**: Apply *one* major optimization at a time (e.g., "Only add Shared Memory Tiling").
*   **Correctness Check**: Every kernel must pass a numerical verification against `cuBLAS` or a CPU reference before benchmarking. Speed is meaningless without accuracy.

### 3. Profiling & Diagnostics
*   **Metric-Driven Analysis**: Use **Nsight Compute (ncu)** to measure:
    *   **SOL components** (Speed-Of-Light for DRAM, L2, SM).
    *   **Occupancy** & Register Pressure.
    *   **Memory Coalescing Efficiency**.
*   **The "Why" Loop**: If performance is below target, identify specific hardware counters that explain the gap.

### 4. Documentation
*   Synthesize the findings into the note. Explain *why* the optimization worked using the collected metrics.

## Directory Structure
*   `src/`: Contains the actual CUDA source code.
*   `notes/`: Contains these markdown files.

## Roadmap
1.  **Naive GEMM**: Understanding the baseline and memory coalescing issues.
2.  **Global Memory Coalescing**: Ensuring aligned memory accesses.
3.  **Shared Memory Tiling**: Relieving global memory pressure.
4.  **1D vs 2D Tiling**: Optimizing block dimensions.
5.  **Vectorized Memory Access**: Using `float4`.
6.  **Tensor Cores (WMMA)**: Using specialized hardware instructions.
