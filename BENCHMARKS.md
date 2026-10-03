# GEMM comparison methodology

`src/02_gemm_comparison.cu` runs four implementations on identical seeded signed FP32 inputs. All compute row-major `C = A B` with alpha 1 and beta 0. cuBLAS uses `CUBLAS_PEDANTIC_MATH`, disabling TF32/Tensor Core math for this FP32 comparison; this is not a comparison to cuBLAS's fastest mixed-precision modes.

## Correctness

cuBLAS is first checked against an independent double-precision CPU sum. Small outputs (<= 65,536 values and K <= 256) are fully checked; other cases check 128 deterministic positions spanning the output. Every implementation is then checked across its full output against cuBLAS, including non-tile-aligned rectangular shapes. The tolerance is `abs(error) <= 1e-4 + 1e-4 * abs(reference)`. Output starts with a NaN sentinel; nonfinite or unwritten results fail. Any validation failure returns nonzero and cannot be recorded as a successful benchmark.

`./scripts/verify.sh` covers scalar, rectangular, odd dimensions, and square matrices, plus invalid arguments. It requires actual CUDA execution; absence of a GPU is not a pass.

## Timing

Five warmups precede 20 separately timed launches per implementation. CUDA events record device elapsed time on the default stream. The CSV reports upper-median latency (sorted sample at index `count // 2`) and throughput `2*M*N*K / seconds`. Allocation, input transfers, CPU checks, and result transfers are outside the measured interval. Timing is not end-to-end application latency. The per-launch event method has overhead; very small matrices are sensitive to launch overhead and timer resolution.

The recorded run used one RTX A5000 (host GPU 4, visible ordinal 0), CUDA compiler/runtime 12.5, driver API 13.3, and cuBLAS 12.5.3. Raw metadata includes the timestamp, driver/runtime/library versions, device-mask and host-activity snapshot, source and executable SHA-256, and Git state. `git_dirty: true` records that the new implementation was uncommitted when measured; the source SHA-256 identifies the exact measured source. The host has other active GPUs. GPU clocks, thermal state, power, execution order, cache reuse, and concurrent work can affect timings; this is one measured run, not a universal performance guarantee or a cold-cache experiment.

## Reproduce or regenerate plots

```bash
make build/02_gemm_comparison ARCH=sm_86
CUDA_VISIBLE_DEVICES=0 ./scripts/verify.sh
python3 -m pip install -r requirements-plots.txt
CUDA_VISIBLE_DEVICES=0 python3 scripts/benchmark.py --iterations 20 --warmup 5 --out results/my-run
python3 scripts/benchmark.py --from-json results/2026-10-03-rtx-a5000/results.json --out /tmp/gemm-report
```

Choose `ARCH` and the device mask for your machine. Only the second command runs the correctness suite; plot regeneration executes no CUDA kernels. Keep the raw JSON beside published charts.

cuBLAS API/math-mode reference: https://docs.nvidia.com/cuda/cublas/index.html
