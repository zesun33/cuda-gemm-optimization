#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
make build/02_gemm_comparison "ARCH=${CUDA_ARCH:-sm_86}"
python3 - <<'PY'
import json
import subprocess
cases = [(1, 1, 1), (17, 31, 23), (63, 65, 37), (128, 128, 128), (256, 256, 256)]
for dims in cases:
    result = subprocess.run(['./build/02_gemm_comparison', *map(str, dims), '3', '1'],
                            text=True, capture_output=True, check=True, timeout=120)
    data = json.loads(result.stdout)
    assert {r['kernel'] for r in data['results']} == {'naive', 'tiled16', 'tiled32', 'cublas_fp32'}
    assert all(r['valid'] and r['max_scaled_error'] <= 1 for r in data['results'])
    assert data['cpu_max_scaled_error'] <= 1
    print(f'PASS: {dims}, all four implementations; CPU samples={data["cpu_reference_samples"]}')
for arguments in [['0','1','1'], ['-1','3','4'], ['1','1','1','0'], ['bad','2','3']]:
    result = subprocess.run(['./build/02_gemm_comparison', *arguments], capture_output=True)
    assert result.returncode != 0
print('PASS: invalid dimensions and iteration counts rejected')
PY
