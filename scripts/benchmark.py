#!/usr/bin/env python3
"""Run checked GEMM comparisons and save raw measurements, CSV, and a plot."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shape(value):
    try:
        dims = tuple(int(part) for part in value.lower().split('x'))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use MxNxK, for example 127x255x63') from error
    if len(dims) != 3 or not all(0 < dim <= 8192 for dim in dims):
        raise argparse.ArgumentTypeError('Use three positive dimensions <= 8192')
    return dims


def report(data, directory):
    rows = []
    for case in data['cases']:
        baseline = case['results'][0]['p50_ms']
        for result in case['results']:
            rows.append({'shape': f"{case['M']}x{case['N']}x{case['K']}",
                         'kernel': result['kernel'], 'p50_ms': result['p50_ms'],
                         'gflops': result['gflops'], 'speedup_vs_naive': baseline / result['p50_ms'],
                         'max_abs_error': result['max_abs_error'],
                         'max_scaled_error': result['max_scaled_error'], 'valid': result['valid']})
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    shapes = list(dict.fromkeys(row['shape'] for row in rows))
    for index, kernel in enumerate(['naive', 'tiled16', 'tiled32', 'cublas_fp32']):
        values = [next(row['gflops'] for row in rows if row['shape'] == size and row['kernel'] == kernel)
                  for size in shapes]
        ax.bar([i + (index - 1.5) * 0.2 for i in range(len(shapes))], values, width=0.2, label=kernel)
    ax.set_xticks(range(len(shapes)), shapes, rotation=20)
    ax.set_ylabel('GFLOP/s (median CUDA-event latency)')
    ax.set_title(f"FP32 GEMM on {data['cases'][0]['gpu']}\nTF32 disabled; kernel time excludes allocation and transfers")
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / 'comparison.png', dpi=160)
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=shape,
                        default=[(128, 128, 128), (256, 256, 256), (512, 512, 512),
                                 (1024, 1024, 1024), (127, 255, 63)])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--from-json', type=Path, help='Regenerate reports without running CUDA')
    args = parser.parse_args()
    if not (0 < args.iterations <= 10000 and 0 < args.warmup <= 1000):
        parser.error('iterations must be 1..10000 and warmup must be 1..1000')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        data = json.loads(args.from_json.read_text())
    else:
        binary = ROOT / 'build/02_gemm_comparison'
        if not binary.is_file():
            parser.error('Build first: make build/02_gemm_comparison ARCH=sm_86 (choose your GPU architecture)')
        def capture(command):
            return subprocess.check_output(command, cwd=ROOT, text=True).strip()
        data = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_sha256': hashlib.sha256((ROOT / 'src/02_gemm_comparison.cu').read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'git_head': capture(['git', 'rev-parse', 'HEAD']),
                'git_dirty': bool(capture(['git', 'status', '--porcelain'])),
                'nvcc': capture(['nvcc', '--version']),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'timing_scope': 'CUDA events; allocation, CPU validation, and host transfers excluded',
                'cases': []}
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True)
        data['host_gpu_snapshot'] = smi.stdout.strip() if smi.returncode == 0 else smi.stderr.strip()
        for dims in args.sizes:
            command = [str(binary), *map(str, dims), str(args.iterations), str(args.warmup)]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stderr)
            case = json.loads(result.stdout)
            assert len(case['results']) == 4 and all(row['valid'] for row in case['results'])
            data['cases'].append(case)
            print(f"{dims}: all four implementations passed", flush=True)
        (args.out / 'results.json').write_text(json.dumps(data, indent=2) + '\n')
    rows = report(data, args.out)
    for row in rows:
        print(f"{row['shape']:>14} {row['kernel']:>12}: {row['gflops']:8.1f} GFLOP/s, {row['speedup_vs_naive']:5.2f}x naive")


if __name__ == '__main__':
    main()
