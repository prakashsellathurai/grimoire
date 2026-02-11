import numpy as np
import time
from numba import njit, types
from numba.extending import register_jitable

# ---------------------------------------------------------
# IMPLEMENTATIONS
# ---------------------------------------------------------

@njit
def mean_manual(a):
    if isinstance(a, (int, float, bool, np.number)):
        if a == -0.0:
            a = 0.0
        return np.float64(a)
    else:
        return np.mean(a)

@njit
def mean_unified(a):
    arr = np.asarray(a)
    # Using a slightly more robust mean logic for the unified path
    if arr.size == 0:
        return np.nan
    return arr.sum() / arr.size

@register_jitable
def _canonicalize(val):
    return 0.0 if val == -0.0 else np.float64(val)

@njit
def mean_helper(a):
    if isinstance(a, (int, float, bool, np.number)):
        return _canonicalize(a)
    return np.mean(a)

# ---------------------------------------------------------
# RANKED BENCHMARK SCRIPT
# ---------------------------------------------------------

def run_benchmarks():
    scalar_val = -0.0
    array_val = np.random.rand(1000)
    iterations = 1_000_000
    results = []

    # Warm up
    for m in [mean_manual, mean_unified, mean_helper]:
        m(scalar_val)
        m(array_val)

    methods = [
        ("Manual Dispatch", mean_manual),
        ("Unified (asarray)", mean_unified),
        ("Jitable Helper", mean_helper)
    ]

    for name, func in methods:
        # Benchmark Scalar
        start = time.perf_counter()
        for _ in range(iterations):
            func(scalar_val)
        t_scalar = time.perf_counter() - start

        # Benchmark Array
        start = time.perf_counter()
        for _ in range(iterations):
            func(array_val)
        t_array = time.perf_counter() - start

        results.append({
            "name": name,
            "scalar": t_scalar,
            "array": t_array
        })

    # Sort by Scalar performance (Ascending - lower is better)
    results.sort(key=lambda x: x['scalar'])

    print(f"\n{'Rank':<5} | {'Method':<20} | {'Scalar (1M)':<15} | {'Array 1k (1M)':<15}")
    print("-" * 65)
    
    for i, res in enumerate(results, 1):
        print(f"#{i:<4} | {res['name']:<20} | {res['scalar']:>13.4f}s | {res['array']:>13.4f}s")

if __name__ == "__main__":
    run_benchmarks()