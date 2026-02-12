import timeit
import numpy as np
from numba import types
from typing import Callable

# ============================================================================
# Implementation 1: Original (two separate functions)
# ============================================================================
def create_scalar_mean_v1(a):
    """Original implementation with separate function definitions."""
    if isinstance(a, (int, bool)):
        def _scalar_mean(a):
            return np.float64(a) + 0.0
    else:
        def _scalar_mean(a):
            return a + 0.0
    return _scalar_mean

# ============================================================================
# Implementation 2: Closure with ternary
# ============================================================================
def create_scalar_mean_v2(a):
    """Closure with ternary operator."""
    is_int_or_bool = isinstance(a, (int, bool))
    
    def _scalar_mean(a):
        return (np.float64(a) if is_int_or_bool else a) + 0.0
    return _scalar_mean

# ============================================================================
# Implementation 3: Lambda approach
# ============================================================================
def create_scalar_mean_v3(a):
    """Lambda cast approach."""
    cast = np.float64 if isinstance(a, (int, bool)) else lambda x: x
    
    def _scalar_mean(a):
        return cast(a) + 0.0
    return _scalar_mean

# ============================================================================
# Test inputs
# ============================================================================
test_inputs = {
    'int_42': 42,
    'int_0': 0,
    'int_negative': -42,
    'bool_true': True,
    'bool_false': False,
    'float_pos': 3.14159,
    'float_neg': -2.71828,
    'float_zero': 0.0,
    'float_neg_zero': -0.0,
    'float_large': 1e100,
    'float_small': 1e-100,
    'np_float32': np.float32(1.5),
    'np_float64': np.float64(2.5),
}

# ============================================================================
# Benchmarking function
# ============================================================================
def benchmark_implementation(impl_func: Callable, name: str, iterations: int = 1_000_000):
    """Benchmark a single implementation across all test inputs."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    results = {}
    
    for input_name, input_value in test_inputs.items():
        # Create the scalar_mean function once
        scalar_mean = impl_func(input_value)
        
        if scalar_mean is None:
            print(f"{input_name:20s}: ERROR - Function returned None!")
            continue
        
        # Verify correctness first
        result = scalar_mean(input_value)
        
        # Time the execution
        timer = timeit.Timer(
            stmt='scalar_mean(value)',
            globals={'scalar_mean': scalar_mean, 'value': input_value}
        )
        
        time_taken = timer.timeit(number=iterations)
        results[input_name] = time_taken
        
        print(f"{input_name:20s}: {time_taken:8.4f}s  |  Result: {result:15.10f}")
    
    if results:
        avg_time = sum(results.values()) / len(results)
        print(f"\n{'Average time:':20s}  {avg_time:8.4f}s")
    else:
        avg_time = float('inf')
    
    return results, avg_time

# ============================================================================
# Run benchmarks
# ============================================================================
def run_all_benchmarks(iterations: int = 1_000_000):
    """Run benchmarks for all implementations."""
    print(f"\n🔬 PERFORMANCE BENCHMARK")
    print(f"Iterations per test: {iterations:,}")
    print(f"NumPy version: {np.__version__}")
    
    implementations = [
        (create_scalar_mean_v1, "V1: Original (separate functions)"),
        (create_scalar_mean_v2, "V2: Closure with ternary"),
        (create_scalar_mean_v3, "V3: Lambda cast approach"),
    ]
    
    all_results = {}
    all_averages = {}
    
    for impl_func, name in implementations:
        results, avg = benchmark_implementation(impl_func, name, iterations)
        all_results[name] = results
        all_averages[name] = avg
    
    # ========================================================================
    # Summary comparison
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    # Sort by performance
    sorted_impls = sorted(all_averages.items(), key=lambda x: x[1])
    
    print(f"\n{'Rank':<6} {'Implementation':<40} {'Avg Time (s)':<15} {'vs Fastest'}")
    print(f"{'-'*70}")
    
    fastest_time = sorted_impls[0][1]
    
    for rank, (name, avg_time) in enumerate(sorted_impls, 1):
        speedup = avg_time / fastest_time
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        percent_slower = (speedup - 1) * 100
        print(f"{emoji} {rank:<4} {name:<40} {avg_time:<15.4f} {speedup:.3f}x (+{percent_slower:.1f}%)")
    
    # ========================================================================
    # Per-input comparison
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"📈 PER-INPUT BREAKDOWN")
    print(f"{'='*70}\n")
    
    print(f"{'Input Type':<20} {'V1 (s)':<12} {'V2 (s)':<12} {'V3 (s)':<12} {'Winner'}")
    print(f"{'-'*70}")
    
    for input_name in test_inputs.keys():
        times = []
        for impl_name in ["V1: Original (separate functions)", 
                          "V2: Closure with ternary", 
                          "V3: Lambda cast approach"]:
            if input_name in all_results[impl_name]:
                times.append(all_results[impl_name][input_name])
            else:
                times.append(float('inf'))
        
        v1, v2, v3 = times
        winner_idx = times.index(min(times))
        winner = ["V1", "V2", "V3"][winner_idx]
        
        print(f"{input_name:<20} {v1:<12.4f} {v2:<12.4f} {v3:<12.4f} {winner}")
    
    # ========================================================================
    # Type-based breakdown
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"📊 TYPE-BASED PERFORMANCE")
    print(f"{'='*70}\n")
    
    type_groups = {
        'Integer': ['int_42', 'int_0', 'int_negative'],
        'Boolean': ['bool_true', 'bool_false'],
        'Float': ['float_pos', 'float_neg', 'float_zero', 'float_neg_zero'],
        'NumPy Float': ['np_float32', 'np_float64'],
    }
    
    for type_name, input_names in type_groups.items():
        print(f"\n{type_name}:")
        for impl_name in all_results.keys():
            type_times = [all_results[impl_name][inp] 
                         for inp in input_names 
                         if inp in all_results[impl_name]]
            if type_times:
                avg = sum(type_times) / len(type_times)
                print(f"  {impl_name:<45} {avg:.4f}s")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Quick test (100K iterations)
    print("Running benchmark (100K iterations)...")
    run_all_benchmarks(iterations=100_000)
    
    # Uncomment for more thorough test
    print("\n" + "="*70)
    print("Running thorough benchmark (1M iterations)...")
    run_all_benchmarks(iterations=1_000_000)