#!/usr/bin/env python3
"""
NumPy Scalar Types Checker
This script explores all NumPy scalar types, creates sample arrays,
and checks their mean values and resulting types.
"""

import numpy as np
import sys

def get_all_numpy_scalar_types():
    """Get all NumPy scalar types organized by category."""
    scalar_types = {
        'Boolean': [
            np.bool_,
        ],
        'Signed Integer': [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ],
        'Unsigned Integer': [
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ],
        'Floating Point': [
            np.float16,
            np.float32,
            np.float64,
        ],
        'Complex': [
            np.complex64,
            np.complex128,
        ],
        'String/Bytes': [
            np.str_,
            np.bytes_,
        ],
        'Other': [
            np.object_,
            np.datetime64,
            np.timedelta64,
        ]
    }
    
    # Add platform-dependent types if different
    if np.int_ not in [np.int8, np.int16, np.int32, np.int64]:
        scalar_types['Signed Integer'].append(np.int_)
    if np.uint not in [np.uint8, np.uint16, np.uint32, np.uint64]:
        scalar_types['Unsigned Integer'].append(np.uint)
    
    return scalar_types

def test_scalar_type(dtype, sample_values):
    """Test a specific NumPy scalar type."""
    try:
        # Create array with the dtype
        arr = np.array(sample_values, dtype=dtype)
        
        # Calculate mean
        try:
            mean_val = np.mean(arr)
            mean_type = type(mean_val).__name__
            mean_dtype = mean_val.dtype if hasattr(mean_val, 'dtype') else 'N/A'
        except (TypeError, ValueError) as e:
            mean_val = f"Error: {e}"
            mean_type = "N/A"
            mean_dtype = "N/A"
        
        return {
            'dtype': dtype,
            'dtype_name': dtype.__name__,
            'array_dtype': arr.dtype,
            'sample_array': arr,
            'mean_value': mean_val,
            'mean_type': mean_type,
            'mean_dtype': mean_dtype,
            'success': True
        }
    except Exception as e:
        return {
            'dtype': dtype,
            'dtype_name': dtype.__name__,
            'error': str(e),
            'success': False
        }

def main():
    """Main function to test all NumPy scalar types."""
    print("=" * 80)
    print("NumPy Scalar Types Analysis")
    print("=" * 80)
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    print("=" * 80)
    print()
    
    scalar_types = get_all_numpy_scalar_types()
    
    for category, dtypes in scalar_types.items():
        print(f"\n{'=' * 80}")
        print(f"Category: {category}")
        print(f"{'=' * 80}\n")
        
        for dtype in dtypes:
            # Choose appropriate sample values based on type
            if category == 'Boolean':
                sample_values = [True, False, True, True, False]
            elif category in ['Signed Integer', 'Unsigned Integer']:
                sample_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            elif category == 'Floating Point':
                sample_values = [1.5, 2.7, 3.2, 4.8, 5.1]
            elif category == 'Complex':
                sample_values = [1+2j, 2+3j, 3+4j, 4+5j]
            elif category == 'String/Bytes':
                if dtype == np.str_:
                    sample_values = ['apple', 'banana', 'cherry']
                else:
                    sample_values = [b'apple', b'banana', b'cherry']
            elif category == 'Other':
                if dtype == np.object_:
                    sample_values = [1, 2, 3, 4, 5]
                elif dtype == np.datetime64:
                    sample_values = ['2024-01-01', '2024-01-02', '2024-01-03']
                elif dtype == np.timedelta64:
                    sample_values = [1, 2, 3, 4, 5]
                else:
                    sample_values = [1, 2, 3, 4, 5]
            else:
                sample_values = [1, 2, 3, 4, 5]
            
            result = test_scalar_type(dtype, sample_values)
            
            if result['success']:
                print(f"Type: {result['dtype_name']}")
                print(f"  Array dtype: {result['array_dtype']}")
                print(f"  Sample array: {result['sample_array'][:5]}...")
                print(f"  Mean value: {result['mean_value']}")
                print(f"  Mean type: {result['mean_type']}")
                print(f"  Mean dtype: {result['mean_dtype']}")
                print()
            else:
                print(f"Type: {result['dtype_name']}")
                print(f"  Error: {result['error']}")
                print()
    
    # Additional analysis: Type promotion in mean calculation
    print("\n" + "=" * 80)
    print("Type Promotion Analysis (What happens to types when computing mean)")
    print("=" * 80 + "\n")
    
    test_types = [
        (np.int8, [1, 2, 3, 4, 5]),
        (np.int16, [1, 2, 3, 4, 5]),
        (np.int32, [1, 2, 3, 4, 5]),
        (np.int64, [1, 2, 3, 4, 5]),
        (np.uint8, [1, 2, 3, 4, 5]),
        (np.float16, [1.0, 2.0, 3.0]),
        (np.float32, [1.0, 2.0, 3.0]),
        (np.float64, [1.0, 2.0, 3.0]),
        (np.complex64, [1+1j, 2+2j]),
        (np.complex128, [1+1j, 2+2j]),
    ]
    
    for dtype, values in test_types:
        arr = np.array(values, dtype=dtype)
        mean_val = np.mean(arr)
        print(f"{dtype.__name__:15} -> mean dtype: {str(mean_val.dtype):15} | "
              f"mean value: {mean_val}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()