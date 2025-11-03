"""
Optimizing NumPy Implementation for Performance
Advanced techniques for maximizing NumPy performance
"""

import numpy as np
import time
import numba
from numba import jit, vectorize, guvectorize
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

def benchmark_operation(func, *args, **kwargs):
    """
    Benchmark a function's execution time
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def naive_matrix_multiplication(A, B):
    """
    Naive Python implementation of matrix multiplication
    """
    m, n = A.shape
    n, p = B.shape
    result = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += A[i, k] * B[k, j]
    
    return result

def optimized_matrix_multiplication(A, B):
    """
    NumPy optimized matrix multiplication
    """
    return np.dot(A, B)

@jit(nopython=True)
def numba_matrix_multiplication(A, B):
    """
    Numba-accelerated matrix multiplication
    """
    return np.dot(A, B)

@jit(nopython=True, parallel=True)
def parallel_matrix_multiplication(A, B):
    """
    Parallel matrix multiplication using Numba
    """
    return np.dot(A, B)

def memory_efficient_operations():
    """
    Demonstrate memory-efficient NumPy operations
    """
    print("=== MEMORY EFFICIENT OPERATIONS ===\n")
    
    # Create large arrays
    large_array = np.random.rand(1000, 1000)
    
    # 1. In-place operations
    print("1. IN-PLACE OPERATIONS:")
    arr = large_array.copy()
    
    # Regular operation (creates new array)
    start_time = time.time()
    result_regular = arr * 2 + 1
    regular_time = time.time() - start_time
    
    # In-place operation (modifies existing array)
    start_time = time.time()
    arr *= 2
    arr += 1
    inplace_time = time.time() - start_time
    
    print(f"   Regular operation time: {regular_time:.6f}s")
    print(f"   In-place operation time: {inplace_time:.6f}s")
    print(f"   Memory saved: {(result_regular.nbytes / (1024**2)):.2f} MB\n")
    
    # 2. Efficient data types
    print("2. EFFICIENT DATA TYPES:")
    
    # Different data types
    float64_arr = np.ones(1000000, dtype=np.float64)
    float32_arr = np.ones(1000000, dtype=np.float32)
    int32_arr = np.ones(1000000, dtype=np.int32)
    
    print(f"   float64 memory: {float64_arr.nbytes / 1024:.2f} KB")
    print(f"   float32 memory: {float32_arr.nbytes / 1024:.2f} KB")
    print(f"   int32 memory: {int32_arr.nbytes / 1024:.2f} KB")
    print(f"   Memory reduction (float32 vs float64): {(1 - float32_arr.nbytes/float64_arr.nbytes)*100:.1f}%\n")
    
    return large_array

def broadcasting_optimization():
    """
    Demonstrate broadcasting for performance
    """
    print("3. BROADCASTING OPTIMIZATION:")
    
    # Create sample data
    matrix = np.random.rand(1000, 1000)
    vector = np.random.rand(1000)
    
    # Naive approach with loops
    def naive_vector_operation(matrix, vector):
        result = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result[i, j] = matrix[i, j] + vector[j]
        return result
    
    # Broadcasting approach
    def broadcast_vector_operation(matrix, vector):
        return matrix + vector
    
    # Benchmark both
    naive_result, naive_time = benchmark_operation(naive_vector_operation, matrix, vector)
    broadcast_result, broadcast_time = benchmark_operation(broadcast_vector_operation, matrix, vector)
    
    print(f"   Naive loop time: {naive_time:.6f}s")
    print(f"   Broadcasting time: {broadcast_time:.6f}s")
    print(f"   Speedup: {naive_time/broadcast_time:.1f}x")
    print(f"   Results equal: {np.allclose(naive_result, broadcast_result)}\n")

def universal_functions_optimization():
    """
    Optimize using universal functions (ufuncs)
    """
    print("4. UNIVERSAL FUNCTIONS OPTIMIZATION:")
    
    # Create large array
    large_arr = np.random.rand(1000000)
    
    # Custom Python function
    def custom_operation(x):
        return x**2 + np.sin(x) + np.log(np.abs(x) + 1)
    
    # Vectorized NumPy operation
    def vectorized_operation(x):
        return x**2 + np.sin(x) + np.log(np.abs(x) + 1)
    
    # Benchmark
    custom_result, custom_time = benchmark_operation(lambda: np.array([custom_operation(x) for x in large_arr]))
    vectorized_result, vectorized_time = benchmark_operation(vectorized_operation, large_arr)
    
    print(f"   Custom function time: {custom_time:.6f}s")
    print(f"   Vectorized function time: {vectorized_time:.6f}s")
    print(f"   Speedup: {custom_time/vectorized_time:.1f}x")
    print(f"   Results equal: {np.allclose(custom_result, vectorized_result)}\n")

@vectorize(['float64(float64)'], nopython=True)
def numba_vectorized_operation(x):
    """
    Numba-vectorized operation
    """
    return x**2 + np.sin(x) + np.log(np.abs(x) + 1)

def numba_optimization_demo():
    """
    Demonstrate Numba JIT compilation optimization
    """
    print("5. NUMBA JIT OPTIMIZATION:")
    
    large_arr = np.random.rand(1000000)
    
    # Regular Python function
    def python_function(arr):
        result = np.zeros_like(arr)
        for i in range(len(arr)):
            result[i] = arr[i]**2 + np.sin(arr[i]) + np.log(np.abs(arr[i]) + 1)
        return result
    
    # Numba-optimized function
    @jit(nopython=True)
    def numba_function(arr):
        result = np.zeros_like(arr)
        for i in range(len(arr)):
            result[i] = arr[i]**2 + np.sin(arr[i]) + np.log(np.abs(arr[i]) + 1)
        return result
    
    # Benchmark
    python_result, python_time = benchmark_operation(python_function, large_arr)
    numba_result, numba_time = benchmark_operation(numba_function, large_arr)
    numba_vectorized_result, numba_vectorized_time = benchmark_operation(numba_vectorized_operation, large_arr)
    
    print(f"   Pure Python time: {python_time:.6f}s")
    print(f"   Numba JIT time: {numba_time:.6f}s")
    print(f"   Numba vectorized time: {numba_vectorized_time:.6f}s")
    print(f"   Speedup (JIT vs Python): {python_time/numba_time:.1f}x")
    print(f"   Results equal: {np.allclose(python_result, numba_result)}\n")

def memory_layout_optimization():
    """
    Optimize using memory layout (C vs F order)
    """
    print("6. MEMORY LAYOUT OPTIMIZATION:")
    
    # Create large 2D array
    size = 2000
    arr_c = np.zeros((size, size), order='C')  # Row-major (C-style)
    arr_f = np.zeros((size, size), order='F')  # Column-major (Fortran-style)
    
    # Row-wise operations (should be faster with C-order)
    def row_wise_operations(arr):
        result = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result += arr[i, j]
        return result
    
    # Column-wise operations (should be faster with F-order)
    def column_wise_operations(arr):
        result = 0
        for j in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                result += arr[i, j]
        return result
    
    # Benchmark row-wise
    c_row_time = time.time()
    row_wise_operations(arr_c)
    c_row_time = time.time() - c_row_time
    
    f_row_time = time.time()
    row_wise_operations(arr_f)
    f_row_time = time.time() - f_row_time
    
    # Benchmark column-wise
    c_col_time = time.time()
    column_wise_operations(arr_c)
    c_col_time = time.time() - c_col_time
    
    f_col_time = time.time()
    column_wise_operations(arr_f)
    f_col_time = time.time() - f_col_time
    
    print("   Row-wise operations:")
    print(f"     C-order: {c_row_time:.6f}s, F-order: {f_row_time:.6f}s")
    print(f"     C-order is {f_row_time/c_row_time:.2f}x faster")
    
    print("   Column-wise operations:")
    print(f"     C-order: {c_col_time:.6f}s, F-order: {f_col_time:.6f}s")
    print(f"     F-order is {c_col_time/f_col_time:.2f}x faster\n")

def parallel_processing_demo():
    """
    Demonstrate parallel processing with NumPy
    """
    print("7. PARALLEL PROCESSING:")
    
    def process_chunk(chunk):
        """Process a chunk of data"""
        return np.sum(chunk**2 + np.sin(chunk))
    
    def sequential_processing(data, chunk_size):
        """Process data sequentially"""
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            results.append(process_chunk(chunk))
        return np.sum(results)
    
    def parallel_processing(data, chunk_size, num_workers):
        """Process data in parallel"""
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        return np.sum(results)
    
    # Create large dataset
    data = np.random.rand(1000000)
    chunk_size = 10000
    num_workers = mp.cpu_count()
    
    # Benchmark
    seq_result, seq_time = benchmark_operation(sequential_processing, data, chunk_size)
    par_result, par_time = benchmark_operation(parallel_processing, data, chunk_size, num_workers)
    
    print(f"   Sequential processing time: {seq_time:.6f}s")
    print(f"   Parallel processing time: {par_time:.6f}s")
    print(f"   Speedup: {seq_time/par_time:.2f}x")
    print(f"   Number of CPU cores: {num_workers}")
    print(f"   Results equal: {np.allclose(seq_result, par_result)}\n")

def visualize_performance_comparison():
    """
    Create performance comparison visualization
    """
    # Simulated performance data for different optimizations
    optimizations = [
        'Naive Loops', 
        'Vectorization', 
        'In-place Ops', 
        'Efficient Dtypes',
        'Numba JIT',
        'Parallel Processing'
    ]
    
    speedups = [1.0, 50.0, 1.5, 2.0, 100.0, 4.0]  # Relative speedups
    
    plt.figure(figsize=(12, 8))
    
    # Performance comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(optimizations, speedups, color='skyblue', edgecolor='navy')
    plt.title('Performance Speedup Comparison')
    plt.ylabel('Relative Speedup (x)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    # Memory usage comparison
    plt.subplot(2, 2, 2)
    memory_usage = [100, 100, 50, 25, 100, 100]  # Relative memory usage
    plt.bar(optimizations, memory_usage, color='lightcoral', edgecolor='darkred')
    plt.title('Relative Memory Usage')
    plt.ylabel('Memory Usage (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Optimization applicability
    plt.subplot(2, 2, 3)
    applicability = [100, 80, 60, 70, 50, 40] # % of use cases
    plt.pie(applicability, labels=optimizations, autopct='%1.1f%%', startangle=90)
    plt.title('Optimization Applicability')
    
    # Code complexity comparison
    plt.subplot(2, 2, 4)
    complexity = [1, 2, 3, 2, 4, 5]  # Relative complexity (1=easiest)
    plt.plot(optimizations, complexity, 'o-', linewidth=2, markersize=8)
    plt.title('Implementation Complexity')
    plt.ylabel('Complexity (1-5 scale)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numpy_performance_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to demonstrate NumPy performance optimization
    """
    print("=== NUMPY PERFORMANCE OPTIMIZATION DEMONSTRATION ===\n")
    
    # Run all optimization demonstrations
    large_array = memory_efficient_operations()
    broadcasting_optimization()
    universal_functions_optimization()
    numba_optimization_demo()
    memory_layout_optimization()
    parallel_processing_demo()
    
    # Matrix multiplication comparison
    print("8. MATRIX MULTIPLICATION COMPARISON:")
    A = np.random.rand(200, 200)
    B = np.random.rand(200, 200)
    
    # Only run naive for small matrices (it's very slow)
    if A.shape[0] <= 200:
        naive_result, naive_time = benchmark_operation(naive_matrix_multiplication, A, B)
    else:
        naive_time = float('inf')
    
    optimized_result, optimized_time = benchmark_operation(optimized_matrix_multiplication, A, B)
    numba_result, numba_time = benchmark_operation(numba_matrix_multiplication, A, B)
    parallel_result, parallel_time = benchmark_operation(parallel_matrix_multiplication, A, B)
    
    print(f"   Naive (Python loops): {naive_time:.6f}s" if naive_time != float('inf') else "   Naive: Too slow for this size")
    print(f"   NumPy optimized: {optimized_time:.6f}s")
    print(f"   Numba JIT: {numba_time:.6f}s")
    print(f"   Numba parallel: {parallel_time:.6f}s")
    
    if naive_time != float('inf'):
        print(f"   NumPy speedup over naive: {naive_time/optimized_time:.1f}x")
    print(f"   All results equal: {np.allclose(optimized_result, numba_result) and np.allclose(optimized_result, parallel_result)}\n")
    
    # Create visualization
    visualize_performance_comparison()
    
    print("=== OPTIMIZATION SUMMARY ===")
    print("1. Use vectorized operations instead of loops")
    print("2. Prefer in-place operations for large arrays")
    print("3. Choose appropriate data types to save memory")
    print("4. Leverage broadcasting for operations between different shapes")
    print("5. Use Numba JIT compilation for numerical functions")
    print("6. Consider memory layout (C vs F order) for access patterns")
    print("7. Utilize parallel processing for independent operations")
    print("8. Use built-in NumPy functions instead of custom Python implementations")

if __name__ == "__main__":
    main()