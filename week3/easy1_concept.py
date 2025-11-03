"""
NumPy Fundamentals - Key Concept Description

NumPy (Numerical Python) is the fundamental package for scientific computing in Python.
Its core concept revolves around the ndarray (n-dimensional array) object - a homogeneous,
multidimensional container for elements of the same data type.

Key Fundamental Concepts:
1. **ndarray**: The core data structure that enables efficient storage and operations on large datasets
2. **Vectorization**: Operations applied to entire arrays without explicit loops, leveraging optimized C/Fortran code
3. **Broadcasting**: Rules for operating on arrays of different shapes automatically
4. **Universal Functions (ufuncs)**: Fast element-wise operations that work on entire arrays
5. **Memory Efficiency**: Contiguous memory allocation and optimized data types

In essence, NumPy provides the foundation for numerical computation by offering:
- Efficient data structures
- Mathematical functions
- Linear algebra capabilities
- Random number generation
- Tools for integrating with other languages
"""

import numpy as np

def explain_numpy_fundamentals():
    """
    Demonstrate the key concepts of NumPy fundamentals
    """
    print("=== NumPy Fundamentals Explained ===\n")
    
    # 1. ndarray demonstration
    print("1. ndarray - Core Data Structure:")
    arr = np.array([1, 2, 3, 4, 5])
    print(f"   Created array: {arr}")
    print(f"   Shape: {arr.shape}, Dtype: {arr.dtype}")
    print(f"   Dimensions: {arr.ndim}D\n")
    
    # 2. Vectorization demonstration
    print("2. Vectorization - No Explicit Loops:")
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    result = arr1 + arr2  # Vectorized operation
    print(f"   {arr1} + {arr2} = {result}")
    print(f"   Compare to loop: [1+4, 2+5, 3+6] = {result}\n")
    
    # 3. Broadcasting demonstration
    print("3. Broadcasting - Operations on Different Shapes:")
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    scalar = 10
    result = arr + scalar  # Broadcasting scalar to array
    print(f"   Array shape {arr.shape} + scalar {scalar}:")
    print(f"   Result:\n{result}\n")
    
    # 4. Universal Functions demonstration
    print("4. Universal Functions (ufuncs):")
    arr = np.array([1, 4, 9, 16])
    sqrt_result = np.sqrt(arr)  # ufunc applied element-wise
    print(f"   sqrt({arr}) = {sqrt_result}\n")
    
    # 5. Memory efficiency
    print("5. Memory Efficiency:")
    large_arr = np.arange(1000000)  # 1 million elements
    print(f"   Created array with 1,000,000 elements")
    print(f"   Memory usage: {large_arr.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Data type: {large_arr.dtype}")

if __name__ == "__main__":
    explain_numpy_fundamentals()