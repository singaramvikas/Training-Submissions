"""
Toy Example Applying NumPy Fundamentals
Basic operations demonstrating core NumPy concepts
"""

import numpy as np

def numpy_toy_example():
    """
    A comprehensive toy example showing various NumPy fundamental operations
    """
    print("=== NumPy Fundamentals Toy Example ===\n")
    
    # 1. Creating arrays
    print("1. Array Creation:")
    # From Python list
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"   From list: {arr1}")
    
    # Using built-in functions
    zeros = np.zeros(5)
    ones = np.ones(5)
    range_arr = np.arange(0, 10, 2)  # 0 to 10, step 2
    linspace_arr = np.linspace(0, 1, 5)  # 5 numbers from 0 to 1
    
    print(f"   Zeros: {zeros}")
    print(f"   Ones: {ones}")
    print(f"   Arange: {range_arr}")
    print(f"   Linspace: {linspace_arr}\n")
    
    # 2. Array operations
    print("2. Array Operations:")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    print(f"   Arrays: a={a}, b={b}")
    print(f"   Addition: {a + b}")
    print(f"   Subtraction: {a - b}")
    print(f"   Multiplication: {a * b}")
    print(f"   Division: {b / a}")
    print(f"   Dot product: {np.dot(a, b)}\n")
    
    # 3. Matrix operations
    print("3. Matrix Operations:")
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    
    print(f"   Matrix A:\n{matrix_a}")
    print(f"   Matrix B:\n{matrix_b}")
    print(f"   Matrix multiplication:\n{np.matmul(matrix_a, matrix_b)}")
    print(f"   Element-wise multiplication:\n{matrix_a * matrix_b}")
    print(f"   Transpose of A:\n{matrix_a.T}\n")
    
    # 4. Statistical operations
    print("4. Statistical Operations:")
    data = np.array([2, 5, 1, 8, 3, 9, 4, 7, 6])
    print(f"   Data: {data}")
    print(f"   Mean: {np.mean(data):.2f}")
    print(f"   Median: {np.median(data):.2f}")
    print(f"   Standard deviation: {np.std(data):.2f}")
    print(f"   Min: {np.min(data)}, Max: {np.max(data)}")
    print(f"   Sum: {np.sum(data)}\n")
    
    # 5. Indexing and slicing
    print("5. Indexing and Slicing:")
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"   2D array:\n{arr_2d}")
    print(f"   Element at [1,2]: {arr_2d[1, 2]}")
    print(f"   First row: {arr_2d[0, :]}")
    print(f"   Second column: {arr_2d[:, 1]}")
    print(f"   Sub-array (first 2 rows, last 2 columns):\n{arr_2d[:2, 1:]}\n")
    
    # 6. Boolean indexing
    print("6. Boolean Indexing:")
    numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    even_numbers = numbers[numbers % 2 == 0]
    greater_than_5 = numbers[numbers > 5]
    
    print(f"   Numbers: {numbers}")
    print(f"   Even numbers: {even_numbers}")
    print(f"   Numbers > 5: {greater_than_5}\n")
    
    # 7. Reshaping and manipulation
    print("7. Reshaping and Manipulation:")
    flat_arr = np.arange(12)
    reshaped = flat_arr.reshape(3, 4)
    flattened = reshaped.flatten()
    
    print(f"   Original: {flat_arr}")
    print(f"   Reshaped to 3x4:\n{reshaped}")
    print(f"   Flattened back: {flattened}")

def solve_linear_equation():
    """
    Solve a system of linear equations using NumPy
    Example: 2x + 3y = 8
             4x - y = 2
    """
    print("\n=== Solving Linear Equations ===")
    
    # Coefficients matrix
    A = np.array([[2, 3], [4, -1]])
    # Constants vector
    b = np.array([8, 2])
    
    # Solve Ax = b
    solution = np.linalg.solve(A, b)
    
    print(f"   Equations: 2x + 3y = 8, 4x - y = 2")
    print(f"   Coefficient matrix A:\n{A}")
    print(f"   Constants vector b: {b}")
    print(f"   Solution: x = {solution[0]:.2f}, y = {solution[1]:.2f}")
    
    # Verify solution
    verification = np.dot(A, solution)
    print(f"   Verification (A * solution): {verification} ≈ {b}")

if __name__ == "__main__":
    numpy_toy_example()
    solve_linear_equation()