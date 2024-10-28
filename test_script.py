import numpy as np
import time
from linalg_core import LinearAlgebra

print("\n-- TRANSPOSE TEST -- ")

small_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

print("\nInitial:")
for row in small_matrix:
    print(row)

transpose_result_small = LinearAlgebra.transpose(small_matrix)
numpy_transpose_small = np.transpose(small_matrix).tolist()

print("\nLinearAlgebra.T:")
for row in transpose_result_small:
    print(row)

print("\nNumPy.T:")
for row in numpy_transpose_small:
    print(row)

# check
assert transpose_result_small == numpy_transpose_small, "Small match failed"

# big matrix
matrix_size = 10000
matrix = np.random.rand(matrix_size, matrix_size).tolist()

# timer 1
start_time = time.time()
transpose_result_large = LinearAlgebra.transpose(matrix)
linear_algebra_time = time.time() - start_time
print(f"\nLinearAlgebra: {linear_algebra_time:.2f}")

# timer 2
start_time = time.time()
numpy_transpose_large = np.transpose(matrix)
numpy_time = time.time() - start_time
print(f"NumPy: {numpy_time:.2f}")

transpose_result_np_large = np.array(transpose_result_large)

# check
assert np.allclose(
    transpose_result_np_large, numpy_transpose_large, atol=1e-8
), "Large match failed"

print(f"\nDiff: {linear_algebra_time / numpy_time:.2f} times\n")
