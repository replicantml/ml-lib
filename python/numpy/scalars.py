"""
https://numpy.org/doc/stable/reference/arrays.scalars.html

NumPy scalars are data types that correspond to Python's built-in types but extend them with additional functionality.
These include data types like `np.int32`, `np.float64`, and `np.complex128`, which mirror their corresponding
Python types but are optimized for use with NumPy arrays.

Scalars are single values (as opposed to arrays or sequences), and they allow for consistent behavior when working
with numerical data in NumPy, especially in operations that involve both arrays and scalar values.
"""

import numpy as np

int_scalar = np.int32(10)
float_scalar = np.float64(3.14)
complex_scalar = np.complex128(1 + 2j)

print('integer scalar: ', int_scalar)
print('float scalar: ', float_scalar)
print('complex scalar: ', complex_scalar)


# scalar attributes and methods

# basic attributes
print('\n# basic attributes')
print('type of int_scalar: ', type(int_scalar))       # type of the scalar
print('shape of int_scalar: ', int_scalar.shape)      # scalars have an empty shape tuple
print('itemsize of int_scalar: ', int_scalar.itemsize)  # size of the scalar in bytes

# conversion to python native types
print('\n# conversion methods')
print('as python int: ', int_scalar.item())           # convert to python int
print('as python float: ', float_scalar.item())       # convert to python float
print('as python complex: ', complex_scalar.item())   # convert to python complex

# casting between scalar types
print('\n# casting methods')
print('int to float: ', int_scalar.astype(np.float64))   # cast int_scalar to float
print('float to complex: ', float_scalar.astype(np.complex128))  # cast float_scalar to complex

# arithmetic operations
print('\n# arithmetic operations')
sum_scalar = int_scalar + float_scalar
product_scalar = float_scalar * complex_scalar
print('sum of int_scalar and float_scalar: ', sum_scalar)
print('product of float_scalar and complex_scalar: ', product_scalar)

# special scalar methods
print('\n# special methods')
print('imaginary part of complex_scalar: ', complex_scalar.imag)  # imaginary part of complex scalar
print('real part of complex_scalar: ', complex_scalar.real)        # real part of complex scalar
print('absolute value of complex_scalar: ', abs(complex_scalar))   # absolute value (magnitude)


# arrays of scalars

# creating an array of numpy scalars
scalar_array = np.array([int_scalar, float_scalar, complex_scalar])
print('\n# arrays of scalars')
print('array of scalars: \n', scalar_array)

# operations on arrays of scalars
print('sum of array elements: ', scalar_array.sum())  # sum of all elements
print('mean of array elements: ', scalar_array.mean())  # mean of all elements


# scalar broadcasting

# broadcasting scalars with arrays
print('\n# scalar broadcasting')
array = np.array([1, 2, 3])
broadcast_sum = array + int_scalar  # int_scalar is broadcast across the array
print('array + int_scalar: ', broadcast_sum)

broadcast_product = array * float_scalar  # float_scalar is broadcast across the array
print('array * float_scalar: ', broadcast_product)


# scalar and array interaction

# scalars interacting with multi-dimensional arrays
multi_dim_array = np.array([[1, 2, 3], [4, 5, 6]])
broadcast_multi_dim = multi_dim_array + complex_scalar
print('\n# scalars with multi-dimensional arrays')
print('multi-dimensional array + complex_scalar: \n', broadcast_multi_dim)

# comparison with python built-in types

# differences between numpy scalars and python native types
python_int = 10
numpy_int = np.int32(10)
print('\n# numpy scalars vs python native types')
print('python int + 5: ', python_int + 5)
print('numpy int32 + 5: ', numpy_int + 5)

# handling of large numbers (numpy vs python)
large_numpy_int = np.int64(2**63 - 1)
print('large numpy int64 + 1: ', large_numpy_int + 1)  # overflow handling
