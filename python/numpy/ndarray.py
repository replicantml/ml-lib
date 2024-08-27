"""
Everything You Need to Know About NumPy `ndarray`
Reference: https://numpy.org/doc/stable/reference/arrays.ndarray.html

An `ndarray` is a (usually fixed-size) multidimensional container of items of the same type and size.
The number of dimensions and items in an array is defined by its shape, which is a tuple of non-negative integers
specifying the sizes of each dimension. The type of items in the array is specified by a data-type object (`dtype`),
one of which is associated with each `ndarray`.
"""

import numpy as np

# example: creating a numpy `ndarray`
x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
print('array: \n', x)

# array attributes

# memory layout
print('flags: \n', x.flags)        # flags set
print('shape: ', x.shape)          # shape (rows, cols) or (dim n, dim n - 1, ..., dim 1)
print('strides: ', x.strides)      # tuple of bytes to step in each dimension when traversing an array
print('dims: ', x.ndim)            # number of array dimensions
print('data: ', x.data)            # python buffer object pointing to the start of the array's data
print('size: ', x.size)            # number of elements in the array
print('itemsize: ', x.itemsize)    # length of one array element in bytes
print('nbytes: ', x.nbytes)        # total bytes consumed by the elements of the array
print('base: ', x.base)            # base object if memory is from some other object

# data type
print('data type: ', x.dtype)      # data type

# other attributes
print('transpose: \n', x.T)        # transpose
print('real: \n', x.real)          # real part
print('imaginary: \n', x.imag)     # imaginary part
print([a for a in x.flat])         # flat view


# array methods

# conversion: https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-conversion
print('\n# conversion methods')
print('to list: ', x.tolist())     # convert array to a python list
# print('to bytes: ', x.tobytes())   # convert array to a bytes object
# print('to file: ')                 # write array to a file
# x.tofile('array.bin')
# print('to string: ', x.tostring()) # convert array to a string representation

# shape manipulation: https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-shape-manipulation
print('\n# shape manipulation methods')
print('reshaped: \n', x.reshape(3, 2))  # reshape the array to 3x2
print('flattened: ', x.flatten())       # flatten the array to 1d
x.resize(3, 2)
print('resized: \n', x)    # resize the array in-place
print('swapped axes: \n', x.swapaxes(0, 1))  # swap axes

# other methods: https://numpy.org/doc/stable/reference/arrays.ndarray.html#other-methods
print('\n# other useful methods')
print('sum: ', x.sum())                # sum of array elements
print('cumulative sum: ', x.cumsum())  # cumulative sum of array elements
print('min: ', x.min())                # minimum element in the array
print('max: ', x.max())                # maximum element in the array
print('mean: ', x.mean())              # mean of the array elements
print('standard deviation: ', x.std()) # standard deviation of the array elements
print('copy: \n', x.copy())            # return a copy of the array
print('view: \n', x.view())            # create a view of the array with the same data
