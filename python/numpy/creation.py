import numpy as np

# Dimen: (3, 2, 1)-d
# Shape: (a, b, c)
# Axes
#   1d:  axis-0 -> along the row
#   2d:  axis-0 -> along the col, axis-1 -> along the row

# Empty (uninitialized)
empty = np.empty((2, 3), dtype=int)
print('Empty: \n', empty)

empty_like = np.empty_like([[1, 2, 3], [0, 0, 1], [1, 1, 1]], dtype=np.float16)
print('Empty Like: \n', empty_like)

# Eye (1's on 2d diagonals)
eye = np.eye(4, 4)
print('Eye: \n', eye)

# Identity (Identity nxn)
identity = np.eye(2)
print('Identity: \n', identity)

# Ones
ones = np.ones((1, 1, 3), dtype=int)
print('Ones: \n', ones)

ones_like = np.ones_like([[1, 2, 3], [0, 0, 1], [1, 1, 1]], dtype=np.float16)
print('Ones Like: \n', ones_like)

# Zeros
zeros = np.zeros((1, 1, 3), dtype=int)
print('Zeros: \n', zeros)

zeros_like = np.zeros_like([[1, 2, 3], [0, 0, 1], [1, 1, 1]], dtype=np.float16)
print('Zeros Like: \n', zeros_like)

# Full
full = np.full((1, 3), 12.2)
print('Full: \n', full)

full_like = np.full_like([[1, 2, 3], [0, 0, 1], [1, 1, 1]], 9)
print('Full Like: \n', full_like)

# array, asarray

# Ranges

# Arange (auto generate numbers w/ equal steps)
step = np.arange(0, 5, step=0.5)
print('Step: \n', step)

stop = np.arange(5, 10)
print('Stop: \n', stop)

# Linspace (auto calculates step size, endpoint=True)
linspace = np.linspace(2, 7, num=5, retstep=True)
print('Linspace: \n', linspace)

# Logspace
logspace = np.logspace(2.0, 3.0, num=4, base=[2.0, 10.0], axis=0)
print('Logspace: \n', logspace)