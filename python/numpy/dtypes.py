"""
Data Types: https://numpy.org/doc/stable/reference/arrays.dtypes.html
Data Type Promotion: https://numpy.org/doc/stable/reference/arrays.promotion.html

NumPy provides a set of data types (`dtypes`) that extend Python's built-in types, offering more control over memory 
usage and performance. Data types is essential for working efficiently with NumPy arrays, 
especially when dealing with large datasets.

Data type promotion in NumPy refers to how NumPy determines the resulting data type when performing operations 
involving different data types. This process ensures that the operation is carried out safely and accurately.
"""

import numpy as np

int_array = np.array([1, 2, 3], dtype=np.int32)
float_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
complex_array = np.array([1+2j, 3+4j, 2+3j], dtype=np.complex128)

print('integer array: ', int_array)
print('float array: ', float_array)
print('complex array: ', complex_array)

# data type attributes and methods

# basic dtype attributes
print('\ndata type attributes')
print('dtype of int_array: ', int_array.dtype)       # data type of the array elements
print('itemsize of int_array: ', int_array.itemsize) # size in bytes of each array element
print('dtype of float_array: ', float_array.dtype)
print('itemsize of float_array: ', float_array.itemsize)

# conversion between dtypes
print('\ndata type conversion')
converted_array = int_array.astype(np.float64)       # convert int array to float
print('converted array (int to float): ', converted_array)
print('dtype of converted array: ', converted_array.dtype)

# custom dtypes
print('\ncustom data types')
# creating a custom dtype for structured arrays
structured_dtype = np.dtype([('name', np.str_, 10), ('age', 'i4'), ('weight', np.float64, (4,))])
structured_array = np.array([('alice', 25, 55.5), ('bob', 30, 75.3)], dtype=structured_dtype)
print('structured array: \n', structured_array)
print('dtype of structured array: ', structured_array.dtype)


# data type promotion

# automatic promotion in operations
print('\ndata type promotion')
result_array = int_array + float_array   # int32 + float64 -> float64
print('result of int_array + float_array: ', result_array)
print('dtype of result array: ', result_array.dtype)

# example of promotion with complex numbers
complex_result = float_array + complex_array  # float64 + complex128 -> complex128
print('result of float_array + complex_array: ', complex_result)
print('dtype of complex result: ', complex_result.dtype)

# data type promotion with mixed types
mixed_array = np.array([1, 2.5, 3], dtype=np.float32)  # promotes to float32
print('\nmixed type array (int and float): ', mixed_array)
print('dtype of mixed type array: ', mixed_array.dtype)

# # specifying the output dtype to prevent promotion
# no_promotion_array = np.add(int_array, float_array, dtype=np.int32)  # prevent promotion by specifying dtype
# print('\npreventing promotion (specified dtype): ', no_promotion_array)
# print('dtype of no promotion array: ', no_promotion_array.dtype)


# type safety and overflow

# handling overflows in different dtypes
print('\ntype safety and overflow')
large_int = np.array([2**31 - 1], dtype=np.int32)
print('large int array: ', large_int)
try:
    overflowed_int = large_int + 1
except OverflowError as e:
    print('overflow error: ', e)

# promotion to avoid overflow
safe_promotion_array = large_int.astype(np.int64) + 1
print('safe promotion array (int64): ', safe_promotion_array)
print('dtype of safe promotion array: ', safe_promotion_array.dtype)


# more on custom types

# define a custom dtype for stock data
stock_dtype = np.dtype([
    ('ticker', np.str_, 10),      # stock ticker as a string (e.g., 'aapl', 'msft')
    ('date', np.str_, 10),        # date of trade (e.g., '2024-08-18')
    ('open', 'f8'),               # opening price of the stock as a float
    ('close', 'f8'),              # closing price of the stock as a float
    ('volume', 'i8')              # trading volume as an integer
])

# create an array of stock records
stock_data = np.array([
    ('aapl', '2024-08-18', 175.3, 178.6, 15000000),
    ('msft', '2024-08-18', 315.0, 320.1, 12000000),
    ('goog', '2024-08-18', 2820.4, 2855.2, 1000000)
], dtype=stock_dtype)

print('stock data array: \n', stock_data)
print('first stock record: ', stock_data[0])
print('closing price of msft: ', stock_data[1]['close'])


# define a custom dtype for star observation data
star_dtype = np.dtype([
    ('id', 'i4'),           # star id as an integer
    ('name', np.str_, 20),        # name of the star as a string
    ('magnitude', 'f4'),    # apparent magnitude as a float
    ('distance', 'f8'),     # distance from earth in light-years as a float
    ('spectral_type', np.str_, 10)# spectral type (e.g., 'g2v', 'm1v')
])

# create an array of star observations
star_data = np.array([
    (1, 'proxima centauri', 11.05, 4.24, 'm5.5v'),
    (2, 'sirius', -1.46, 8.60, 'a1v'),
    (3, 'betelgeuse', 0.42, 642.5, 'm2ib')
], dtype=star_dtype)

print('\nstar data array: \n', star_data)
print('name of the first star: ', star_data[0]['name'])
print('distance of betelgeuse: ', star_data[2]['distance'])


# define a custom dtype for iot sensor data
iot_dtype = np.dtype([
    ('device_id', np.str_, 15),   # device id as a string
    ('timestamp', np.str_, 19),   # timestamp of the reading (e.g., '2024-08-18 15:30:00')
    ('temperature', 'f4'),  # temperature reading as a float
    ('humidity', 'f4'),     # humidity reading as a float
    ('status', np.str_, 10)       # status of the device (e.g., 'active', 'inactive')
])

# create an array of iot sensor records
iot_data = np.array([
    ('sensor_01', '2024-08-18 15:30:00', 22.5, 45.0, 'active'),
    ('sensor_02', '2024-08-18 15:31:00', 23.1, 46.3, 'active'),
    ('sensor_03', '2024-08-18 15:32:00', 21.9, 44.7, 'inactive')
], dtype=iot_dtype)

print('\niot data array: \n', iot_data)
print('temperature from sensor_01: ', iot_data[0]['temperature'])
print('status of sensor_03: ', iot_data[2]['status'])


# sorting structured arrays by a specific field
sorted_stock_data = np.sort(stock_data, order='volume')
print('\nsorted stock data by volume: \n', sorted_stock_data)

# filtering data based on conditions
high_volume_stocks = stock_data[stock_data['volume'] > 12000000]
print('\nstocks with volume > 12,000,000: \n', high_volume_stocks)

# vectorized operations on specific fields
stock_data['close'] *= 1.05  # increase all closing prices by 5%
print('\nstock data after price increase: \n', stock_data)

# aggregation functions on structured arrays
average_close = np.mean(stock_data['close'])
print('\naverage closing price: ', average_close)
