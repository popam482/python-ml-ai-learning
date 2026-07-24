import numpy as np

# numbers squared

numbers_vec = np.array([1, 2, 3, 4, 5])

squared_vec = numbers_vec ** 2

print(squared_vec)

# data filter

data = np.array([10, 12, 9, 2, 34, 53, 7, 44])

mask = data > 10
print(mask)

result = data[data > 10]
print(result)

# final price for each product

prices = np.array([150, 45, 300, 12, 85, 210])
quantity = np.array([2, 10, 1, 5, 0, 3])

final_prices = prices * quantity

print(final_prices)

# apply 10% discount for prices > 100

discount = np.where(prices > 100, prices * 0.9, prices)  # condition, value for true, value for false

print(discount)

# final sum for all the products
final_sum = np.sum(discount)

print(final_sum)

# sum, mean, median, min, max, std

sales = np.array([100, 200, 900, 450])

print(np.sum(sales))
print(np.mean(sales))
print(np.median(sales))
print(np.min(sales))
print(np.max(sales))
print(np.std(sales))

# np.argmax(array) - index of the max value
# np.argmin(array) - index of the min value

print("most expensive: ", np.argmax(sales))
print("cheapest: ", np.argmin(sales))

# np.logical_and()
# np.logica_or()

prices1 = np.array([150, 45, 300, 50, 100, 650])
# without np.logical_and
filter1 = (prices > 50) & (prices < 400)

# with np.logical_and
filter2 = np.logical_and(prices > 50, prices < 400)
print(prices1[filter2])

# np.any(cond) returns true if at least one element satisfies cond
# np.all(cond) returns true if all elements satisfies cond

stock = np.array([10, 0, 8, 3, 0, 0, 20])
print(np.any(stock == 0))
print(np.all(stock != 0))

# np.round(array, decimals)
# np.clip(array, min, max) - limits the values in an array e.g. if an interval of [0,1] is defined, values smaller than 0 become 0, values
#                            larger than 1 become 1

grades = np.array([3, 7, 6, 11, -2, 9])
valid_grades = np.clip(grades, 1, 10)
print(valid_grades)

# broadcasting
prices2 = np.array([100, 200, 300])
prices_with_tax = prices2 + 50
print(prices_with_tax)

prices3 = np.array([100, 200, 300])
discount_per = np.array([0.2, 0.1, 0.15])
final_prices3 = prices3 * (1 - discount_per)
print(final_prices3)

# fancy indexing
stocks = np.array([10, 5, 0, 20, 0, 10, 0])
stocks[stocks == 0] = 10
print(stocks)

prices4 = np.array([100, 150, 50, 200, 350])
target_indexes = [0, 2, 4]
prices4[target_indexes] += 50
print(prices4)

# 1D final
daily_sales = np.array([120, 0, 45, 0, 300, 150, 0, 80])
daily_sales[daily_sales == 0] = np.mean(daily_sales[daily_sales != 0])
print(daily_sales)

target_indexes1 = [0, 4, 7]
new_vector = daily_sales[target_indexes1]
print(new_vector)

daily_sales = daily_sales * (1 - 0.19)

print(daily_sales)

# 2D

matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

print(matrix.shape)  # (lines, columns)
print(matrix.ndim)  # lines * columns

print(matrix[1, 2])  # single element
print(matrix[0, :])  # whole line
print(matrix[:, 1])  # whole column

sub_matrix = matrix[0:2, 1:3]  # lines 0 - 1 columns 1 - 2
print(sub_matrix)

# numpy axis
# axis = 0 vertically
# axis = 1 horizontally

print(np.sum(matrix))  # summ for all the matrix
print(np.sum(matrix, axis=0))  # sum for each column
print(np.sum(matrix, axis=1))  # sum for each line

# np.reshape(lines, columns) 1D -> 2D
# np.flatten() 2D -> 1D
data_1d = np.arange(1, 13)
print(data_1d)

matrix_2d = data_1d.reshape(3, 4)
print(matrix_2d)

vector = matrix_2d.flatten()
print(vector)

# lines - shops
# columns - trimesters

shop_sales = np.array([
    [1500, 2200, 1800, 3100],
    [800, 950, 1200, 1100],
    [2100, 2400, 3000, 3800]
])

print(np.sum(shop_sales, axis=1))
print(np.argmax(np.mean(shop_sales, axis=0)))

matrix1 = shop_sales[[0, 2], 2:]

print(matrix1)

shop_sales_vector = shop_sales.flatten()
shop_sales_matrix = shop_sales_vector.reshape(6, 2)

# broadcasting
expenses_q = np.array([500, 600, 500, 700])
profit = shop_sales - expenses_q

print('\n', profit)

shop_tax = np.array([100, 50, 200]).reshape(3, 1)
net_profit = profit - shop_tax
print(net_profit)

# np.vdstack() - adds new lines
# np.hstack() - adds new columns
# np.concatenate() -  combines multiple arrays into a single array along a specified axis

import numpy as np

year_2023 = np.array([
    [100, 200],
    [300, 400]
])

year_2024 = np.array([
    [150, 250],
    [350, 450]
])

vert = np.vstack((year_2023, year_2024))
print('\n', vert)

horiz = np.hstack((year_2023, year_2024))
print('\n', horiz)

# np.where()

matrix_mask = np.array([
    [10, 85, 120],
    [45, 200, 15]
])

result = np.where(matrix_mask < 50, 0, matrix_mask)  # target on values < 50 and we replace them with 0

print(result)

# 2D exercise

# lines - students
# columns - subject (maths, physics, informatics)

st_grades = np.array([
    [8.5, 4.0, 9.0],
    [5.0, 3.5, 6.0],
    [9.0, 8.0, 10.0],
    [4.0, 5.5, 4.5]
])

st_grades[:, 1] += 1.0

print('\n', st_grades)

new_st = np.array([7.0, 6.5, 8.0])

st_grades = np.vstack((st_grades, new_st))

print('\n', st_grades)

final_grades = np.where(st_grades < 5.0, 4.0, st_grades)
print('\n', final_grades)

best_students = np.argmax(final_grades, axis=0)

print('\n', best_students)

