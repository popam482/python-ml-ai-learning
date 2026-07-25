"""
Series - 1D vector (a single column)
       - has an index for every line

DataFrame - 2D table made out of more Series next to each other
"""

import pandas as pd
import numpy as np

data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Headphones', 'Monitor', 'Laptop', 'Joystick', 'Laptop', 'Laptop'],
    'Category': ['Electronics', 'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Electronics',
                 'Accessories', 'Electronics', 'Electronics'],
    'Price': [3500, 250, 450, 500, 1600, 1200, 300, 2500, 3100],
    'Sales': [10, 0, 18, 9, 15, 12, 0, 4, 10]
}

df = pd.DataFrame(data)
print(df)
print('\n', df.head())  # shows first n lines, or all of them if there is no specification
print('\n', df.tail(2))  # shows last n lines, or all of them if there is no specification
print('\n', df.info())  # complete resume: number of lines, columns, data types and missing values
print('\n Fast statistics: ', df.describe())  # statistics

# .iloc[] integer location from 0 to n-1
# .loc[] label location based on the column name/line tag and boolean conditions

products = df['Product']
print('\n', products)
subset = df[['Product', 'Price']]
print('\n', subset)

sub_table_num = df.iloc[0:3, 0:2]  # first 3 lines and first 2 columns
print('\n', sub_table_num)

sub_table_name = df.loc[0:2, ['Product', 'Price']]
print('\n', sub_table_name)

# filters and masks

mask = (df['Product'] == 'Laptop') & (df['Price'] < 3500)

cheap_laptops = df[mask]
print('\n', cheap_laptops)

# add/change columns

df['New Price'] = df['Price'] * 1.21
print('\n', df)
df['Low sales'] = df['Sales'] < 10
print('\n', df)

elec_mask = (df['Category'] == 'Electronics') & (df['Price'] > 1500)
expensive_electronics = df[elec_mask]
print('\n Expensive electronics:\n', expensive_electronics)

no_stock = df.loc[df['Sales'] == 0, ['Product', 'Sales']]
print('\n', no_stock)

# group and aggregate

data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop', 'Mouse'],
    'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics', 'Accessories'],
    'Price': [3500, 250, 450, 1600, 2500, 150],
    'Sales': [10, 20, 18, 15, 4, 30]
}

df1 = pd.DataFrame(data)
# group by category and calculate price and sales mean
group_mean = df.groupby('Category')[['Price', 'Sales']].mean()
print(group_mean)

# group by category and apply different functions for columns:
# for price we get the maximum
# for sales we get the sum

group_custom = df.groupby('Category').agg({
    'Price': 'max',
    'Sales': 'sum'
})

print('\n', group_custom)

# sort values

# sort by price - most expensive to cheapest
df_sorted = df.sort_values(by='Price', ascending=False)
print('\n values sorted by price: \n', df_sorted)

# sort by category in the first place
# sort bt sales in the second place
df_multi_sort = df.sort_values(by=['Category', 'Sales'], ascending=[True, False])

print('\n Multicriteria sort: \n', df_multi_sort)

# cleaning missing data in pandas

data_missing = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'Price': [3500, np.nan, 450, 1600],
    'Sales': [10, 20, np.nan, 15]
}

data_miss = pd.DataFrame(data_missing)

# check for where there are NaN's - returns true/false

print('\n NaN values: \n', data_miss.isna)

# count how many values are missing on every column

print('\n Missing values: \n', data_miss.isna().sum())

# delete the lines containing at least a NaN
df_clean = data_miss.dropna()

# imputation
# replace NaN from Price with the price mean
# replace NaN from Sales with 0

df_filled = data_miss.copy()
df_filled['Price'] = df_filled['Price'].fillna(df_filled['Price'].mean())
df_filled['Sales'] = df_filled['Sales'].fillna(0)

print('\n Clean table:\n', df_filled)

# tag products as expensive (>1000) or cheap (<=1000)
df['Tag_Price'] = df['Price'].apply(lambda x: 'Expensive' if x > 1000 else 'Cheap')
print('\n Cheap/Expensive tags:\n', df)

data_to_process = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Headphones', 'Monitor', 'Laptop', 'Joystick', 'Laptop', 'Laptop'],
    'Category': ['Electronics', 'Accessories', 'Accessories', 'Accessories', 'Electronics', 'Electronics',
                 'Accessories', 'Electronics', 'Electronics'],
    'Price': [3500, 250, 450, 500, 1600, 1200, 300, 2500, 3100],
    'Sales': [10, 0, 18, 9, 15, 12, 0, 4, 10]
}

df2 = pd.DataFrame(data_to_process)

df2['Revenue'] = df2['Price'] * df2['Sales']

print('\n Revenue table: \n', df2)
print('\n Revenue sum: ', df2['Revenue'].sum())

sort_by_sales_price = df.sort_values(by=['Sales', 'Price'], ascending=[False, True])
print('\n Elements sorted by sales and prices:\n', sort_by_sales_price)

group_mean = df.groupby('Category')['Price'].mean()
result = group_mean[group_mean > 500]

# or using lambda: print(df.groupby('Category')['Price'].mean().loc[lambda x: x > 500])

print('\n', result)

df2['Stock_status'] = df2['Sales'].apply(lambda x: 'No sales' if x == 0 else 'Active')
print('\n Updated table: \n', df2)

# merge & concat
# pd.concat() - concatenate two tables that have the same structure

january_sales = pd.DataFrame({
    'Product': ['Laptop', 'Mouse'],
    'Sales': [12, 15]
})

february_sales = pd.DataFrame({
    'Product': ['Laptop', 'Mouse', 'Keyboard'],
    'Sales': [10, 20, 15]
})

total_sales = pd.concat([january_sales, february_sales], ignore_index=True)
print('\n Total sales: \n', total_sales)

# pd.merge() - merge 2 tables that contain complementary info (common column)
# join types: how='inner' keep the rows that match in both tables
#             how='left' keep all the data from the left table and bring the data from right
#             how='right' keep all the data from the right table
#             how='outer' keep all the data from both tables

df_orders = pd.DataFrame({
    'Order_ID': [101, 102, 103, 104],
    'Product_ID': ['P1', 'P2', 'P1', 'P3'],
    'Quantity': [2, 1, 5, 1]
})

df_products = pd.DataFrame({
    'Product_ID': ['P1', 'P2', 'P4'],
    'Product_name': ['Laptop', 'Mouse', 'Monitor'],
    'Price': [3500, 150, 1200]
})

merge_result = pd.merge(df_orders, df_products, on='Product_ID', how='left')

print('\n Merge result: \n', merge_result)

# pivot tables

df3 = pd.DataFrame({
    'Shop': ['S1', 'S1', 'S2', 'S2'],
    'Category': ['IT', 'Electronics', 'Electronics', 'IT'],
    'Sales': [100, 200, 150, 100]
})

pivot = df3.pivot_table(index='Shop', columns='Category', values='Sales', aggfunc=sum)
print('\n Pivot: ', pivot)

df_employee = pd.DataFrame({
    'Employee_ID': [1, 2, 3, 4, 5],
    'Name': ['Alex', 'John', 'Diana', 'Theo', 'Nate'],
    'Dept_ID': ['D1', 'D2', 'D1', 'D3', 'D2']
})

df_department = pd.DataFrame({
    'Dept_ID': ['D1', 'D2', 'D4'],
    'Department_name': ['IT', 'HR', 'Logistics'],
    'Budget': [50000, 20000, 35000]
})

merge_employee_department = pd.merge(df_employee, df_department, on='Dept_ID', how='left')

print('\n Merge employee department: \n', merge_employee_department)

merge_employee_department['Department_name'] = merge_employee_department['Department_name'].fillna('No department')

print('\n No NaN values: \n', df_department)

inner_join = pd.merge(df_employee, df_department, how='inner')
print('\n Inner join table: \n', inner_join)
