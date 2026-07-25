"""

E-COMMERCE STORE ANALYTICS & DATA CLEANING
2 datasets representing an e-commerce platform's raw log: customer transactions & product catalog


"""

import numpy as np
import pandas as pd


def data_cleaning(raw_data):
    df_to_clean = raw_data.copy()
    df_to_clean['Quantity'] = df_to_clean['Quantity'].fillna(df_to_clean['Quantity'].median())
    return df_to_clean


def data_merging(transactions, products):
    merge_result = pd.merge(transactions, products, on='Product_Code', how='left')
    return merge_result


def add_gross_total(df_merged):
    df_merged['Gross_total'] = df_merged['Quantity'] * df_merged['Unit_Price']
    return df_merged


def add_net_total(df_merged):
    df_merged['Net_total'] = df_merged['Gross_total'] * (1 - df_merged['Discount_Pct'])
    return df_merged


def calculate_total_net_revenue(df_merged):
    total_net_revenue = df_merged.groupby('Category').agg({
        'Net_total': 'sum',
    })
    return total_net_revenue

def calculate_top_customer(df_merged):
    top_customer = df_merged.groupby("Customer_ID")["Net_total"].sum().idxmax()
    return top_customer

def create_pivot_table(data_frame):
    pivot_table = data_frame.pivot_table(index='Customer_ID', columns='Category', values='Quantity', aggfunc=sum, fill_value=0)
    return pivot_table


def apply_mask(df_merged):
    df_merged['Value_tag'] = df_merged['Net_total'].apply(lambda x: 'High value' if x > 500 else 'Standard')
    return df_merged


def main():
    # Dataset 1: Customer Transactions
    transactions_data = {
        'Transaction_ID': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'Customer_ID': ['C10', 'C20', 'C10', 'C30', 'C40', 'C20', 'C50', 'C10', 'C30', 'C60'],
        'Product_Code': ['P_LAP', 'P_MOU', 'P_KEY', 'P_LAP', 'P_MON', 'P_KEY', 'P_MOU', 'P_MON', 'P_LAP', 'P_KEY'],
        'Quantity': [1, 2, np.nan, 1, 2, 1, 3, np.nan, 1, 5],
        'Discount_Pct': [0.10, 0.00, 0.05, 0.15, 0.00, 0.00, 0.20, 0.10, 0.00, 0.05]
    }

    # Dataset 2: Product Catalog
    catalog_data = {
        'Product_Code': ['P_LAP', 'P_MOU', 'P_KEY', 'P_MON', 'P_HDP'],
        'Category': ['Hardware', 'Peripherals', 'Peripherals', 'Hardware', 'Peripherals'],
        'Unit_Price': [1200.0, 25.0, 45.0, 300.0, 80.0]
    }

    df_trans = pd.DataFrame(transactions_data)
    df_cat = pd.DataFrame(catalog_data)
    df_trans = data_cleaning(df_trans)
    print('\n Cleared transactions: \n', df_trans)
    df_merged = data_merging(df_trans, df_cat)
    print('\n Merged data: \n', df_merged)
    df_merged = add_gross_total(df_merged)
    print('\n After adding gross total: \n', df_merged)
    df_merged = add_net_total(df_merged)
    print('\n After adding net total: \n', df_merged)
    total_net_revenue = calculate_total_net_revenue(df_merged)
    print('\n Total net revenue: \n', total_net_revenue)
    top_customer = calculate_top_customer(df_merged)
    print('\n Top customer: \n', top_customer)
    pivot_table = create_pivot_table(df_merged)
    print('\n Pivot table: \n', pivot_table)
    df_merged = apply_mask(df_merged)
    print('\n After applying mask: \n', df_merged)


if __name__ == '__main__':
    main()
