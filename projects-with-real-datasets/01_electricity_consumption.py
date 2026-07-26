"""
NATIONAL GRID POWER ANALYTICS ENGINE

- energy independence analysis to see when Romania needed to import/export energy

- green foot vs fossil to see the energetic transition

"""

import pandas as pd


def read_dataframe():
    df = pd.read_csv('../datasets/electricityConsumptionAndProductioction.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    return df


def clear_data(df):
    df_to_clean = df.copy()
    df_to_clean['Nuclear'] = df_to_clean['Nuclear'].fillna(df_to_clean['Nuclear'].median())
    df_to_clean['Wind'] = df_to_clean['Wind'].fillna(df_to_clean['Wind'].median())
    df_to_clean['Hydroelectric'] = df_to_clean['Hydroelectric'].fillna(df_to_clean['Hydroelectric'].median())
    df_to_clean['Oil and Gas'] = df_to_clean['Oil and Gas'].fillna(df_to_clean['Oil and Gas'].median())
    df_to_clean['Coal'] = df_to_clean['Coal'].fillna(df_to_clean['Coal'].median())
    df_to_clean['Solar'] = df_to_clean['Solar'].fillna(df_to_clean['Solar'].median())
    df_to_clean['Biomass'] = df_to_clean['Biomass'].fillna(df_to_clean['Biomass'].median())
    return df_to_clean


def calculate_net_balance(df):
    df['Net Balance'] = df['Production'] - df['Consumption']
    return df


def calculate_highest_deficit(clean_df):
    deficit_time = clean_df['Net Balance'].idxmin()
    highest_deficit = clean_df['Net Balance'].min()
    print('The moment with the highest deficit: ', deficit_time, ' with a deficit of: ', highest_deficit)


def calculate_highest_surplus(clean_df):
    surplus_time = clean_df['Net Balance'].idxmax()
    highest_surplus = clean_df['Net Balance'].max()
    print('The moment with the highest surplus: ', surplus_time, ' with a surplus of: ', highest_surplus)


def fossil_foot(clean_df):
    clean_df['Total Fossil'] = clean_df['Coal'] + clean_df['Oil and Gas']
    return clean_df


def green_foot(clean_df):
    clean_df['Total Green'] = clean_df['Wind'] + clean_df['Hydroelectric'] + clean_df['Solar'] + clean_df['Biomass']
    clean_df['Green Percentage'] = clean_df['Total Green'] / clean_df['Production'] * 100
    return clean_df

def seasonal_variation(clean_df):
    clean_df.reset_index(inplace=True)
    clean_df['Month'] = clean_df['DateTime'].dt.month
    monthly_stats = clean_df.groupby('Month')[['Consumption', 'Green Percentage']].mean()
    return monthly_stats

def main():
    df = read_dataframe()
    clean_df = clear_data(df)
    clean_df = calculate_net_balance(clean_df)
    calculate_highest_deficit(clean_df)
    calculate_highest_surplus(clean_df)
    clean_df = fossil_foot(clean_df)
    clean_df = green_foot(clean_df)
    monthly_green_foot = seasonal_variation(clean_df)
    print(monthly_green_foot)

if __name__ == '__main__':
    main()
