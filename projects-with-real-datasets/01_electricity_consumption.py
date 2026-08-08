"""
NATIONAL GRID POWER ANALYTICS ENGINE

- energy independence analysis to see when Romania needed to import/export energy

- green foot vs fossil to see the energetic transition

"""
import os

import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def read_dataframe():
    df = pd.read_csv('../datasets/electricityConsumptionAndProductioction.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # df.set_index('DateTime', inplace=True)
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


def correlations_analysis(df):
    columns_of_interest = ['Consumption', 'Total Fossil', 'Total Green', 'Coal', 'Wind']
    correlation_matrix = df[columns_of_interest].corr()

    return correlation_matrix


def export_results(clean_df, monthly_green_foot, correlation_matrix):
    if not os.path.exists("../exports"):
        os.makedirs("../exports")
    clean_df.to_csv('../exports/energy_cleaned_data.csv', index=False)
    monthly_green_foot.to_csv('../exports/monthly_green_foot.csv')
    correlation_matrix.to_csv('../exports/correlation_matrix.csv')
    print('Files have been exported successfully\n')


def hourly_energy_profile(ax, df):
    hourly_mean = df.groupby(df['DateTime'].dt.hour)[['Consumption', 'Production', 'Net Balance']].mean()

    ax.plot(hourly_mean.index,
            hourly_mean['Consumption'],
            color='red',
            label='Consumption')

    ax.plot(hourly_mean.index,
            hourly_mean['Production'],
            color='blue',
            label='Production')

    ax2 = ax.twinx()

    ax2.plot(hourly_mean.index,
             hourly_mean['Net Balance'],
             color='green',
             linestyle='--',
             label='Net Balance')

    peak_hour = hourly_mean['Consumption'].idxmax()
    peak_value = hourly_mean['Consumption'].max()

    ax.annotate(f'Peak hour',
                xy=(peak_hour, peak_value),
                xytext=(peak_hour + 0.5, peak_value + 0.5),
                arrowprops=dict(facecolor='red', width=2, headwidth=5),
                fontsize=10)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(lines1 + lines2, labels1 + labels2)

    ax.grid(alpha=0.3)
    ax.set_title("Hourly Energy Profile & Net Balance")


def monthly_energy_generation_mix(ax, df):
    monthly_mean = df.groupby(df['Month'])[['Total Green', 'Total Fossil', 'Nuclear']].mean()

    months = monthly_mean.index

    green = monthly_mean['Total Green']
    nuclear = monthly_mean['Nuclear']
    fossil = monthly_mean['Total Fossil']

    ax.bar(
        months,
        green,
        color='green',
        label='Total Green'
    )

    ax.bar(
        months,
        nuclear,
        bottom=green,
        color='orange',
        label='Total Nuclear'
    )

    ax.bar(
        months,
        fossil,
        bottom=green + nuclear,
        color='black',
        label='Total Fossil'
    )

    ax.set_xticks(months)

    ax.set_xticklabels([
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec'
    ])

    ax.set_title('Monthly Energy Generation Mix (MW)')
    ax.legend()


def green_energy_percentage_distribution(ax, df):
    box_data = []
    for month in range(1, 13):
        values = df[df['Month'] == month]['Green Percentage']
        box_data.append(values)

    bp = ax.boxplot(box_data, patch_artist=True)

    for box in bp['boxes']:
        box.set_facecolor('green')

    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)

    ax.set_xticks(range(1, 13))

    ax.set_xticklabels([
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec'
    ])

    ax.set_ylabel('Green Energy (%)')
    ax.set_title('Green Energy Percentage Distribution by Month')


def thermal_dep_total_demand(ax, df):
    consumption = df['Consumption']
    fossil = df['Total Fossil']
    green_percentage = df['Green Percentage']
    scatter = ax.scatter(consumption, fossil, c=green_percentage, s=5, cmap='RdYlGn', alpha=0.6,
                         edgecolors='black', linewidths=1)
    plt.colorbar(scatter, ax=ax)
    ax.set_xlabel("Consumption")
    ax.set_ylabel("Total Fossil")
    ax.set_title("Thermal Dependency vs. Total Demand")


def save_chart(fig):
    fig.savefig('../exports/energy_analytics_dashboard.png', dpi=300)


# 2x2 subplots saved as energy_analysis_dashboard.png in /exports (300 DPI)
def generate_visual_dashboard(clean_df, monthly_green_foot, correlation_matrix):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    hourly_energy_profile(axes[0, 0], clean_df)
    monthly_energy_generation_mix(axes[0, 1], clean_df)
    green_energy_percentage_distribution(axes[1, 0], clean_df)
    thermal_dep_total_demand(axes[1, 1], clean_df)

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('NATIONAL GRID POWER ANALYSIS ENGINE - EXECUTIVE DASHBOARD', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_chart(fig)

    plt.show()


def train_energy_model(clean_df):
    df_ml = clean_df.copy()

    df_ml['Hour'] = df_ml['DateTime'].dt.hour
    df_ml['Day'] = df_ml['DateTime'].dt.dayofweek
    df_ml['Month'] = df_ml['DateTime'].dt.month

    #define features and target
    feature_cols = ['Total Green', 'Nuclear', 'Total Fossil', 'Hour', 'Day', 'Month']
    X = df_ml[feature_cols]
    y = df_ml['Consumption']

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    #model training
    regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    regressor.fit(X_train, y_train)

    #prediction on test data
    y_pred = regressor.predict(X_test)

    #evaluation
    mae = mean_absolute_error(y_test, y_pred)
    print(f'\n --- ML MODEL RESULTS --- ')
    print(f'MAE: {mae:.2f} MW')

    return regressor

def feature_importance(regression):
    print('\n --- FEATURE IMPORTANCE ---')
    importances = regression.feature_importances_
    feature_cols = ['Total Green', 'Nuclear', 'Total Fossil', 'Hour', 'Day', 'Month']
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"Impact {col}: {imp *100 :.2f}%")

def main():
    df = read_dataframe()
    clean_df = clear_data(df)
    clean_df = calculate_net_balance(clean_df)
    calculate_highest_deficit(clean_df)
    calculate_highest_surplus(clean_df)
    clean_df = fossil_foot(clean_df)
    clean_df = green_foot(clean_df)
    monthly_green_foot = seasonal_variation(clean_df)
    correlation_matrix = correlations_analysis(clean_df)
    export_results(clean_df, monthly_green_foot, correlation_matrix)
    generate_visual_dashboard(clean_df, monthly_green_foot, correlation_matrix)
    regression = train_energy_model(clean_df)
    feature_importance(regression)


if __name__ == '__main__':
    main()
