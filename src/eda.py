import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Life expectancy distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Life expectancy'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Life Expectancy')
    plt.xlabel('Life Expectancy (Years)')
    plt.savefig(os.path.join(output_dir, 'life_expectancy_distribution.png'))
    plt.close()
    
    # 2. Life expectancy trend over years
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Year', y='Life expectancy', estimator=np.mean)
    plt.title('Average Life Expectancy Trend (2000-2015)')
    plt.savefig(os.path.join(output_dir, 'life_expectancy_trend.png'))
    plt.close()
    
    # 3. Life expectancy vs Status
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Status', y='Life expectancy')
    plt.title('Life Expectancy: Developed vs Developing')
    plt.savefig(os.path.join(output_dir, 'life_expectancy_status.png'))
    plt.close()
    
    # 4. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 5. Life Expectancy vs GDP
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='GDP', y='Life expectancy', hue='Status', alpha=0.6)
    plt.title('Life Expectancy vs GDP')
    plt.savefig(os.path.join(output_dir, 'life_expectancy_vs_gdp.png'))
    plt.close()
    
    # 6. Life Expectancy vs Schooling
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Schooling', y='Life expectancy', hue='Status', alpha=0.6)
    plt.title('Life Expectancy vs Schooling')
    plt.savefig(os.path.join(output_dir, 'life_expectancy_vs_schooling.png'))
    plt.close()
    
if __name__ == "__main__":
    df = pd.read_csv('../data/Life Expectancy Data.csv')
    df.columns = df.columns.str.strip()
    perform_eda(df, '../outputs/figures')
    print("EDA completed and figures saved.")
