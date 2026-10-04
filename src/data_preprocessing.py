import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Clean whitespace in column names
    df.columns = df.columns.str.strip()
    return df

def perform_eda(df, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Life expectancy distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Life expectancy'], bins=30, kde=True)
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
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

def preprocess_data(df):
    # Drop rows where target is missing
    df = df.dropna(subset=['Life expectancy'])
    
    X = df.drop(['Life expectancy', 'Country'], axis=1) # Exclude Country as it's too high cardinality and target
    y = df['Life expectancy']
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

if __name__ == "__main__":
    df = load_and_clean_data('../data/Life Expectancy Data.csv')
    perform_eda(df, '../outputs/figures')
    X, y, preprocessor = preprocess_data(df)
    print("Preprocessing completed.")
