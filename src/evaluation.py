import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import load_and_clean_data, preprocess_data
from sklearn.model_selection import train_test_split

def evaluate_model(model, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Evaluation Metrics on Test Set:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Life Expectancy')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # Residual Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'))
    plt.close()

if __name__ == "__main__":
    df = load_and_clean_data('../data/Life Expectancy Data.csv')
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_path = '../models/best_model.pkl'
    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
        evaluate_model(best_model, X_test, y_test, '../outputs/figures')
        print("Evaluation completed and figures saved.")
    else:
        print("Model not found. Run train_models.py first.")
