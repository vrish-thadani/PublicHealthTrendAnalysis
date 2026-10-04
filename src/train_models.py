import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from data_preprocessing import load_and_clean_data, preprocess_data

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    results = []
    trained_pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2})
        trained_pipelines[name] = pipeline
        
    results_df = pd.DataFrame(results)
    return results_df, trained_pipelines

def get_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out()
    num_features = feature_names
    
    all_features = np.concatenate([num_features, cat_features])
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        return importance_df.sort_values(by='Importance', ascending=False)
    return None

if __name__ == "__main__":
    df = load_and_clean_data('../data/Life Expectancy Data.csv')
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_df, trained_pipelines = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    print("Model Comparison:")
    print(results_df)
    
    os.makedirs('../outputs', exist_ok=True)
    results_df.to_csv('../outputs/model_results.csv', index=False)
    
    best_model_name = results_df.loc[results_df['R²'].idxmax()]['Model']
    print(f"Best model based on R²: {best_model_name}")
    
    best_pipeline = trained_pipelines[best_model_name]
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_pipeline, '../models/best_model.pkl')
    print("Best model saved.")
    
    # Feature importance
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    importance_df = get_feature_importance(best_pipeline, num_features)
    if importance_df is not None:
        importance_df.to_csv('../outputs/feature_importance.csv', index=False)
        print("Feature importance saved.")
