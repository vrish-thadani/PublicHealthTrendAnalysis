<img width="1199" height="628" alt="Screenshot 2026-10-05 at 2 21 52 AM" src="https://github.com/user-attachments/assets/797146ff-9ab3-4016-93dc-01d5607715d1" />
<img width="1197" height="752" alt="Screenshot 2026-10-05 at 2 22 00 AM" src="https://github.com/user-attachments/assets/bc086fa0-3bdf-4fc1-93a9-65732e97bdf6" />
<img width="1190" height="759" alt="Screenshot 2026-10-05 at 2 22 16 AM" src="https://github.com/user-attachments/assets/ea74fce1-4688-4700-a1be-f2dc1a4e3837" />
<img width="1195" height="752" alt="Screenshot 2026-10-05 at 2 22 29 AM" src="https://github.com/user-attachments/assets/b60d5d38-30e9-45fb-aef5-c926a1177a92" />
<img width="1178" height="726" alt="Screenshot 2026-10-05 at 2 22 37 AM" src="https://github.com/user-attachments/assets/ea4203be-9f2b-4e76-9fe1-e407951a113c" />
# 🌍 Public Health Trend Analysis & Life Expectancy Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline that analyzes historical public-health data from the **World Health Organization (WHO)**. The goal of this project is to identify meaningful socio-economic, demographic, and health-related patterns, and to deploy a robust regression model capable of forecasting life expectancy based on these global indicators.

## 🎯 Problem Statement
> *"A public-health organization wants to analyze historical observations and identify meaningful patterns useful for planning."*

To solve this, we formulated the problem as a supervised Machine Learning regression task. By understanding which features (e.g., Schooling, Adult Mortality, GDP, Immunization rates) most strongly correlate with life expectancy, organizations can allocate resources more effectively.

---

## 📁 Repository Structure

```text
Public_Health_ML_Project/
│
├── app.py                  # 🚀 Main Streamlit application
├── requirements.txt        # 📦 Project dependencies
├── README.md               # 📖 Project documentation
│
├── data/
│   └── Life Expectancy Data.csv   # WHO Dataset (193 Countries, 2000-2015)
│
├── notebooks/
│   └── public_health_analysis.ipynb # Jupyter notebook with end-to-end EDA and ML workflow
│
├── src/                    # ⚙️ Source Code Modules
│   ├── data_preprocessing.py   # Data cleaning, imputation, scaling, and encoding
│   ├── eda.py                  # Exploratory Data Analysis and visualization generation
│   ├── train_models.py         # Pipeline for training and selecting ML models
│   └── evaluation.py           # Evaluation metrics computation and residual plots
│
├── models/
│   └── best_model.pkl          # Serialized (joblib) Random Forest Regressor pipeline
│
└── outputs/                # 📊 Auto-generated results
    ├── model_results.csv       # Comparison of Linear Regression, Decision Tree, Random Forest
    └── figures/                # Visualizations (Heatmaps, Distributions, Residuals)
```

---

## 🔬 Machine Learning Methodology

Our approach follows industry best practices to prevent data leakage and ensure model generalization.

### 1. Data Cleaning & Preprocessing
*   **Missing Values**: Utilized `SimpleImputer` within an `sklearn.pipeline.Pipeline`. We imputed numeric features using the `median` (highly robust against outliers like GDP and population) and categorical features using the `most_frequent` strategy.
*   **Feature Scaling**: Standardized all continuous features using `StandardScaler` to ensure algorithms like Linear Regression converge efficiently and don't bias heavily scaled features.
*   **Categorical Encoding**: Handled the high-cardinality `Country` feature by dropping it to prevent overfitting, and encoded the `Status` (Developed vs Developing) using `OneHotEncoder`.

### 2. Model Architecture
Three distinct algorithms were evaluated:
1.  **Linear Regression**: Used as a baseline model.
2.  **Decision Tree Regressor**: Captures non-linear relationships.
3.  **Random Forest Regressor**: An ensemble approach utilized to minimize variance, combat overfitting, and yield the highest predictive accuracy.

### 3. Evaluation Metrics
Models were scored on a rigorous 80/20 train-test split using:
*   **R² (Coefficient of Determination)**: To understand the variance explained by the model.
*   **RMSE (Root Mean Squared Error)**: To heavily penalize large prediction errors (measured in years).
*   **MAE (Mean Absolute Error)**: To understand the average absolute drift in predictions.

---

## 📊 Key Insights & Results

The **Random Forest Regressor** outperformed all baseline models, achieving an **R² score of ~0.96** and an **RMSE of ~1.6 years**.

**Key Discoveries:**
1.  **Developed vs Developing Disparity**: Developed countries exhibit significantly higher life expectancy with much lower variance.
2.  **The Power of Education**: Features such as *Schooling* and *Income composition of resources* showed an incredibly high positive correlation with life expectancy. Access to education consistently predicts better health outcomes.
3.  **Critical Health Factors**: High *Adult Mortality* rates and *HIV/AIDS* prevalence heavily diminish expected lifespans across the board.

---

## 💻 Installation & Setup

If you wish to run this project locally, follow the steps below.

### Prerequisites
*   Python 3.8+ installed on your system.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Public-Health-Life-Expectancy.git
cd Public-Health-Life-Expectancy/Public_Health_ML_Project
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Running the Data Pipeline
If you want to train the models from scratch and generate fresh visualizations:
```bash
cd src
python eda.py
python train_models.py
python evaluation.py
cd ..
```
*The resulting graphs and metrics will populate the `outputs/` folder, and the final model will be saved to `models/best_model.pkl`.*

### 2. Launching the Interactive Dashboard
To explore the dataset and make live predictions using our modern, custom-styled UI:
```bash
streamlit run app.py
```
This will launch a local server (usually at `http://localhost:8501`) where you can interact with the dataset, view evaluation benchmarks, and input custom health parameters to predict life expectancy.

### 3. Exploring the Jupyter Notebook
For a step-by-step walkthrough of the data science lifecycle:
```bash
jupyter notebook notebooks/public_health_analysis.ipynb
```

---

## 🔮 Future Scope
*   **Time-Series Forecasting**: Shifting from standard regression to time-series models (ARIMA, Prophet) to forecast a specific country's trajectory over the next decade.
*   **Expanded Data**: Incorporating more recent WHO data (post-2015) to account for modern global health events.
*   **Deep Learning**: Experimenting with Neural Networks for potentially higher accuracy on complex, high-dimensional datasets.

---
*Created as part of an advanced machine learning analysis project.*
