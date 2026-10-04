import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Public Health Trend Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling (High Contrast & Enterprise Ready) ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, li, span {
        color: #334155;
    }
    
    /* Sidebar styling without breaking dropdowns */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] .css-17lntkn {
        color: #0f172a;
    }
    
    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        font-weight: 600;
        border-radius: 0.375rem;
        border: none;
        padding: 0.625rem 1.25rem;
        transition: background-color 0.15s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }
    
    /* Information Cards */
    .corporate-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .corporate-card h4 {
        margin-top: 0;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }
    .corporate-card p {
        margin: 0;
        color: #475569;
        line-height: 1.6;
    }
    
    /* Decorative Left Borders for emphasis */
    .border-blue { border-left: 4px solid #3b82f6; }
    .border-emerald { border-left: 4px solid #10b981; }
    .border-indigo { border-left: 4px solid #6366f1; }
    
    /* Prediction Banner */
    .prediction-banner {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #22c55e;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
    }
    .prediction-banner h2 {
        color: #166534 !important;
        margin: 0 0 0.5rem 0;
        font-size: 2.25rem;
    }
    .prediction-banner p {
        color: #15803d;
        margin: 0;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    /* Divider */
    hr {
        border-top: 1px solid #e2e8f0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/Life Expectancy Data.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Model Loading ---
@st.cache_resource
def load_model():
    if os.path.exists('models/best_model.pkl'):
        return joblib.load('models/best_model.pkl')
    return None

model = load_model()

# --- Sidebar Navigation ---
st.sidebar.markdown("""
<div style='margin-bottom: 2rem; padding: 0.5rem 0;'>
    <h2 style='color: #0f172a; font-weight: 800; margin-bottom: 0.2rem; font-size: 1.4rem;'>Public Health Analytics</h2>
    <p style='color: #64748b; font-size: 0.85rem; margin-top: 0; font-weight: 500;'>Life Expectancy Prediction Engine</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "", 
    ["Dashboard Overview", "Dataset Explorer", "Exploratory Analysis", "Predictive Model", "Model Evaluation", "Analytical Insights"]
)

st.sidebar.markdown("---")

# --- Page Content ---
if page == "Dashboard Overview":
    st.markdown("<h1 style='margin-bottom: 0.5rem;'>Global Health Intelligence Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #475569; margin-bottom: 2.5rem;'>Executive summary of global life expectancy and critical health metrics.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Countries", df['Country'].nunique())
    col2.metric("Data Year Range", f"{df['Year'].min()} - {df['Year'].max()}")
    col3.metric("Total Records", f"{len(df):,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    col4.metric("Average Life Expectancy", f"{df['Life expectancy'].mean():.1f} Years")
    
    if os.path.exists('outputs/model_results.csv'):
        results = pd.read_csv('outputs/model_results.csv')
        best_r2 = results['R²'].max()
        best_model_name = results.loc[results['R²'].idxmax()]['Model']
        col5.metric(f"Primary Prediction Engine", f"{best_model_name} ({best_r2 * 100:.1f}% R²)")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class='corporate-card border-blue'>
        <h4>Project Directives</h4>
        <ul style='color: #475569; margin-top: 0.5rem; padding-left: 1.5rem;'>
            <li style='margin-bottom: 0.5rem;'><b>Analyze</b> historical public health datasets covering 193 nations.</li>
            <li style='margin-bottom: 0.5rem;'><b>Identify</b> key socioeconomic and demographic risk factors influencing life expectancy.</li>
            <li><b>Deploy</b> a highly accurate supervised machine learning model capable of forecasting future health outcomes based on indicator adjustments.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.markdown("<p style='color: #475569; font-size: 1.1rem;'>Filter and inspect the foundational World Health Organization dataset.</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<h4 style='color: #1e293b; margin-bottom: 1rem;'>Data Filters</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        countries = col1.multiselect("Select Countries", options=sorted(df['Country'].unique()))
        status = col2.multiselect("Select Status (Developed/Developing)", options=df['Status'].unique())
        
        filtered_df = df.copy()
        if countries:
            filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
        if status:
            filtered_df = filtered_df[filtered_df['Status'].isin(status)]
            
        st.dataframe(filtered_df, use_container_width=True, height=400)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #1e293b;'>Summary Statistics</h4>", unsafe_allow_html=True)
        st.dataframe(filtered_df.describe(), use_container_width=True)
    with col2:
        st.markdown("<h4 style='color: #1e293b;'>Missing Values Profile</h4>", unsafe_allow_html=True)
        st.dataframe(filtered_df.isna().sum().rename("Missing Count"), use_container_width=True)

elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("<p style='color: #475569; font-size: 1.1rem;'>Visual interpretation of macro-trends, distributions, and feature correlations.</p>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Distributions & Trends", "Socioeconomic Factors", "Correlation Matrix"])
    
    with tab1:
        st.image('outputs/figures/life_expectancy_distribution.png', use_column_width=True)
        st.image('outputs/figures/life_expectancy_trend.png', use_column_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.image('outputs/figures/life_expectancy_status.png', use_column_width=True)
        with col2:
            st.image('outputs/figures/life_expectancy_vs_gdp.png', use_column_width=True)
        st.image('outputs/figures/life_expectancy_vs_schooling.png', use_column_width=True)
        
    with tab3:
        st.image('outputs/figures/correlation_heatmap.png', use_column_width=True)

elif page == "Predictive Model":
    st.title("Predictive Intelligence Model")
    st.markdown("<p style='color: #475569; font-size: 1.1rem;'>Utilize the serialized Machine Learning pipeline to forecast life expectancy based on user-defined inputs.</p>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not found. Please train the model pipeline first.")
    else:
        with st.form("prediction_form"):
            st.markdown("<h4 style='color: #1e293b; margin-top: 0;'>Health & Mortality Indicators</h4>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            year = col1.number_input("Year", min_value=2000, max_value=2025, value=2015)
            status = col1.selectbox("Country Status", options=["Developed", "Developing"])
            adult_mortality = col1.number_input("Adult Mortality (per 1000)", value=float(df['Adult Mortality'].median()))
            infant_deaths = col2.number_input("Infant Deaths (per 1000)", value=float(df['infant deaths'].median()))
            alcohol = col2.number_input("Alcohol Consumption (Liters)", value=float(df['Alcohol'].median()))
            percentage_expenditure = col2.number_input("Health Exp. (% of GDP)", value=float(df['percentage expenditure'].median()))
            
            hepatitis_b = col3.number_input("Hepatitis B Coverage (%)", value=float(df['Hepatitis B'].median()))
            measles = col3.number_input("Measles Cases (per 1000)", value=float(df['Measles'].median()))
            bmi = col3.number_input("Average BMI", value=float(df['BMI'].median()))
            
            st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #1e293b; margin-top: 0;'>Economic & Demographic Indicators</h4>", unsafe_allow_html=True)
            col4, col5, col6 = st.columns(3)
            
            under_five_deaths = col4.number_input("Under-five Deaths", value=float(df['under-five deaths'].median()))
            polio = col4.number_input("Polio Coverage (%)", value=float(df['Polio'].median()))
            total_expenditure = col4.number_input("Total Expenditure (%)", value=float(df['Total expenditure'].median()))
            
            diphtheria = col5.number_input("Diphtheria Coverage (%)", value=float(df['Diphtheria'].median()))
            hiv_aids = col5.number_input("HIV/AIDS Deaths (per 1000)", value=float(df['HIV/AIDS'].median()))
            gdp = col5.number_input("GDP (USD)", value=float(df['GDP'].median()))
            
            population = col6.number_input("Population", value=float(df['Population'].median()))
            thinness_1_19 = col6.number_input("Thinness 1-19 years (%)", value=float(df['thinness  1-19 years'].median()))
            thinness_5_9 = col6.number_input("Thinness 5-9 years (%)", value=float(df['thinness 5-9 years'].median()))
            
            st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #1e293b; margin-top: 0;'>Social Indicators</h4>", unsafe_allow_html=True)
            col7, col8 = st.columns(2)
            income_comp = col7.number_input("Income Comp. of Resources", value=float(df['Income composition of resources'].median()))
            schooling = col8.number_input("Schooling (Years)", value=float(df['Schooling'].median()))
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="Execute Prediction Engine")
            
            if submit_button:
                input_data = pd.DataFrame({
                    'Year': [year],
                    'Status': [status],
                    'Adult Mortality': [adult_mortality],
                    'infant deaths': [infant_deaths],
                    'Alcohol': [alcohol],
                    'percentage expenditure': [percentage_expenditure],
                    'Hepatitis B': [hepatitis_b],
                    'Measles': [measles],
                    'BMI': [bmi],
                    'under-five deaths': [under_five_deaths],
                    'Polio': [polio],
                    'Total expenditure': [total_expenditure],
                    'Diphtheria': [diphtheria],
                    'HIV/AIDS': [hiv_aids],
                    'GDP': [gdp],
                    'Population': [population],
                    'thinness  1-19 years': [thinness_1_19],
                    'thinness 5-9 years': [thinness_5_9],
                    'Income composition of resources': [income_comp],
                    'Schooling': [schooling]
                })
                
                prediction = model.predict(input_data)[0]
                
                st.markdown(f"""
                <div class='prediction-banner'>
                    <h2>{prediction:.1f} Years</h2>
                    <p>Estimated Life Expectancy Based on Current Parameters</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "Model Evaluation":
    st.title("Model Evaluation & Benchmarks")
    st.markdown("<p style='color: #475569; font-size: 1.1rem;'>Comparing algorithm performance across standard regression metrics.</p>", unsafe_allow_html=True)
    
    if os.path.exists('outputs/model_results.csv'):
        results = pd.read_csv('outputs/model_results.csv')
        
        st.markdown("<h4 style='color: #1e293b;'>Algorithm Comparison Matrix</h4>", unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True)
        
        best_row = results.loc[results['R²'].idxmax()]
        
        st.markdown(f"""
        <div class='corporate-card border-blue' style='margin-top: 2rem;'>
            <h3 style='margin-top: 0; color: #0f172a;'>Primary Selected Model: {best_row['Model']}</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.5rem;'>
                <div style='background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #e2e8f0;'>
                    <h2 style='color: #0f172a; margin: 0;'>{best_row['R²']:.4f}</h2>
                    <p style='margin: 0; color: #64748b; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;'>R² Score</p>
                </div>
                <div style='background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #e2e8f0;'>
                    <h2 style='color: #0f172a; margin: 0;'>{best_row['RMSE']:.4f}</h2>
                    <p style='margin: 0; color: #64748b; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;'>RMSE (Years)</p>
                </div>
                <div style='background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #e2e8f0;'>
                    <h2 style='color: #0f172a; margin: 0;'>{best_row['MAE']:.4f}</h2>
                    <p style='margin: 0; color: #64748b; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;'>MAE (Years)</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #1e293b;'>Error Analysis Visualizations</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image('outputs/figures/actual_vs_predicted.png', caption="Actual vs Predicted Spread", use_column_width=True)
        with col2:
            st.image('outputs/figures/residual_distribution.png', caption="Residual (Error) Distribution", use_column_width=True)
    else:
        st.error("No evaluation results found.")

elif page == "Analytical Insights":
    st.title("Key Analytical Insights")
    st.markdown("<p style='color: #475569; font-size: 1.1rem;'>Derived conclusions based on rigorous exploratory analysis and ML feature importance.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='display: flex; flex-direction: column; gap: 1rem;'>
        <div class='corporate-card border-indigo'>
            <h4>Developed vs Developing Disparity</h4>
            <p>Developed countries exhibit significantly higher life expectancy with lower variance compared to developing nations. Socioeconomic stability clearly acts as a primary foundational determinant for public health outcomes.</p>
        </div>
        
        <div class='corporate-card border-emerald'>
            <h4>The Impact of Education</h4>
            <p>Features such as <b>Schooling</b> and <b>Income composition of resources</b> showed a dominant positive correlation with life expectancy. Access to education consistently predicts better health outcomes globally.</p>
        </div>
        
        <div class='corporate-card border-blue'>
            <h4>Critical Health Factors</h4>
            <p>Metrics like <b>Adult Mortality</b> and <b>HIV/AIDS</b> prevalence heavily diminish expected lifespans. Regions with unmanaged communicable diseases suffer from drastically lower life expectancies despite economic improvements.</p>
        </div>
        
        <div class='corporate-card border-indigo'>
            <h4>Model Robustness</h4>
            <p>Our ensemble models (such as the Random Forest Regressor) successfully captured non-linear relationships in the highly-dimensional health data, predicting life expectancy with high precision (R² > 0.95), proving the viability of ML in public health planning.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
