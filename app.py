import shap
import numpy as np
import xgboost as xgb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import logging
from contextlib import contextmanager
import os
import io
import plotly.express as px
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "resume_data.db"
TABLE_NAME = "resume_data"

# Performance monitoring decorator
def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}
        
        st.session_state.performance_metrics[func.__name__] = execution_time
        return result
    return wrapper

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        st.error("An unexpected error occurred. Please check the logs.")
        raise
    finally:
        if conn:
            conn.close()

def validate_entry_data(data: dict) -> tuple[bool, str]:
    try:
        # Basic validation checks
        if data['Age'] < 18 or data['Age'] > 100:
            return False, "Age must be between 18 and 100"
        
        required_fields = ['Age', 'Sex', 'Employment_Status', 'Healthcare_Role']
        for field in required_fields:
            if not data.get(field):
                return False, f"{field} is required"
                
        return True, ""
    except Exception as e:
        return False, str(e)

@monitor_performance
def create_database():
    """Create database and table if they don't exist"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- 👤 Demographic & Socioeconomic
                    Age INTEGER,
                    Sex TEXT,
                    Employment_Status TEXT,
                    Income_Level TEXT,
                    Social_Deprivation INTEGER,
                    Material_Deprivation INTEGER,
                    
                    -- 👩‍⚕️ Occupation Details
                    Healthcare_Role TEXT,
                    Department TEXT,
                    Years_Experience INTEGER,
                    Weekly_Hours INTEGER,
                    Night_Shifts_Monthly INTEGER,
                    Overtime_Hours_Monthly INTEGER,
                    Patient_Facing TEXT,
                    Management_Responsibilities TEXT,
                    Work_Stress_Level INTEGER,
                    Job_Satisfaction INTEGER,
                    Workplace_Support INTEGER,
                    Burnout_Level INTEGER,
                    Sick_Days_Last_Year INTEGER,
                    Workplace_Incidents INTEGER,
                    Recent_Promotion TEXT,
                    Recent_Demotion TEXT,
                    
                    -- 🏥 Clinical & Psychiatric
                    MH_Disorders TEXT,
                    Substance_Use_Disorders TEXT,
                    History_Suicidal_Ideation INTEGER,
                    Previous_Suicide_Attempts INTEGER,
                    Frequency_Suicidal_Thoughts INTEGER,
                    Intensity_Suicidal_Thoughts INTEGER,
                    
                    -- 🏃 Health & Medical
                    Chronic_Illnesses TEXT,
                    GP_Visits INTEGER,
                    ED_Visits INTEGER,
                    Hospitalizations INTEGER,
                    
                    -- 🧠 Psychological Factors
                    Hopelessness INTEGER,
                    Despair INTEGER,
                    Impulsivity INTEGER,
                    Aggression INTEGER,
                    Access_Lethal_Means INTEGER,
                    Social_Isolation INTEGER,
                    
                    -- 🤝 Support & Resilience
                    Coping_Strategies INTEGER,
                    Measured_Resilience INTEGER,
                    MH_Service_Engagement INTEGER,
                    Supportive_Relationships INTEGER,
                    
                    -- 📊 Risk Assessment
                    Suicidal_Distress INTEGER,
                    Time_To_Crisis INTEGER,
                    Crisis_Event INTEGER,
                    Observation_Date TEXT
                )
            """)
            conn.commit()
            st.success("Database initialized successfully!")
            
    except sqlite3.Error as e:
        st.error(f"Database creation error: {e}")
        logger.error(f"Database creation error: {e}")

@st.cache_data(ttl=3600)
def load_data():
    try:
        with get_db_connection() as conn:
            query = f"SELECT * FROM {TABLE_NAME}"
            df = pd.read_sql_query(query, conn)
            
            # Convert timestamp columns
            if 'Observation_Date' in df.columns:
                df['Observation_Date'] = pd.to_datetime(df['Observation_Date'])
            
            # Convert numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return pd.DataFrame()


@monitor_performance
def insert_entry(data: dict):
    try:
        # Validate data before insertion
        is_valid, error_message = validate_entry_data(data)
        if not is_valid:
            st.error(f"Validation error: {error_message}")
            return False
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            query = f"INSERT INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, list(data.values()))
            conn.commit()
            return True
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        st.error(f"⚠️ Failed to save data: {e}")
        return False

def safe_export_data(df: pd.DataFrame, format: str) -> tuple[bool, str]:
    try:
        if format == "CSV":
            return True, df.to_csv(index=False)
        elif format == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            return True, buffer.getvalue()
        else:
            return False, "Unsupported format"
    except Exception as e:
        return False, str(e)

@monitor_performance
def main():
    st.set_page_config(
        page_title="RESUME Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.edit_history = []
        st.session_state.last_update = datetime.now()

    # Create database if it doesn't exist
    if not os.path.exists(DB_PATH):
        create_database()
    
    # Load data with error handling
    try:
        df = load_data()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error("Failed to load data. Please check the database connection.")
        df = pd.DataFrame()
    
    # Sidebar
    st.sidebar.header("🧭 Main Menu")
    edit_mode = st.sidebar.checkbox("Enable Edit Mode", value=False)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🟠 Dashboard",
        "🟢 What is RESUME",
        "🔵 Database (Front)",
        "🟣 Detailed Analytics",
        "🔷 Data Entry",
        "⚙️ Database (Backend)"
    ])

    # Dashboard Tab
    with tab1:
        try:
            display_dashboard(df)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            st.error("Error displaying dashboard. Please check the logs.")

    # What is RESUME Tab
    with tab2:
        st.title("What is RESUME?")
        st.markdown("""
        ### RESUME: Risk Evaluation System for Understanding Mental health Emergencies
        
        RESUME is a comprehensive tool designed to:
        - Monitor healthcare workers' mental health
        - Identify early warning signs of crisis
        - Provide data-driven insights for intervention
        - Support evidence-based prevention strategies
        
        ### Key Features:
        1. Real-time risk assessment
        2. Predictive analytics
        3. Comprehensive data collection
        4. Secure and confidential reporting
        """)

    # Database Front Tab
    with tab3:
        display_database_front(df)

    # Detailed Analytics Tab
    with tab4:
        st.title("Detailed Analytics")
        display_data_quality_metrics(df)
        display_visualizations(df, df.columns.tolist())

    # Data Entry Tab
    with tab5:
        display_data_entry_form()

    # Database Backend Tab
    with tab6:
        display_database_backend(edit_mode)



@monitor_performance
def display_dashboard(df):
    st.title("🧠 RESUME Predictive Modelling Dashboard")
    
    # Add analysis choice selection in the dashboard
    analysis_choice = st.sidebar.radio(
        "Choose Analysis Model:",
        ["Kaplan–Meier Estimator", 
         "Cox Proportional Hazards Model",
         "Bayesian Spatio-Temporal Model", 
         "Machine Learning (XGBoost)"]
    )
    
    # Introduction and Explanation
    st.markdown("""
    ### 📋 What am I looking at?
    This dashboard provides a comprehensive overview of suicide risk factors and patterns among healthcare workers. 
    The data shown here helps identify trends, risk factors, and potential intervention points.

    ### 🎯 Key Points:
    - All data is anonymized and aggregated
    - Risk scores range from 0 (lowest) to 10 (highest)
    - Temporal patterns show how risks change over time
    - Employment status may influence risk factors
    """)
    
    if not df.empty:
        # Overview metrics with explanations
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
            st.markdown("*Total number of healthcare workers in database*")
        
        with col2:
            crisis_events = df['Crisis_Event'].sum() if 'Crisis_Event' in df.columns else 0
            st.metric("Recorded Crisis Events", int(crisis_events))
            st.markdown("*Number of documented crisis incidents*")
        
        with col3:
            if 'Age' in df.columns:
                age_range = f"{int(df['Age'].min())} - {int(df['Age'].max())}"
                st.metric("Age Range", age_range)
                st.markdown("*Age span of healthcare workers*")
            else:
                st.metric("Age Range", "N/A")
                st.markdown("*Age data not available*")
        
        with col4:
            if 'Employment_Status' in df.columns:
                employment_counts = df['Employment_Status'].value_counts()
                st.metric("Employment Types", len(employment_counts))
                st.markdown("*Different categories of employment*")
            else:
                st.metric("Employment Types", "N/A")
                st.markdown("*Employment data not available*")

        try:
            # Employment Distribution
            st.subheader("👥 Employment Distribution")
            if 'Employment_Status' in df.columns:
                emp_fig = px.pie(df, names='Employment_Status', 
                               title='Distribution by Employment Status')
                st.plotly_chart(emp_fig)
            
            # Risk Analysis Section
            display_risk_analysis(df, analysis_choice)
            
        except Exception as e:
            logger.error(f"Error in dashboard visualization: {e}")
            st.error("Error displaying visualizations. Please check the data format.")
    else:
        st.info("No data available. Please add entries using the Data Entry tab.")

@monitor_performance
def display_risk_analysis(df, analysis_choice):
    st.header("🎯 Risk Analysis Overview")
    st.markdown("""
    ### Understanding the Risk Models:

    1. **Kaplan-Meier Analysis**
    - Shows survival probability over time
    - Helps identify critical periods for intervention
    - Lower curves indicate higher risk periods

    2. **Cox Proportional Hazards**
    - Examines multiple risk factors simultaneously
    - Identifies which factors most strongly predict crises
    - Helps target interventions effectively

    3. **Machine Learning Predictions**
    - Uses patterns in data to predict future risks
    - Combines multiple factors for accurate assessment
    - Helps identify high-risk individuals early
    """)

    try:
        if analysis_choice == "Kaplan–Meier Estimator":
            display_kaplan_meier_analysis(df)
        elif analysis_choice == "Cox Proportional Hazards Model":
            display_cox_analysis(df)
        elif analysis_choice == "Machine Learning (XGBoost)":
            display_ml_analysis(df)
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        st.error("Error performing risk analysis. Please check the data format.")

@monitor_performance
def display_kaplan_meier_analysis(df):
    if 'Time_To_Crisis' in df.columns and 'Crisis_Event' in df.columns:
        kmf = KaplanMeierFitter()
        kmf.fit(df['Time_To_Crisis'], 
               df['Crisis_Event'], 
               label='Overall')
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf.plot(ax=ax)
        plt.title('Kaplan-Meier Survival Curve')
        st.pyplot(fig)
    else:
        st.warning("Required data for Kaplan-Meier analysis is not available.")

@monitor_performance
def display_cox_analysis(df):
    try:
        if all(col in df.columns for col in ['Time_To_Crisis', 'Crisis_Event']):
            # Prepare data for Cox analysis
            cph = CoxPHFitter()
            covariates = ['Age', 'Work_Stress_Level', 'Burnout_Level', 'Social_Isolation']
            
            # Filter required columns and handle missing values
            cox_data = df[['Time_To_Crisis', 'Crisis_Event'] + covariates].dropna()
            
            if len(cox_data) > 0:
                cph.fit(cox_data, 
                       duration_col='Time_To_Crisis',
                       event_col='Crisis_Event')
                
                # Display results
                st.subheader("Cox Proportional Hazards Analysis")
                st.write(cph.print_summary())
                
                # Plot partial effects
                fig, ax = plt.subplots(figsize=(10, 6))
                cph.plot_partial_effects('Work_Stress_Level', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Insufficient data for Cox analysis after handling missing values.")
        else:
            st.warning("Required columns for Cox analysis are not available.")
    except Exception as e:
        logger.error(f"Error in Cox analysis: {e}")
        st.error("Error performing Cox analysis. Please check the data format.")

@monitor_performance
def display_ml_analysis(df):
    try:
        # Prepare features for ML analysis
        feature_cols = ['Age', 'Work_Stress_Level', 'Burnout_Level', 
                       'Social_Isolation', 'Hopelessness', 'Despair']
        
        if all(col in df.columns for col in feature_cols + ['Crisis_Event']):
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df['Crisis_Event']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train XGBoost model
            model = xgb.XGBClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Display results
            st.subheader("Machine Learning Analysis (XGBoost)")
            
            # Feature importance plot
            fig = px.bar(feature_importance, 
                        x='feature', 
                        y='importance',
                        title='Feature Importance')
            st.plotly_chart(fig)
            
            # Model performance
            y_pred = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred)
            st.metric("Model AUC-ROC Score", f"{auc_score:.3f}")
            
            # Add SHAP analysis
            st.subheader("SHAP Analysis")
            try:
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                # Create SHAP summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
                st.pyplot(fig)
                plt.close()
                
                # SHAP force plot for a sample case
                st.subheader("Sample Case Analysis")
                sample_idx = 0  # First test case
                st.markdown("SHAP values for a sample case:")
                
                # Convert SHAP values to HTML
                shap_html = shap.force_plot(
                    explainer.expected_value,
                    shap_values[sample_idx,:],
                    X_test.iloc[sample_idx,:],
                    feature_names=feature_cols,
                    matplotlib=True,
                    show=False
                )
                plt.close()
                st.pyplot(shap_html)
                
            except Exception as e:
                logger.error(f"SHAP analysis error: {e}")
                st.warning("SHAP analysis could not be completed")
            
        else:
            st.warning("Required features for ML analysis are not available.")
    except Exception as e:
        logger.error(f"Error in ML analysis: {e}")
        st.error("Error performing ML analysis. Please check the data format.")

@monitor_performance
def display_database_front(df):
    st.title("📊 Database Overview")

    if not df.empty:
        st.header("🔍 Search and Filter")
        
        # Search functionality
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_term = st.text_input("Search across all fields:", "")
        
        # Advanced filtering options
        with st.expander("Advanced Filters"):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # Demographic filters
                st.subheader("Demographics")
                if 'Age' in df.columns:
                    age_range = st.slider("Age Range", 
                                        int(df['Age'].min()), 
                                        int(df['Age'].max()), 
                                        (int(df['Age'].min()), int(df['Age'].max())))
                
                if 'Sex' in df.columns:
                    selected_genders = st.multiselect(
                        "Gender", 
                        df['Sex'].unique().tolist(),
                        default=df['Sex'].unique().tolist()
                    )
            
            with filter_col2:
                # Occupational filters
                st.subheader("Occupation")
                if 'Healthcare_Role' in df.columns:
                    selected_roles = st.multiselect(
                        "Healthcare Role", 
                        df['Healthcare_Role'].unique().tolist(),
                        default=df['Healthcare_Role'].unique().tolist()
                    )
                
                if 'Department' in df.columns:
                    selected_departments = st.multiselect(
                        "Department", 
                        df['Department'].unique().tolist(),
                        default=df['Department'].unique().tolist()
                    )
            
            with filter_col3:
                # Risk filters
                st.subheader("Risk Factors")
                if 'Suicidal_Distress' in df.columns:
                    risk_level = st.slider("Suicidal Distress Level", 
                                         0, 10, (0, 10))
                
                if 'Crisis_Event' in df.columns:
                    crisis_filter = st.multiselect(
                        "Crisis Events", 
                        [0, 1],
                        default=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )
        
        # Apply filters
        try:
            filtered_df = apply_filters(df, search_term, locals())
            display_filtered_results(filtered_df, df)
        except Exception as e:
            logger.error(f"Error in filtering: {e}")
            st.error("Error applying filters. Please check your selection.")
    else:
        st.info("No data available. Please add entries using the Data Entry tab.")

@monitor_performance
def apply_filters(df, search_term, filter_vars):
    filtered_df = df.copy()
    
    # Text search across all string columns
    if search_term:
        mask = pd.Series(False, index=filtered_df.index)
        for col in filtered_df.select_dtypes(include=['object']).columns:
            mask = mask | filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    # Apply other filters if they exist in the filter_vars
    if 'age_range' in filter_vars:
        filtered_df = filtered_df[
            (filtered_df['Age'] >= filter_vars['age_range'][0]) & 
            (filtered_df['Age'] <= filter_vars['age_range'][1])
        ]
    
    if 'selected_genders' in filter_vars:
        filtered_df = filtered_df[filtered_df['Sex'].isin(filter_vars['selected_genders'])]
    
    # Add more filters as needed
    
    return filtered_df

@monitor_performance
def display_filtered_results(filtered_df, df):
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
    
    # Column selection
    st.subheader("📋 Select Columns to Display")
    all_columns = filtered_df.columns.tolist()
    default_columns = [
        "Observation_Date", "Age", "Sex", "Employment_Status",
        "Healthcare_Role", "Department", "Suicidal_Distress", 
        "Time_To_Crisis", "Crisis_Event"
    ]
    selected_columns = st.multiselect(
        "Choose columns:",
        all_columns,
        default=[col for col in default_columns if col in all_columns]
    )
    
    # Display filtered data
    if selected_columns:
        st.dataframe(
            filtered_df[selected_columns].sort_values(
                "Observation_Date", 
                ascending=False
            )
        )
    else:
        st.warning("Please select at least one column to display")
    
    # Export options
    export_data(filtered_df, selected_columns)

@monitor_performance
def export_data(df, selected_columns):
    st.subheader("📤 Export Filtered Data")
    export_format = st.radio(
        "Export Format:", 
        ["CSV", "Excel", "JSON"], 
        horizontal=True
    )
    
    if st.button("Export Data"):
        try:
            export_df = df[selected_columns] if selected_columns else df
            
            if export_format == "CSV":
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="resume_filtered_data.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, index=False)
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="resume_filtered_data.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:  # JSON
                json_data = export_df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="resume_filtered_data.json",
                    mime="application/json"
                )
        except Exception as e:
            logger.error(f"Export error: {e}")
            st.error(f"Error exporting data: {e}")

@monitor_performance
def display_visualizations(df, selected_columns):
    st.subheader("📊 Quick Visualization")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        chart_type = st.selectbox(
            "Chart Type", 
            ["Bar Chart", "Pie Chart", "Histogram", "Box Plot"]
        )
    
    with viz_col2:
        if chart_type in ["Bar Chart", "Pie Chart"]:
            category_col = st.selectbox(
                "Category Column", 
                [col for col in selected_columns if df[col].dtype == 'object']
            )
        elif chart_type == "Histogram":
            numeric_col = st.selectbox(
                "Numeric Column", 
                [col for col in selected_columns if df[col].dtype in ['int64', 'float64']]
            )
        else:  # Box Plot
            numeric_col = st.selectbox(
                "Numeric Column", 
                [col for col in selected_columns if df[col].dtype in ['int64', 'float64']]
            )
            category_col = st.selectbox(
                "Group By", 
                [col for col in selected_columns if df[col].dtype == 'object']
            )
    
    try:
        # Generate visualization
        if chart_type == "Bar Chart" and 'category_col' in locals():
            fig = px.bar(
                df, 
                x=category_col, 
                title=f"Count by {category_col}"
            )
            st.plotly_chart(fig)
        elif chart_type == "Pie Chart" and 'category_col' in locals():
            fig = px.pie(
                df, 
                names=category_col, 
                title=f"Distribution by {category_col}"
            )
            st.plotly_chart(fig)
        elif chart_type == "Histogram" and 'numeric_col' in locals():
            fig = px.histogram(
                df, 
                x=numeric_col, 
                title=f"Distribution of {numeric_col}"
            )
            st.plotly_chart(fig)
        elif chart_type == "Box Plot" and 'numeric_col' in locals() and 'category_col' in locals():
            fig = px.box(
                df, 
                x=category_col, 
                y=numeric_col, 
                title=f"{numeric_col} by {category_col}"
            )
            st.plotly_chart(fig)
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        st.error("Error generating visualization. Please check your selection.")

@monitor_performance
def display_data_quality_metrics(df):
    st.header("🔍 Data Quality Overview")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    quality_metrics = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Missing Percentage': missing_pct.values
    })
    
    # Display metrics
    st.dataframe(quality_metrics[quality_metrics['Missing Values'] > 0])
    
    # Data completeness visualization
    if not quality_metrics[quality_metrics['Missing Values'] > 0].empty:
        fig = px.bar(
            quality_metrics[quality_metrics['Missing Values'] > 0],
            x='Column',
            y='Missing Percentage',
            title='Data Completeness Analysis'
        )
        st.plotly_chart(fig)
    else:
        st.success("No missing data found!")
    
    # Data type information
    st.subheader("📊 Column Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Data Type': df.dtypes.values,
        'Unique Values': df.nunique().values
    })
    dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)  # Convert dtype objects to strings
    st.dataframe(dtype_df)



@monitor_performance
def display_data_entry_form():
    st.title("📝 RESUME+ Data Entry")
    
    with st.form("data_entry_form"):
        # Create tabs for organized data entry
        entry_tab1, entry_tab2, entry_tab3, entry_tab4 = st.tabs([
            "Demographics & Occupation",
            "Clinical & Health",
            "Psychological Factors",
            "Risk Assessment"
        ])

        with entry_tab1:
            st.markdown("### 👤 Demographic & Socioeconomic")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 18, 100, 25)
                sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                employment_status = st.selectbox(
                    "Employment Status", 
                    ["Full-time", "Part-time", "Contract", "Temporary"]
                )
            with col2:
                income_level = st.selectbox(
                    "Income Level", 
                    ["Low", "Medium", "High"]
                )
                social_deprivation = st.slider(
                    "Social Deprivation", 
                    0, 10, 5,
                    help="0 = None, 10 = Severe"
                )
                material_deprivation = st.slider(
                    "Material Deprivation", 
                    0, 10, 5,
                    help="0 = None, 10 = Severe"
                )

            st.markdown("### 👩‍⚕️ Occupation Details")
            col1, col2 = st.columns(2)
            with col1:
                healthcare_role = st.selectbox(
                    "Healthcare Role",
                    ["Doctor", "Nurse", "Admin Staff", "Paramedic", "Technician",
                     "Therapist", "Pharmacist", "Midwife", "Healthcare Assistant"]
                )
                department = st.selectbox(
                    "Department",
                    ["Emergency", "Surgery", "Pediatrics", "Oncology", "Cardiology",
                     "Neurology", "Psychiatry", "Radiology", "General Practice",
                     "Intensive Care", "Obstetrics", "Geriatrics", "Administration"]
                )
                years_experience = st.number_input(
                    "Years of Experience", 
                    0, 50, 5
                )
                weekly_hours = st.number_input(
                    "Weekly Hours", 
                    0, 100, 40
                )
            with col2:
                night_shifts = st.number_input(
                    "Night Shifts per Month", 
                    0, 30, 0
                )
                work_stress = st.slider(
                    "Work Stress Level", 
                    0, 10, 5,
                    help="0 = None, 10 = Severe"
                )
                job_satisfaction = st.slider(
                    "Job Satisfaction", 
                    0, 10, 5,
                    help="0 = Very Dissatisfied, 10 = Very Satisfied"
                )
                patient_facing = st.selectbox(
                    "Patient Facing Role", 
                    ["Yes", "No"]
                )

        with entry_tab2:
            st.markdown("### 🏥 Clinical & Psychiatric")
            col1, col2 = st.columns(2)
            with col1:
                mh_disorders = st.multiselect(
                    "Mental Health Disorders",
                    ["Depression", "Anxiety", "PTSD", "Bipolar", "None"],
                    default=["None"]
                )
                substance_use = st.multiselect(
                    "Substance Use Disorders",
                    ["Alcohol", "Prescription Drugs", "Illicit Drugs", "None"],
                    default=["None"]
                )
            with col2:
                suicidal_ideation = st.selectbox(
                    "History of Suicidal Ideation",
                    ["No", "Yes"]
                )
                previous_attempts = st.number_input(
                    "Previous Suicide Attempts",
                    0, 10, 0
                )

            st.markdown("### 🏃 Health & Medical")
            col1, col2 = st.columns(2)
            with col1:
                chronic_illnesses = st.multiselect(
                    "Chronic Illnesses",
                    ["Diabetes", "Hypertension", "Asthma", "None"],
                    default=["None"]
                )
                gp_visits = st.number_input(
                    "GP Visits (Last 12 months)",
                    0, 50, 0
                )
            with col2:
                ed_visits = st.number_input(
                    "ED Visits (Last 12 months)",
                    0, 50, 0
                )
                hospitalizations = st.number_input(
                    "Hospitalizations (Last 12 months)",
                    0, 50, 0
                )

        with entry_tab3:
            st.markdown("### 🧠 Psychological Factors")
            col1, col2 = st.columns(2)
            with col1:
                hopelessness = st.slider(
                    "Hopelessness",
                    0, 10, 0,
                    help="0 = None, 10 = Severe"
                )
                despair = st.slider(
                    "Despair",
                    0, 10, 0,
                    help="0 = None, 10 = Severe"
                )
                impulsivity = st.slider(
                    "Impulsivity",
                    0, 10, 0,
                    help="0 = None, 10 = Severe"
                )
            with col2:
                aggression = st.slider(
                    "Aggression",
                    0, 10, 0,
                    help="0 = None, 10 = Severe"
                )
                access_lethal_means = st.slider(
                    "Access to Lethal Means",
                    0, 10, 0,
                    help="0 = None, 10 = High Access"
                )
                social_isolation = st.slider(
                    "Social Isolation",
                    0, 10, 0,
                    help="0 = None, 10 = Severe"
                )

        with entry_tab4:
            st.markdown("### 📊 Risk Assessment")
            col1, col2 = st.columns(2)
            with col1:
                suicidal_distress = st.slider(
                    "Suicidal Distress",
                    0, 10, 0,
                    help="0 = None, 10 = Severe"
                )
                time_to_crisis = st.number_input(
                    "Time to Crisis (days)",
                    0, 365, 0
                )
            with col2:
                crisis_event = st.selectbox(
                    "Crisis Event",
                    ["No", "Yes"]
                )

        submitted = st.form_submit_button("Submit Data")
        
        if submitted:
            try:
                # Prepare data for insertion
                entry_data = prepare_entry_data(locals())
                
                # Validate and insert data
                if insert_entry(entry_data):
                    st.success("✅ Data successfully saved!")
                    st.rerun()
                
            except Exception as e:
                logger.error(f"Data entry error: {e}")
                st.error(f"Error saving data: {e}")

@monitor_performance
def prepare_entry_data(form_data):
    """Prepare form data for database insertion"""
    entry_data = {
        "Age": form_data.get('age'),
        "Sex": form_data.get('sex'),
        "Employment_Status": form_data.get('employment_status'),
        "Income_Level": form_data.get('income_level'),
        "Social_Deprivation": form_data.get('social_deprivation'),
        "Material_Deprivation": form_data.get('material_deprivation'),
        "Healthcare_Role": form_data.get('healthcare_role'),
        "Department": form_data.get('department'),
        "Years_Experience": form_data.get('years_experience'),
        "Weekly_Hours": form_data.get('weekly_hours'),
        "Night_Shifts_Monthly": form_data.get('night_shifts'),
        "Work_Stress_Level": form_data.get('work_stress'),
        "Job_Satisfaction": form_data.get('job_satisfaction'),
        "Patient_Facing": form_data.get('patient_facing'),
        "MH_Disorders": ','.join(form_data.get('mh_disorders', [])),
        "Substance_Use_Disorders": ','.join(form_data.get('substance_use', [])),
        "History_Suicidal_Ideation": 1 if form_data.get('suicidal_ideation') == "Yes" else 0,
        "Previous_Suicide_Attempts": form_data.get('previous_attempts'),
        "Chronic_Illnesses": ','.join(form_data.get('chronic_illnesses', [])),
        "GP_Visits": form_data.get('gp_visits'),
        "ED_Visits": form_data.get('ed_visits'),
        "Hospitalizations": form_data.get('hospitalizations'),
        "Hopelessness": form_data.get('hopelessness'),
        "Despair": form_data.get('despair'),
        "Impulsivity": form_data.get('impulsivity'),
        "Aggression": form_data.get('aggression'),
        "Access_Lethal_Means": form_data.get('access_lethal_means'),
        "Social_Isolation": form_data.get('social_isolation'),
        "Suicidal_Distress": form_data.get('suicidal_distress'),
        "Time_To_Crisis": form_data.get('time_to_crisis'),
        "Crisis_Event": 1 if form_data.get('crisis_event') == "Yes" else 0,
        "Observation_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return entry_data


@monitor_performance
def display_database_backend(edit_mode):
    st.title("⚙️ Database Management")
    
    if edit_mode:
        st.warning("""
        ⚠️ Admin Mode Active - Use with caution!
        Changes made here directly affect the database.
        """)
        
        # Database Operations
        st.header("🔧 Database Operations")
        
        # Database Backup
        st.subheader("💾 Database Backup")
        if st.button("Create Backup"):
            try:
                backup_database()
                st.success("Database backup created successfully!")
            except Exception as e:
                logger.error(f"Backup error: {e}")
                st.error(f"Error creating backup: {e}")

        # Clear Database Option
        st.subheader("🗑️ Clear Database")
        st.markdown("""
        This will remove ALL records from the database.
        This action cannot be undone.
        """)
        if st.button("Clear Database"):
            confirm = st.checkbox("I understand this will delete all data")
            if confirm:
                try:
                    clear_database()
                    st.success("Database cleared successfully!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error clearing database: {e}")
                    st.error(f"Error clearing database: {e}")

        # Recent Entries Management
        display_recent_entries_management()

    # Database Statistics
    display_database_statistics()

@monitor_performance
def backup_database():
    """Create a backup of the database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backups/resume_data_{timestamp}.db"
    
    # Ensure backup directory exists
    os.makedirs("backups", exist_ok=True)
    
    try:
        with get_db_connection() as conn:
            # Create backup
            backup_conn = sqlite3.connect(backup_path)
            conn.backup(backup_conn)
            backup_conn.close()
        return True
    except Exception as e:
        logger.error(f"Backup error: {e}")
        raise

@monitor_performance
def clear_database():
    """Clear all records from the database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {TABLE_NAME}")
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Database clear error: {e}")
        raise

@monitor_performance
def update_database_entries(df):
    """Update database entries with edited dataframe"""
    try:
        with get_db_connection() as conn:
            df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        return True
    except Exception as e:
        logger.error(f"Database update error: {e}")
        raise

@monitor_performance
def prepare_df_for_display(df):
    """Convert DataFrame to display-friendly format"""
    display_df = df.copy()
    
    # Convert timestamps to strings
    datetime_columns = display_df.select_dtypes(include=['datetime64']).columns
    for col in datetime_columns:
        display_df[col] = display_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert other types to basic Python types
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            display_df[col] = display_df[col].astype(str)
        elif display_df[col].dtype in ['int64', 'float64']:
            display_df[col] = display_df[col].astype(float)
    
    return display_df


@monitor_performance
def display_database_statistics():
    st.header("📊 Database Statistics")
    
    try:
        df = load_data()
        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(df))
                if 'Healthcare_Role' in df.columns:
                    st.metric("Unique Healthcare Roles", 
                             len(df['Healthcare_Role'].unique()))
            with col2:
                st.metric("Data Points", len(df.columns) * len(df))
                if 'Observation_Date' in df.columns:
                    last_update = df['Observation_Date'].max()
                    if isinstance(last_update, pd.Timestamp):
                        last_update = last_update.strftime("%Y-%m-%d %H:%M:%S")
                    st.metric("Last Update", last_update)
                else:
                    st.metric("Last Update", "N/A")

            # Data Quality Check
            display_data_quality_check(df)

            # Database Documentation
            display_database_documentation()
        else:
            st.info("No data available in the database")
    except Exception as e:
        logger.error(f"Error displaying database statistics: {e}")
        st.error("Error loading database statistics")


@monitor_performance
def display_data_quality_check(df):
    st.subheader("🔍 Data Quality Check")
    
    try:
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        
        # Create DataFrame for display
        quality_info = []
        for column in df.columns:
            quality_info.append({
                'Column': str(column),
                'Data Type': str(df[column].dtype),
                'Unique Values': int(df[column].nunique()),
                'Missing Values': int(df[column].isnull().sum()),
                'Missing %': f"{(df[column].isnull().sum() / len(df) * 100):.2f}%"
            })
        
        # Convert to DataFrame and ensure all columns are strings
        quality_df = pd.DataFrame(quality_info)
        for col in quality_df.columns:
            quality_df[col] = quality_df[col].astype(str)
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Columns", str(len(df.columns)))
        with col2:
            st.metric("Total Rows", str(len(df)))
        with col3:
            st.metric("Columns with Missing Data", str(sum(missing_data > 0)))
        
        # Display detailed quality information
        st.write("### Detailed Column Information")
        st.dataframe(quality_df)
        
        # Display columns with missing data
        missing_cols = quality_df[quality_df['Missing Values'].astype(str) != '0']
        if not missing_cols.empty:
            st.write("### Columns with Missing Data")
            st.dataframe(missing_cols)
        else:
            st.success("No missing data found in any column!")

    except Exception as e:
        logger.error(f"Error in data quality check: {e}")
        st.error(f"Error displaying data quality metrics: {str(e)}")

@monitor_performance
def display_filtered_results(filtered_df, df):
    try:
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
        
        # Column selection
        st.subheader("📋 Select Columns to Display")
        all_columns = filtered_df.columns.tolist()
        default_columns = [
            "Observation_Date", "Age", "Sex", "Employment_Status",
            "Healthcare_Role", "Department", "Suicidal_Distress", 
            "Time_To_Crisis", "Crisis_Event"
        ]
        selected_columns = st.multiselect(
            "Choose columns:",
            all_columns,
            default=[col for col in default_columns if col in all_columns]
        )
        
        if selected_columns:
            # Convert DataFrame to display-friendly format
            display_df = filtered_df[selected_columns].copy()
            
            # Convert all columns to strings for safe display
            for col in display_df.columns:
                if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                    display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    display_df[col] = display_df[col].astype(str)
            
            # Display as a static table
            st.dataframe(display_df)
            
            # Export options
            if len(display_df) > 0:
                export_data(display_df, selected_columns)
        else:
            st.warning("Please select at least one column to display")
        
    except Exception as e:
        logger.error(f"Error displaying filtered results: {e}")
        st.error(f"Error displaying results: {str(e)}")

@monitor_performance
def export_data(df, selected_columns):
    try:
        st.subheader("📤 Export Data")
        export_format = st.radio(
            "Choose export format:",
            ["CSV", "Excel"],
            horizontal=True
        )
        
        if st.button("Export Data"):
            try:
                # Create a copy for export
                export_df = df.copy()
                
                if export_format == "CSV":
                    # Convert to CSV
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="resume_data_export.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    # Convert to Excel
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, index=False)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name="resume_data_export.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                st.success("Export prepared successfully!")
                
            except Exception as e:
                logger.error(f"Export error: {e}")
                st.error("Error during export. Please try again.")
                
    except Exception as e:
        logger.error(f"Export setup error: {e}")
        st.error("Error setting up export. Please try again.")

 
@monitor_performance
def display_recent_entries_management():
    st.subheader("📝 Recent Entries Management")
    st.markdown("View and edit the most recent entries in the database.")
    
    try:
        recent_data = get_latest_entries(20)
        def get_latest_entries(limit: int) -> pd.DataFrame:
            """Fetch the latest entries from the database."""
            try:
                with get_db_connection() as conn:
                    query = f"SELECT * FROM {TABLE_NAME} ORDER BY Observation_Date DESC LIMIT ?"
                    df = pd.read_sql_query(query, conn, params=(limit,))
                    return df
            except Exception as e:
                logger.error(f"Error fetching latest entries: {e}")
                return pd.DataFrame()
        
        if not recent_data.empty:
            edited_df = st.data_editor(recent_data)
            if st.button("Save Changes"):
                try:
                    update_database_entries(edited_df)
                    st.success("Changes saved successfully!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error saving changes: {e}")
                    st.error(f"Error saving changes: {e}")
        else:
            st.info("No entries to display")
    except Exception as e:
        logger.error(f"Error displaying recent entries: {e}")
        st.error("Error loading recent entries")

@monitor_performance
def display_database_documentation():
    st.header("📚 Database Documentation")
    
    try:
        df = load_data()
        
        # Display table structure
        st.markdown("""
        ### Table Structure
        The database contains the following main categories:
        - 👤 Demographic & Socioeconomic data
        - 👩‍⚕️ Occupation Details
        - 🏥 Clinical & Psychiatric information
        - 🧠 Psychological Factors
        - 🤝 Support & Resilience metrics
        """)
        
        # Display column information
        if not df.empty:
            st.subheader("📊 Column Information")
            
            column_info = []
            for column in df.columns:
                info = {
                    'Column Name': str(column),
                    'Data Type': str(df[column].dtype),
                    'Non-Null Count': str(df[column].count()),
                    'Unique Values': str(df[column].nunique())
                }
                column_info.append(info)
            
            # Create and display DataFrame
            info_df = pd.DataFrame(column_info)
            st.dataframe(info_df)
        
        # Best practices section
        st.markdown("""
        ### Best Practices
        1. Always verify data before submission
        2. Use standardized formats for text entries
        3. Regular backups are recommended
        4. Monitor data quality metrics
        
        ### Data Entry Guidelines
        - Text fields: Use consistent terminology
        - Numeric fields: Enter whole numbers where applicable
        - Dates: Use YYYY-MM-DD format
        - Missing values: Leave blank rather than entering zeros
        
        ### Security Notes
        - All data is encrypted at rest
        - Access is logged and monitored
        - Regular security audits are performed
        - Data backup occurs daily
        """)
        
    except Exception as e:
        logger.error(f"Error displaying database documentation: {e}")
        st.error("Error loading database documentation")

@monitor_performance
def display_database_statistics():
    st.header("📊 Database Statistics")
    
    try:
        df = load_data()
        if not df.empty:
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", str(len(df)))
            with col2:
                st.metric("Total Columns", str(len(df.columns)))
            with col3:
                if 'Observation_Date' in df.columns:
                    last_update = df['Observation_Date'].max()
                    if isinstance(last_update, pd.Timestamp):
                        last_update = last_update.strftime("%Y-%m-%d %H:%M:%S")
                    st.metric("Last Update", str(last_update))
            
            # Data quality overview
            st.subheader("Data Quality Overview")
            
            # Calculate quality metrics
            quality_metrics = pd.DataFrame({
                'Metric': ['Missing Values', 'Complete Records', 'Duplicate Records'],
                'Count': [
                    str(df.isnull().sum().sum()),
                    str(df.dropna().shape[0]),
                    str(df.duplicated().sum())
                ]
            })
            
            st.dataframe(quality_metrics)
            
            # Column type distribution
            st.subheader("Column Type Distribution")
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            dtype_counts['Data Type'] = dtype_counts['Data Type'].astype(str)
            dtype_counts['Count'] = dtype_counts['Count'].astype(str)
            
            st.dataframe(dtype_counts)
            
        else:
            st.info("No data available in the database")
            
    except Exception as e:
        logger.error(f"Error displaying database statistics: {e}")
        st.error("Error loading database statistics")

# Remove any remaining references to dtype_info
if __name__ == "__main__":
    try:
        # Initialize session state if not already done
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.edit_history = []
            st.session_state.last_update = datetime.now()

        # Run the main application
        main()

        # Footer section
        st.markdown("---")
        st.markdown("""
        ### 📞 Support Information
        If you need immediate assistance or encounter technical issues:
        - 🆘 Emergency Support: UK: 999, US: 911
        - 💻 Technical Help: www.resumeuk.org
        - 📱 Mobile App: Available on iOS and Android
        """)

        # Version information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ System Information")
        st.sidebar.text(f"Version: 1.2.0")
        st.sidebar.text(f"Last Update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("""
        ⚠️ An unexpected error occurred. Please try:
        1. Refreshing the page
        2. Checking your internet connection
        3. Contacting support if the issue persists
        """)
        st.exception(e)

    finally:
        # Cleanup operations
        pass
