# Import required libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="HR Employee Attrition Analysis",
    page_icon="📊",
    layout="wide"
)

# Cache the dataset loading
@st.cache_data
def load_dataset():
    """Load and preprocess the dataset."""
    try:
        # Try different paths for the dataset
        possible_paths = [
            "data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
            os.path.join(os.path.dirname(__file__), "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Convert categorical variables
                categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
                for col in categorical_columns:
                    df[col] = df[col].astype('category').cat.codes
                return df
        
        raise FileNotFoundError("Dataset not found. Please upload the dataset file to the 'data' directory.")
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Define the main function
def main():
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Data Analysis", "Model Training", "Predictions", "About Dataset"],
        key="page_selector"
    )

    # Show loading spinner while loading dataset
    with st.spinner("Loading dataset..."):
        df = load_dataset()
        if df is None:
            return

    if page == "Home":
        st.title("HR Employee Attrition Analysis")
        st.write("Welcome to the HR Employee Attrition Analysis Dashboard!")
        st.write("This interactive dashboard helps analyze employee attrition patterns and predict future attrition.")
        st.write("The goal is to predict which employees are likely to leave the company based on various features.")
        st.write("Understanding employee attrition patterns can help organizations take proactive measures to retain valuable employees.")
        st.write("Navigate through different sections using the sidebar to explore the data and build predictive models.")

    elif page == "Data Analysis":
        st.header("Data Analysis")
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Show basic statistics
        st.subheader("Basic Statistics")
        st.write(df.describe())
        
        # Show attrition distribution
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='Attrition')
        plt.title('Distribution of Attrition')
        st.pyplot(fig)
        
        # Show correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
        # Show top correlated features
        st.subheader("Top Correlated Features with Attrition")
        attrition_corr = corr['Attrition'].sort_values(ascending=False)
        top_corr = attrition_corr[attrition_corr.abs() > 0.1]
        st.bar_chart(top_corr)

    elif page == "Model Training":
        st.header("Model Training")
        
        with st.spinner("Training model..."):
            try:
                # Select features
                features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
                
                # Split the data
                X = df[features]
                y = df['Attrition']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train the model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Show metrics
                st.subheader("Model Performance")
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label=1)
                recall = recall_score(y_test, y_pred, pos_label=1)
                f1 = f1_score(y_test, y_pred, pos_label=1)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precision", f"{precision:.2%}")
                col3.metric("Recall", f"{recall:.2%}")
                col4.metric("F1 Score", f"{f1:.2%}")
                
                # Show classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report)
                
                # Show confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)
                
                # Show feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': abs(model.coef_[0])
                }).sort_values('Importance', ascending=False)
                st.bar_chart(feature_importance.set_index('Feature'))
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

    elif page == "Predictions":
        st.header("Predict Employee Attrition")
        
        # Create input form
        st.write("Enter employee details to predict attrition:")
        
        # Create input fields
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        distance_from_home = st.number_input("Distance from Home", min_value=0, max_value=100, value=10)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5], key="education_level")
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], key="job_level")
        monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=2)
        
        # Create input data
        input_data = {
            'Age': age,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'JobLevel': job_level,
            'MonthlyIncome': monthly_income,
            'YearsAtCompany': years_at_company
        }
        
        # Train the model
        features = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 'YearsAtCompany']
        X = df[features]
        y = df['Attrition']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Make prediction
        if st.button("Predict Attrition"):
            try:
                # Preprocess input data
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)[:, 1][0]
                
                # Display result
                if prediction[0] == 1:
                    st.error(f"This employee is likely to leave the company.")
                else:
                    st.success(f"This employee is likely to stay with the company.")
                
                st.write(f"Probability of attrition: {probability:.2%}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    elif page == "About Dataset":
        st.header("About Dataset")
        
        st.write("The HR Employee Attrition dataset contains information about employees and their characteristics.")
        st.write("The goal is to predict which employees are likely to leave the company based on various features.")
        
        st.subheader("Features Description")
        st.write("- Age: Age of the employee")
        st.write("- BusinessTravel: Travel requirement for the job")
        st.write("- Department: Department of the employee")
        st.write("- DistanceFromHome: Distance from home to work")
        st.write("- Education: Education level")
        st.write("- JobLevel: Level of the job")
        st.write("- MonthlyIncome: Monthly income")
        st.write("- YearsAtCompany: Number of years at the company")
        st.write("- ... and many more features")
        
        st.subheader("Target Variable")
        st.write("- Attrition: Whether the employee left the company (1) or stayed (0)")
        
        st.subheader("Data Source")
        st.write("The dataset is from Kaggle and contains synthetic employee data.")

if __name__ == "__main__":
    main()

# Define the main function
def main():
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Data Analysis", "Model Training", "Predictions", "About Dataset"],
        key="page_selector"
    )

    if page == "Home":
        st.title("HR Employee Attrition Analysis")
        st.write("Welcome to the HR Employee Attrition Analysis Dashboard!")
        st.write("This interactive dashboard helps analyze employee attrition patterns and predict future attrition.")
        st.write("The goal is to predict which employees are likely to leave the company based on these features.")
        st.write("Understanding employee attrition patterns can help organizations take proactive measures to retain valuable employees.")
        st.write("Navigate through different sections using the sidebar to explore the data and build predictive models.")

    elif page == "Data Analysis":
        st.header("Data Analysis")
        
        # Load the dataset
        try:
            # Try different paths for the dataset
            possible_paths = [
                "data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
                "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
                "WA_Fn-UseC_-HR-Employee-Attrition.csv"
            ]
            
            dataset_path = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    dataset_path = path
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    st.error(f"Error loading dataset from {path}: {str(e)}")
                    return
            
            if dataset_path:
                st.success(f"Dataset loaded successfully from {dataset_path}")
            else:
                st.error("Dataset not found. Please upload the dataset file to the 'data' directory.")
                return
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return
        
        # Convert categorical variables
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Show basic statistics
        st.subheader("Basic Statistics")
        st.write(df.describe())
        
        # Show attrition distribution
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x='Attrition')
        plt.title('Distribution of Attrition')
        st.pyplot(fig)
        
        # Show correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
        # Show top correlated features
        st.subheader("Top Correlated Features with Attrition")
        attrition_corr = corr['Attrition'].sort_values(ascending=False)
        top_corr = attrition_corr[attrition_corr.abs() > 0.1]
        st.bar_chart(top_corr)

    elif page == "Model Training":
        st.header("Model Training")
        
        # Load and preprocess data
        try:
            df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
            
            # Convert categorical variables
            categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
            for col in categorical_columns:
                df[col] = df[col].astype('category').cat.codes
            
            # Select features
            features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
            
            # Split the data
            X = df[features]
            y = df['Attrition']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Show metrics
            st.subheader("Model Performance")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("Precision", f"{precision:.2%}")
            col3.metric("Recall", f"{recall:.2%}")
            col4.metric("F1 Score", f"{f1:.2%}")
            
            # Show classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)
            
            # Show confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
            # Show feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': abs(model.coef_[0])
            }).sort_values('Importance', ascending=False)
            st.bar_chart(feature_importance.set_index('Feature'))
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")

    elif page == "Predictions":
        st.header("Predict Employee Attrition")
        
        # Load and preprocess data
        try:
            df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
            
            # Convert categorical variables
            categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']
            for col in categorical_columns:
                df[col] = df[col].astype('category').cat.codes
            
            # Select features
            features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
            
            # Create input form
            st.write("Enter employee details to predict attrition:")
            
            # Create input fields
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            distance_from_home = st.number_input("Distance from Home", min_value=0, max_value=100, value=10)
            education = st.selectbox("Education Level", [1, 2, 3, 4, 5], key="education_level")
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], key="job_level")
            monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=2)
            
            # Create input data
            input_data = {
                'Age': age,
                'DistanceFromHome': distance_from_home,
                'Education': education,
                'JobLevel': job_level,
                'MonthlyIncome': monthly_income,
                'YearsAtCompany': years_at_company
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Train the model
            X = df[features]
            y = df['Attrition']
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            # Make prediction
            if st.button("Predict Attrition"):
                try:
                    # Preprocess input data
                    for col in categorical_columns[:-1]:  # Exclude Attrition
                        if col in input_df.columns:
                            input_df[col] = input_df[col].astype('category').cat.codes
                    
                    # Align input features with training features
                    input_features = input_df[features]
                    
                    # Make prediction
                    prediction = model.predict(input_features)
                    probability = model.predict_proba(input_features)[:, 1][0]
                    
                    # Display result
                    if prediction[0] == 1:
                        st.error(f"This employee is likely to leave the company.")
                    else:
                        st.success(f"This employee is likely to stay with the company.")
                    
                    st.write(f"Probability of attrition: {probability:.2%}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    elif page == "About Dataset":
        st.header("About Dataset")
        
        st.write("The HR Employee Attrition dataset contains information about employees and their characteristics.")
        st.write("The goal is to predict which employees are likely to leave the company based on various features.")
        
        st.subheader("Features Description")
        st.write("- Age: Age of the employee")
        st.write("- BusinessTravel: Travel requirement for the job")
        st.write("- Department: Department of the employee")
        st.write("- DistanceFromHome: Distance from home to work")
        st.write("- Education: Education level")
        st.write("- JobLevel: Level of the job")
        st.write("- MonthlyIncome: Monthly income")
        st.write("- YearsAtCompany: Number of years at the company")
        st.write("- ... and many more features")
        
        st.subheader("Target Variable")
        st.write("- Attrition: Whether the employee left the company (1) or stayed (0)")
        
        st.subheader("Data Source")
        st.write("The dataset is from Kaggle and contains synthetic employee data.")

if __name__ == "__main__":
    main()
