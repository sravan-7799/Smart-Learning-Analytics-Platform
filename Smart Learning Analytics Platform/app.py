import streamlit as st
st.set_page_config(page_title="Student Performance Prediction App", page_icon="📊")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load external stylesheet
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

csv_path = os.path.join(BASE_DIR, "StudentPerformanceFactors.csv")

# Check if file exists before reading
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    st.write("### Student Data Preview", data.head())
else:
    st.error(f"CSV file not found at: {csv_path}")


# Add binary target column for pass/fail
pass_threshold = 70  # Define a passing threshold for exam scores
data['Pass_Fail'] = data['Exam_Score'].apply(lambda x: 1 if x >= pass_threshold else 0)

# Sidebar for page selection
page = st.sidebar.selectbox("Select Page", ["Home", "Data Visualization", "Prediction"])

# Home Page
if page == "Home":
    st.title("Welcome to the Student Performance Analysis App")
    image_path = os.path.join(BASE_DIR, "image_home.jpeg")
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.warning("⚠️ image_home.jpeg not found. Please place it in the same folder as app.py")

    st.write(
        """
        This app predicts student performance based on various factors. 
        Use the Data Visualization page to explore insights in the dataset,
        or head to the Prediction page to make predictions based on your inputs.
        """
    )

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("Explore various visualizations of the student performance dataset.")

    # Sidebar for selecting plot type
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Correlation Heatmap", "Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Scatter Plot", "Histogram"])

    # Dropdown for feature selection based on the plot type
    if plot_type in ["Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Histogram"]:
        feature = st.sidebar.selectbox("Select Feature", data.columns)

    elif plot_type == "Scatter Plot":
        x_feature = st.sidebar.selectbox("Select X-axis Feature", data.columns)
        y_feature = st.sidebar.selectbox("Select Y-axis Feature", data.columns)

    # Plotting based on selected plot type
    st.subheader(f"{plot_type} Visualization")

    # Correlation Heatmap
    if plot_type == "Correlation Heatmap":
        st.write("Correlation Heatmap of the Dataset")
        
        # Select only numerical columns for correlation calculation
        numerical_data = data.select_dtypes(include=[np.number])
        corr = numerical_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Bar Chart
    elif plot_type == "Bar Chart":
        st.write(f"Bar Chart of {feature}")
        fig, ax = plt.subplots()
        data[feature].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Pie Chart
    elif plot_type == "Pie Chart":
        st.write(f"Pie Chart of {feature}")
        fig, ax = plt.subplots()
        data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

    # Line Chart
    elif plot_type == "Line Chart":
        st.write(f"Line Chart of {feature}")
        fig, ax = plt.subplots()
        data[feature].plot(kind='line', ax=ax)
        ax.set_xlabel("Index")
        ax.set_ylabel(feature)
        st.pyplot(fig)

    # Boxplot
    elif plot_type == "Boxplot":
        st.write(f"Boxplot of {feature}")
        fig, ax = plt.subplots()
        sns.boxplot(data=data, y=feature, ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    elif plot_type == "Scatter Plot":
        st.write(f"Scatter Plot between {x_feature} and {y_feature}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_feature, y=y_feature, ax=ax)
        st.pyplot(fig)

    # Histogram
    elif plot_type == "Histogram":
        st.write(f"Histogram of {feature}")
        fig, ax = plt.subplots()
        data[feature].plot(kind='hist', bins=20, ax=ax)
        ax.set_xlabel(feature)
        st.pyplot(fig)

# Prediction Page
elif page == "Prediction":
    st.title("Smart Learning Analytics Platform")

    # Define features and target variable for classification and regression
    classification_target = 'Pass_Fail'  # Binary target: 1 for pass, 0 for fail
    regression_target = 'Exam_Score'  # Target for score prediction
    feature_columns = ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 'Tutoring_Sessions']
    X = data[feature_columns]
    y_class = data[classification_target]
    y_reg = data[regression_target]

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Split data
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Train classification model
    classifier = RandomForestClassifier()
    classifier.fit(X_train_class, y_train_class)

    # Calculate classification accuracy
    y_pred_class = classifier.predict(X_test_class)
    classification_accuracy = accuracy_score(y_test_class, y_pred_class)

    st.write(f"Classification Model Accuracy: {classification_accuracy * 100:.2f}%")

    # Train regression model
    regressor = RandomForestRegressor()
    regressor.fit(X_train_reg, y_train_reg)

    # Calculate regression accuracy (RMSE)
    y_pred_reg = regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)

    st.write(f"Regression Model RMSE: {rmse:.2f}")

    # Input fields for the specific features
    st.subheader("Enter student data:")

    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
    hours_studied = st.number_input("Hours Studied per Week", min_value=0.0, max_value=50.0, step=0.5)
    previous_scores = st.number_input("Previous Scores (out of 100)", min_value=0.0, max_value=100.0, step=1.0)
    access_to_resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
    tutoring_sessions = st.number_input("Number of Tutoring Sessions per Week", min_value=0, max_value=7, step=1)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Attendance': [attendance],
        'Hours_Studied': [hours_studied],
        'Previous_Scores': [previous_scores],
        'Access_to_Resources': [access_to_resources],
        'Tutoring_Sessions': [tutoring_sessions]
    })

    # One-hot encode input data and align with training data columns
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X_train_class.columns, fill_value=0)

    # Make predictions
    if st.button("Predict"):
        # Predict pass/fail outcome
        predicted_class = classifier.predict(input_data)[0]
        grade = "🎉 Pass 🎉" if predicted_class == 1 else "❌ Fail ❌"
        
        # Predict exam score
        predicted_score = regressor.predict(input_data)[0]

        st.write(f"Predicted Outcome: {grade}")
        st.write(f"Predicted Exam Score: {predicted_score:.2f}")

        # Display GIF based on pass/fail outcome
        if predicted_class == 1:
            st.image('pass.gif', width=300)  # Replace with your pass GIF file path
        else:
            st.image('fail.gif', width=300)  # Replace with your fail GIF file path
