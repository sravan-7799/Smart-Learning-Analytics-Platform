import streamlit as st
st.set_page_config(page_title="Student Performance Prediction App", page_icon="📊", layout="wide")
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

# Load dataset
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    st.markdown("### 📂 Student Data Preview")
    st.dataframe(data.head(), use_container_width=True)
else:
    st.error(f"❌ CSV file not found at: {csv_path}")

# Add binary target column for pass/fail
pass_threshold = 70
data['Pass_Fail'] = data['Exam_Score'].apply(lambda x: 1 if x >= pass_threshold else 0)

# Sidebar
st.sidebar.image(os.path.join(BASE_DIR, "image_home.jpeg"), use_column_width=True)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualization", "Prediction"])

# Home Page
if page == "Home":
    st.markdown('<h1 class="title">Welcome to the Student Performance Analysis App</h1>', unsafe_allow_html=True)

    image_path = os.path.join(BASE_DIR, "image_home.jpeg")
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True, caption="Smart Learning Analytics")
    else:
        st.warning("⚠️ image_home.jpeg not found. Please place it in the same folder as app.py")

    st.markdown(
        """
        <div class="section">
            <p>
            This app predicts <b>student performance</b> based on multiple factors.  
            ✅ Explore insights in the <b>Data Visualization</b> page.  
            ✅ Use the <b>Prediction</b> page to test with your own inputs.  
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Data Visualization Page
elif page == "Data Visualization":
    st.markdown('<h1 class="title">📊 Data Visualization</h1>', unsafe_allow_html=True)

    plot_type = st.sidebar.selectbox("Select Plot Type", 
                                     ["Correlation Heatmap", "Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Scatter Plot", "Histogram"])

    if plot_type in ["Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Histogram"]:
        feature = st.sidebar.selectbox("Select Feature", data.columns)

    elif plot_type == "Scatter Plot":
        x_feature = st.sidebar.selectbox("Select X-axis Feature", data.columns)
        y_feature = st.sidebar.selectbox("Select Y-axis Feature", data.columns)

    st.subheader(f"{plot_type} Visualization")

    if plot_type == "Correlation Heatmap":
        numerical_data = data.select_dtypes(include=[np.number])
        corr = numerical_data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

    elif plot_type == "Bar Chart":
        fig, ax = plt.subplots()
        data[feature].value_counts().plot(kind='bar', ax=ax, color="#2E86C1")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif plot_type == "Pie Chart":
        fig, ax = plt.subplots()
        data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, colors=["#2E86C1","#5DADE2","#AED6F1"])
        ax.set_ylabel('')
        st.pyplot(fig)

    elif plot_type == "Line Chart":
        fig, ax = plt.subplots()
        data[feature].plot(kind='line', ax=ax, color="#2E86C1", linewidth=2)
        ax.set_xlabel("Index")
        ax.set_ylabel(feature)
        st.pyplot(fig)

    elif plot_type == "Boxplot":
        fig, ax = plt.subplots()
        data[feature].plot(kind='box', ax=ax, color="#2E86C1")
        st.pyplot(fig)

    elif plot_type == "Scatter Plot":
        fig, ax = plt.subplots()
        ax.scatter(data[x_feature], data[y_feature], alpha=0.7, color="#2E86C1")
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        st.pyplot(fig)

    elif plot_type == "Histogram":
        fig, ax = plt.subplots()
        data[feature].plot(kind='hist', bins=20, ax=ax, color="#2E86C1", alpha=0.8)
        ax.set_xlabel(feature)
        st.pyplot(fig)

# Prediction Page
elif page == "Prediction":
    st.markdown('<h1 class="title">🤖 Smart Learning Analytics Platform</h1>', unsafe_allow_html=True)

    classification_target = 'Pass_Fail'
    regression_target = 'Exam_Score'
    feature_columns = ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 'Tutoring_Sessions']
    X = data[feature_columns]
    y_class = data[classification_target]
    y_reg = data[regression_target]

    X = pd.get_dummies(X)

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier()
    classifier.fit(X_train_class, y_train_class)
    y_pred_class = classifier.predict(X_test_class)
    classification_accuracy = accuracy_score(y_test_class, y_pred_class)

    st.markdown(f"<div class='section result'>✅ Classification Accuracy: {classification_accuracy*100:.2f}%</div>", unsafe_allow_html=True)

    regressor = RandomForestRegressor()
    regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = regressor.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    st.markdown(f"<div class='section result'>📏 Regression RMSE: {rmse:.2f}</div>", unsafe_allow_html=True)

    st.subheader("📝 Enter student data:")

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
        hours_studied = st.number_input("Hours Studied per Week", min_value=0.0, max_value=50.0, step=0.5)
        previous_scores = st.number_input("Previous Scores (out of 100)", min_value=0.0, max_value=100.0, step=1.0)
    with col2:
        access_to_resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
        tutoring_sessions = st.number_input("Tutoring Sessions per Week", min_value=0, max_value=7, step=1)

    input_data = pd.DataFrame({
        'Attendance': [attendance],
        'Hours_Studied': [hours_studied],
        'Previous_Scores': [previous_scores],
        'Access_to_Resources': [access_to_resources],
        'Tutoring_Sessions': [tutoring_sessions]
    })

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X_train_class.columns, fill_value=0)

    if st.button("🚀 Predict"):
        predicted_class = classifier.predict(input_data)[0]
        grade = "🎉 Pass 🎉" if predicted_class == 1 else "❌ Fail ❌"
        predicted_score = regressor.predict(input_data)[0]

        st.markdown(f"<div class='result'>Predicted Outcome: {grade}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'>Predicted Exam Score: {predicted_score:.2f}</div>", unsafe_allow_html=True)

        gif_file = "pass.gif" if predicted_class == 1 else "fail.gif"
        gif_path = os.path.join(BASE_DIR, "gifs", gif_file)

        if os.path.exists(gif_path):
            st.image(gif_path, width=300)
        else:
            st.error(f"GIF file not found: {gif_path}")
