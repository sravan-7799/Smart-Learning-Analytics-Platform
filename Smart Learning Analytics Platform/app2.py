import streamlit as st
st.set_page_config(page_title="Student Performance Prediction App", page_icon="📊", layout="wide")

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== Custom Header ==========
def custom_header(text, emoji="✨"):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #42a5f5, #1e88e5);
            padding: 14px 25px;
            border-radius: 12px;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            color: white;
            font-size: 1.5em;
            font-weight: 600;
        ">
        {emoji} {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ========== Load External CSS ==========
css_path = os.path.join(BASE_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ========== Load Dataset ==========
csv_path = os.path.join(BASE_DIR, "StudentPerformanceFactors.csv")

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    custom_header("📂 Student Data Preview")
    st.dataframe(data.head(), use_container_width=True, height=250)
else:
    st.error(f"❌ CSV file not found at: {csv_path}")
    st.stop()

# Add binary target column for pass/fail
pass_threshold = 70
data['Pass_Fail'] = data['Exam_Score'].apply(lambda x: 1 if x >= pass_threshold else 0)

# ========== Sidebar ==========
st.sidebar.image(os.path.join(BASE_DIR, "image_home.jpeg"), use_column_width=True)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualization", "Prediction"])

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", False)
if dark_mode:
    st.markdown(
        """
        <style>
        body { background: #121212; color: #e0e0e0; }
        .section { background: #1e1e1e !important; color: #e0e0e0; }
        .result { background: #263238 !important; color: #ffffff !important; }
        </style>
        """, unsafe_allow_html=True
    )

# ========== Home Page ==========
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

# ========== Data Visualization Page ==========
elif page == "Data Visualization":
    custom_header("📊 Data Visualization")

    plot_type = st.sidebar.selectbox("Select Plot Type", 
                                     ["Correlation Heatmap", "Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Scatter Plot", "Histogram"])

    if plot_type in ["Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Histogram"]:
        feature = st.sidebar.selectbox("Select Feature", data.columns)

    elif plot_type == "Scatter Plot":
        x_feature = st.sidebar.selectbox("Select X-axis Feature", data.columns)
        y_feature = st.sidebar.selectbox("Select Y-axis Feature", data.columns)

    st.subheader(f"{plot_type} Visualization")

    if plot_type == "Correlation Heatmap":
        corr = data.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Bar Chart":
        fig = px.bar(data, x=feature, title=f"Bar Chart of {feature}", color=feature)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Pie Chart":
        fig = px.pie(data, names=feature, title=f"Pie Chart of {feature}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Line Chart":
        fig = px.line(data, y=feature, title=f"Line Chart of {feature}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Boxplot":
        fig = px.box(data, y=feature, title=f"Boxplot of {feature}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter Plot":
        fig = px.scatter(data, x=x_feature, y=y_feature, color=x_feature, title=f"Scatter Plot of {x_feature} vs {y_feature}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Histogram":
        fig = px.histogram(data, x=feature, nbins=20, title=f"Histogram of {feature}")
        st.plotly_chart(fig, use_container_width=True)

# ========== Prediction Page ==========
elif page == "Prediction":
    custom_header("🤖 Smart Learning Analytics Platform")

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

    regressor = RandomForestRegressor()
    regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = regressor.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Classification Accuracy", f"{classification_accuracy*100:.2f}%")
    with col2:
        st.metric("📏 Regression RMSE", f"{rmse:.2f}")

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
        with st.spinner("⏳ Analyzing student performance..."):
            time.sleep(1.5)  # Simulate processing
            predicted_class = classifier.predict(input_data)[0]
            grade = "🎉 Pass 🎉" if predicted_class == 1 else "❌ Fail ❌"
            predicted_score = regressor.predict(input_data)[0]

        st.markdown(
            f"""
            <div class='result'>
            📊 <b>Predicted Outcome:</b> {grade}<br>
            📝 <b>Predicted Exam Score:</b> {predicted_score:.2f}
            </div>
            """, unsafe_allow_html=True
        )

        gif_file = "pass.gif" if predicted_class == 1 else "fail.gif"
        gif_path = os.path.join(BASE_DIR, "gifs", gif_file)

        if os.path.exists(gif_path):
            st.image(gif_path, width=300)
        else:
            st.error(f"GIF file not found: {gif_path}")

# ========== Footer ==========
st.markdown(
    """
    <footer>
        Made with ❤️ using Streamlit | Smart Learning Analytics Platform © 2025
    </footer>
    """,
    unsafe_allow_html=True
)
