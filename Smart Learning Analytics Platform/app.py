import os
import time
import json
import io

import streamlit as st
st.set_page_config(page_title="Student Performance Prediction App", page_icon="📊", layout="wide")

import pandas as pd
import numpy as np

# Plotly first for interactivity
import plotly.express as px
import plotly.graph_objects as go

# Matplotlib only for rare fallbacks
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, mean_squared_error, confusion_matrix, r2_score
)

# --------- Constants / Paths (do not change) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "StudentPerformanceFactors.csv")
IMG_HOME = os.path.join(BASE_DIR, "image_home.jpeg")
GIFS_DIR = os.path.join(BASE_DIR, "gifs")
STYLE_PATH = os.path.join(BASE_DIR, "style.css")

# --------- Utilities ----------
@st.cache_data(show_spinner=False)
def load_csv_safely(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def inject_css():
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def header_block(text: str, emoji: str = "✨"):
    st.markdown(
        f"""
        <div class="slap-header">
            <span class="emoji">{emoji}</span>
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def pill(text: str):
    st.markdown(f'<span class="pill">{text}</span>', unsafe_allow_html=True)

def safe_plotly_chart(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --------- Load CSS ----------
inject_css()

# --------- Sidebar ----------
with st.sidebar:
    if os.path.exists(IMG_HOME):
        st.image(IMG_HOME, use_column_width=True)
    else:
        st.warning("⚠️ Home image not found. Please add it to the project folder.")
    st.title("📊 Navigation")
    page = st.radio("Go to", ["Home", "Data Visualization", "Prediction"], index=0)
    dark_mode = st.toggle("🌙 Dark Mode", False)

# Optional dark-mode CSS injection (non-destructive)
if dark_mode:
    st.markdown(
        """
        <style>
        body { background: #0f1116; color: #e0e3eb; }
        .section, .slap-card { background: #151926 !important; color: #e0e3eb !important; }
        .result { background: #1c2233 !important; color: #ffffff !important; }
        .pill { background: #1e2a44 !important; color: #e0e3eb !important; }
        .slap-header { background: linear-gradient(90deg, #233048, #1d2740) !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------- Data loading (keeps your path; adds safe fallback) ----------
data = load_csv_safely(CSV_PATH)

# If CSV missing, allow upload without changing the default path behavior
if data.empty:
    st.error(f"❌ CSV file not found at: {CSV_PATH}")
    with st.expander("Upload a CSV temporarily (optional)"):
        up = st.file_uploader("Upload CSV to proceed (won't replace your default path)", type=["csv"])
        if up is not None:
            data = pd.read_csv(up)
            st.success("✅ Using uploaded CSV for this session.")
        else:
            st.stop()

# --------- Add binary target column (exact same logic) ----------
pass_threshold = 70
# Ensure column present
if 'Exam_Score' not in data.columns:
    st.error("❌ 'Exam_Score' column not found in dataset. Please include it.")
    st.stop()

data['Pass_Fail'] = data['Exam_Score'].apply(lambda x: 1 if x >= pass_threshold else 0)

# --------- Shared Feature Setup (unchanged) ----------
classification_target = 'Pass_Fail'
regression_target = 'Exam_Score'
feature_columns = ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 'Tutoring_Sessions']
missing_feats = [c for c in feature_columns if c not in data.columns]
if missing_feats:
    st.error(f"❌ Missing required feature columns: {missing_feats}")
    st.stop()

# For modeling
def build_matrices(df: pd.DataFrame):
    X = df[feature_columns].copy()
    y_class = df[classification_target]
    y_reg = df[regression_target]
    X = pd.get_dummies(X)  # safe for 'Access_to_Resources'
    return X, y_class, y_reg

# --------- Home Page ----------
if page == "Home":
    st.markdown('<h1 class="title">Welcome to the Student Performance Analysis App</h1>', unsafe_allow_html=True)

    # Hero section
    with st.container():
        colA, colB = st.columns([1.2, 1])
        with colA:
            st.markdown(
                """
                <div class="section">
                    <p>
                    Predict <b>student performance</b> with a clean, modern dashboard.<br>
                    Explore trends in <b>Data Visualization</b> and test scenarios in <b>Prediction</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            pill("🚀 Random Forest Models")
            pill("📈 Interactive Plotly Charts")
            pill("🌓 Light/Dark Mode")
            pill("💾 Downloadable Outputs")
        with colB:
            if os.path.exists(IMG_HOME):
                st.image(IMG_HOME, use_column_width=True, caption="Smart Learning Analytics")
            else:
                st.info("ℹ️ Add image_home.jpeg next to app.py for a hero image.")

    # Quick KPIs
    header_block("Overview", "📌")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{len(data):,}")
    with c2:
        st.metric("Features Used", f"{len(feature_columns)}")
    with c3:
        st.metric("Pass Threshold", f"{pass_threshold}")
    with c4:
        st.metric("Pass Rate", f"{(data['Pass_Fail'].mean()*100):.1f}%")

    # Sample preview
    st.markdown("### 📂 Student Data Preview")
    st.dataframe(data.head(20), use_container_width=True, height=300)

    # Quick distribution (Exam Score)
    if pd.api.types.is_numeric_dtype(data['Exam_Score']):
        fig = px.histogram(data, x='Exam_Score', nbins=30, title="Exam Score Distribution", opacity=0.9)
        safe_plotly_chart(fig)

    # Template & current-data downloads
    with st.expander("📥 Downloads"):
        # Template
        template = pd.DataFrame({
            'Attendance': [90],
            'Hours_Studied': [12],
            'Previous_Scores': [78],
            'Access_to_Resources': ['Medium'],
            'Tutoring_Sessions': [2],
            'Exam_Score': [82]
        })
        buf_t = io.StringIO(); template.to_csv(buf_t, index=False)
        st.download_button("⬇️ Download Input Template CSV", buf_t.getvalue(), file_name="template_student_performance.csv", mime="text/csv")

        # Current data head
        buf_h = io.StringIO(); data.head(100).to_csv(buf_h, index=False)
        st.download_button("⬇️ Download Current Data (Head)", buf_h.getvalue(), file_name="student_data_sample.csv", mime="text/csv")

# --------- Data Visualization ----------
elif page == "Data Visualization":
    header_block("Data Visualization", "📊")

    # Tabs inside Data Visualization
    t1, t2, t3, t4 = st.tabs(["Overview", "Univariate", "Bivariate", "Correlation"])

    with t1:
        st.markdown("#### Quick Insights")
        left, right = st.columns([1,1])
        with left:
            # Bar: Access_to_Resources vs Count
            if 'Access_to_Resources' in data.columns:
                fig = px.bar(data, x='Access_to_Resources', title="Resource Access Distribution", text_auto=True)
                safe_plotly_chart(fig)
        with right:
            # Pie: Pass/Fail
            fig = px.pie(data, names='Pass_Fail', title="Pass vs Fail", hole=0.35)
            safe_plotly_chart(fig)

    with t2:
        st.markdown("### 📊 Univariate Analysis")

        # Column selection with search
        col = st.selectbox(
            "🔎 Select a column for analysis",
            data.columns,
            index=list(data.columns).index("Exam_Score") if "Exam_Score" in data.columns else 0,
        )

        # Numeric column case
        if pd.api.types.is_numeric_dtype(data[col]):
            tab1, tab2 = st.tabs(["📈 Distribution", "📦 Boxplot"])

            with tab1:
                fig = px.histogram(
                    data,
                    x=col,
                    nbins=30,
                    title=f"Distribution of {col}",
                    marginal="rug",
                    color_discrete_sequence=["#636EFA"],
                )
                fig.update_layout(
                    xaxis_title=col,
                    yaxis_title="Frequency",
                    template="plotly_white",
                    bargap=0.05,
                )
                safe_plotly_chart(fig)

            with tab2:
                fig2 = px.box(
                    data,
                    y=col,
                    title=f"Boxplot of {col}",
                    color_discrete_sequence=["#EF553B"],
                )
                fig2.update_layout(
                    yaxis_title=col,
                    template="plotly_white",
                )
                safe_plotly_chart(fig2)

        # Categorical column case
        else:
            fig = px.bar(
                data[col].value_counts().reset_index(),
                x="index",
                y=col,
                text_auto=True,
                title=f"Counts of {col}",
                color_discrete_sequence=["#00CC96"],
            )
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Count",
                template="plotly_white",
            )
            safe_plotly_chart(fig)


    with t3:
        st.markdown("#### Bivariate")
        x_feature = st.selectbox("X", data.columns, index=data.columns.get_loc('Hours_Studied') if 'Hours_Studied' in data.columns else 0)
        y_feature = st.selectbox("Y", data.columns, index=data.columns.get_loc('Exam_Score') if 'Exam_Score' in data.columns else 1)
        color_by = st.selectbox("Color By (optional)", ["(none)"] + list(data.columns), index=0)

        if x_feature and y_feature and x_feature != y_feature:
            if color_by == "(none)":
                fig = px.scatter(data, x=x_feature, y=y_feature, trendline="ols", title=f"{x_feature} vs {y_feature}")
            else:
                fig = px.scatter(data, x=x_feature, y=y_feature, color=color_by, trendline="ols", title=f"{x_feature} vs {y_feature} by {color_by}")
            safe_plotly_chart(fig)

    with t4:
        st.markdown("#### Correlation Heatmap (numeric only)")
        corr = data.corr(numeric_only=True)
        if corr.shape[0] > 0:
            fig = px.imshow(corr, text_auto=False, color_continuous_scale="Blues", title="Feature Correlations")
            safe_plotly_chart(fig)
        else:
            st.info("No numeric columns available for correlation.")

# --------- Prediction ----------
elif page == "Prediction":
    header_block("Smart Learning Analytics Platform", "🤖")

    # Prepare matrices
    X, y_class, y_reg = build_matrices(data)

    # Split (unchanged)
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Train models (same algorithms)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train_class, y_train_class)
    y_pred_class = classifier.predict(X_test_class)
    classification_accuracy = accuracy_score(y_test_class, y_pred_class)

    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = regressor.predict(X_test_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)

    # KPI cards
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("✅ Classification Accuracy", f"{classification_accuracy*100:.2f}%")
    with k2:
        st.metric("📏 Regression RMSE", f"{rmse:.2f}")
    with k3:
        st.metric("📐 Regression R²", f"{r2:.3f}")

    # Quick charts (Confusion Matrix / Residuals)
    ctab = st.tabs(["Classifier: Confusion Matrix", "Regressor: Residuals", "Feature Importance"])
    with ctab[0]:
        cm = confusion_matrix(y_test_class, y_pred_class, labels=[0,1])
        z = cm
        x = ["Fail (0)", "Pass (1)"]
        y = ["Fail (0)", "Pass (1)"]
        fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=0, zmax=max(cm.flatten()) if cm.size else 1))
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        safe_plotly_chart(fig)

    with ctab[1]:
        residuals = y_test_reg - y_pred_reg
        fig = px.scatter(x=y_pred_reg, y=residuals, labels={'x': 'Predicted Score', 'y': 'Residuals'}, title="Residuals vs Predicted")
        fig.add_hline(y=0, line_dash="dot")
        safe_plotly_chart(fig)

    with ctab[2]:
        # Use the regressor for importance (continuous target), fall back to classifier if needed
        importances = None
        model_name = ""
        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            model_name = "RandomForestRegressor"
        elif hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            model_name = "RandomForestClassifier"

        if importances is not None:
            imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
            fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title=f"Feature Importance ({model_name})")
            safe_plotly_chart(fig)
        else:
            st.info("Feature importances are not available for this model.")

    st.markdown("### 📝 Enter student data")

    # Inputs (unchanged fields; improved help text)
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0, help="Overall attendance percentage")
        hours_studied = st.number_input("Hours Studied per Week", min_value=0.0, max_value=50.0, step=0.5, help="Average hours studied per week")
        previous_scores = st.number_input("Previous Scores (out of 100)", min_value=0.0, max_value=100.0, step=1.0, help="Average previous exam scores")
    with col2:
        access_to_resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'], help="Perceived access to learning resources")
        tutoring_sessions = st.number_input("Tutoring Sessions per Week", min_value=0, max_value=7, step=1, help="Number of weekly tutoring sessions")

    # Show a compact summary card of inputs
    with st.expander("👤 Student Input Summary"):
        st.json({
            "Attendance": attendance,
            "Hours_Studied": hours_studied,
            "Previous_Scores": previous_scores,
            "Access_to_Resources": access_to_resources,
            "Tutoring_Sessions": tutoring_sessions
        })

    # Prepare model input (unchanged logic)
    input_data = pd.DataFrame({
        'Attendance': [attendance],
        'Hours_Studied': [hours_studied],
        'Previous_Scores': [previous_scores],
        'Access_to_Resources': [access_to_resources],
        'Tutoring_Sessions': [tutoring_sessions]
    })
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X_train_class.columns, fill_value=0)

    # Predict button
    if st.button("🚀 Predict"):
        with st.spinner("⏳ Analyzing student performance..."):
            time.sleep(1.0)  # gentle UX pause
            predicted_class = classifier.predict(input_data)[0]
            grade = "🎉 Pass 🎉" if predicted_class == 1 else "❌ Fail ❌"
            predicted_score = float(regressor.predict(input_data)[0])

        # Result card
        st.markdown(
            f"""
            <div class='result'>
            📊 <b>Predicted Outcome:</b> {grade}<br>
            📝 <b>Predicted Exam Score:</b> {predicted_score:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Celebrations
        if predicted_class == 1:
            st.balloons()
        else:
            st.snow()

        # Keep your GIFs logic & paths
        gif_file = "pass.gif" if predicted_class == 1 else "fail.gif"
        gif_path = os.path.join(GIFS_DIR, gif_file)
        if os.path.exists(gif_path):
            st.image(gif_path, width=300)
        else:
            st.info(f"ℹ️ Optional GIF not found at {gif_path} — add it to enhance feedback.")

        # Download prediction artifact
        result_payload = {
            "Predicted_Pass_Fail": int(predicted_class),
            "Predicted_Exam_Score": round(predicted_score, 2),
            "Inputs": {
                "Attendance": attendance,
                "Hours_Studied": hours_studied,
                "Previous_Scores": previous_scores,
                "Access_to_Resources": access_to_resources,
                "Tutoring_Sessions": tutoring_sessions
            }
        }
        st.download_button(
            "⬇️ Download Prediction (JSON)",
            data=json.dumps(result_payload, indent=2),
            file_name="prediction_result.json",
            mime="application/json"
        )

# --------- Footer ----------
st.markdown(
    """
    <footer>
        Made with ❤️ using Streamlit | Smart Learning Analytics Platform © 2025
    </footer>
    """,
    unsafe_allow_html=True
)