
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import altair as alt

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DigitalExposome | Wellbeing AI",
    page_icon="🌿",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "ouputs" / "model" / "wellbeing_model.joblib"

LABELS = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# ==========================================
# 2. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    bundle = load_model()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle["feature_cols"]
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def engineer_features(X):
    X = X.copy()

    X["physio_index"] = X[
        ["IBI", "HR", "BVP", "EDA"]
    ].mean(axis=1)

    X["env_index"] = X[
        ["NO2", "NH3", "CO", "Noise"]
    ].mean(axis=1)

    X["pm_ratio_1_10"] = (
        X["PM1"] / (X["PM10"] + 1e-6)
    )

    X["pm_ratio_25_10"] = (
        X["PM25"] / (X["PM10"] + 1e-6)
    )

    X["physio_env_interaction"] = (
        X["physio_index"] * X["env_index"]
    )

    return X

# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.title("🌿 DigitalExposome Wellbeing AI")
st.write(
    "A machine learning research application for "
    "five-class wellbeing classification using "
    "environmental and physiological measurements."
)

st.caption(
    f"Model: {bundle.get('model_name', 'Random Forest')} | "
    f"Imbalance handling: {bundle.get('imbalance_method', 'SMOTE')}"
)

st.info(
    "Research prototype only. Predictions are model estimates, "
    "not medical diagnoses or clinical assessments."
)

# ==========================================
# 5. NAVIGATION
# ==========================================

prediction_tab, performance_tab, about_tab = st.tabs(
    ["🔍 Wellbeing Prediction", "📊 Model Performance", "ℹ️ About"]
)

# ==========================================
# 6. PREDICTION PAGE
# ==========================================

with prediction_tab:

    st.subheader("Enter Sensor Measurements")

    st.write(
        "Provide the 11 input measurements using the same "
        "units and measurement conventions as the training dataset."
    )

    input_values = {}

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_cols):

        target_col = col1 if index % 2 == 0 else col2

        with target_col:
            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"input_{feature}"
            )

    st.divider()

    if st.button("Predict Wellbeing", type="primary",
                 use_container_width=True):

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_cols
        )

        # Same preprocessing used during model training
        processed = preprocessor.transform(input_df)

        processed_df = pd.DataFrame(
            processed,
            columns=feature_cols
        )

        # Same feature engineering used during training
        final_input = engineer_features(processed_df)

        # Model prediction
        prediction = int(model.predict(final_input)[0])
        predicted_label = LABELS[prediction]

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Wellbeing Category: {predicted_label}"
        )

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(final_input)[0]

            probability_df = pd.DataFrame({
                "Category": [
                    LABELS[int(class_id)]
                    for class_id in model.classes_
                ],
                "Probability (%)": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                "Probability (%)",
                ascending=False
            )

            st.subheader("Class Probability Distribution")

            chart = (
                alt.Chart(probability_df)
                .mark_bar()
                .encode(
                    x=alt.X("Category:N", sort=None, title="Wellbeing Category"),
                    y=alt.Y(
                        "Probability (%):Q",
                        scale=alt.Scale(domain=[0, 100]),
                        title="Probability (%)"
                    ),
                    tooltip=[
                        "Category",
                        alt.Tooltip("Probability (%):Q", format=".2f")
                    ]
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Probabilities represent the model's estimated "
            "class probabilities for this input."
        )

# ==========================================
# 7. MODEL PERFORMANCE DASHBOARD
# ==========================================

with performance_tab:

    st.subheader("Model Evaluation Results")

    st.write(
        "The following metrics summarize the model's "
        "reported evaluation on the group-based test split "
        "and group cross-validation."
    )

    # Reported group-split evaluation metrics
    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Accuracy",
        "97.32%"
    )

    metric2.metric(
        "Macro-F1 Score",
        "96.73%"
    )

    metric3.metric(
        "Macro ROC-AUC",
        "0.999"
    )

    st.divider()

    st.subheader("Cross-Validation Performance")

    cv1, cv2 = st.columns(2)

    cv1.metric(
        "5-Fold Group CV Macro-F1",
        "96.34%"
    )

    cv2.metric(
        "Standard Deviation",
        "±0.21 percentage points"
    )

    st.divider()

    st.subheader("Understanding the Metrics")

    with st.expander("Accuracy"):
        st.write(
            "The proportion of all test predictions that "
            "were classified correctly."
        )

    with st.expander("Macro-F1 Score"):
        st.write(
            "The unweighted average of the F1 scores across "
            "the five classes. It gives each class equal "
            "importance, regardless of its sample count."
        )

    with st.expander("Macro ROC-AUC"):
        st.write(
            "The macro one-vs-rest ROC-AUC summarizes how "
            "well the model separates each class from the "
            "remaining classes."
        )

    with st.expander("Group Cross-Validation"):
        st.write(
            "Five-fold group cross-validation evaluates "
            "performance across multiple group-based splits. "
            "The mean and standard deviation summarize "
            "the Macro-F1 results."
        )

    st.warning(
        "These are previously reported evaluation metrics. "
        "They are not recalculated from a new test dataset "
        "when this application runs."
    )

# ==========================================
# 8. ABOUT PAGE
# ==========================================

with about_tab:

    st.subheader("About This Research Project")

    st.write(
        "This application uses machine learning to classify "
        "wellbeing into five categories using environmental "
        "and physiological measurements."
    )

    st.markdown("### Input Features")

    st.write(", ".join(feature_cols))

    st.markdown("### Feature Engineering")

    st.write(
        "Five additional features are created from the "
        "original measurements before prediction:"
    )

    st.markdown(
        """
        - Physiological index
        - Environmental index
        - PM1-to-PM10 ratio
        - PM2.5-to-PM10 ratio
        - Physiological-environmental interaction
        """
    )

    st.markdown("### Classification Categories")

    for class_id, label in LABELS.items():
        st.write(f"{class_id}: {label}")

    st.markdown("### Model Information")

    st.write(f"Selected model: {bundle.get('model_name', 'Random Forest')}")
    st.write(f"Imbalance method: {bundle.get('imbalance_method', 'SMOTE')}")
    st.write(f"Number of original features: {len(feature_cols)}")
    st.write("Number of engineered features: 5")
    st.write("Total model input features: 16")

    st.caption(
        "This is an academic research demonstration. "
        "It should not be used for medical decisions."
    )

# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "DigitalExposome Wellbeing AI | Academic Research MVP"
)
