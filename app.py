import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import altair as alt

# ==========================================
# 1. PAGE CONFIGURATION & MINIMALIST STYLING
# ==========================================

st.set_page_config(
    page_title="DigitalExposome | Wellbeing AI",
    page_icon="🌿",
    layout="centered"
)

st.markdown(
    """
    <style>
        /* Clean top padding and typography */
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            max-width: 820px;
        }
        /* Subtle badge styling for model metadata */
        .meta-badge {
            display: inline-block;
            background-color: #F1F5F9;
            color: #475569;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
            margin-right: 6px;
            margin-bottom: 8px;
        }
        /* Subtle disclaimer note */
        .research-note {
            border-left: 3px solid #10B981;
            background-color: #F8FAFC;
            padding: 10px 14px;
            font-size: 0.88rem;
            color: #475569;
            border-radius: 0 6px 6px 0;
            margin: 0.75rem 0 1.25rem 0;
        }
        /* Metric card refinement */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 14px 16px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
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

PHYSIO_FEATURES = ["IBI", "HR", "BVP", "EDA"]

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
    X["physio_index"] = X[["IBI", "HR", "BVP", "EDA"]].mean(axis=1)
    X["env_index"] = X[["NO2", "NH3", "CO", "Noise"]].mean(axis=1)
    X["pm_ratio_1_10"] = X["PM1"] / (X["PM10"] + 1e-6)
    X["pm_ratio_25_10"] = X["PM25"] / (X["PM10"] + 1e-6)
    X["physio_env_interaction"] = X["physio_index"] * X["env_index"]
    return X

# ==========================================
# 4. MINIMALIST HEADER
# ==========================================

st.markdown("## 🌿 DigitalExposome Wellbeing AI")
st.markdown(
    "Five-class wellbeing classification using environmental and physiological sensor measurements."
)

model_name = bundle.get("model_name", "Random Forest")
imbalance_method = bundle.get("imbalance_method", "SMOTE")

st.markdown(
    f"""
    <div>
        <span class="meta-badge">Model: {model_name}</span>
        <span class="meta-badge">Sampling: {imbalance_method}</span>
        <span class="meta-badge">IEEE ICES&T 2026 Research</span>
    </div>
    <div class="research-note">
        <strong>Research Prototype:</strong> Outputs represent statistical model estimates and are not intended for clinical or medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 5. NAVIGATION TABS
# ==========================================

prediction_tab, performance_tab, about_tab = st.tabs(
    ["🔍 Predict", "📊 Performance", "ℹ️ Methodology"]
)

# ==========================================
# 6. PREDICTION TAB
# ==========================================

with prediction_tab:

    physio_cols = [f for f in feature_cols if f in PHYSIO_FEATURES]
    env_cols = [f for f in feature_cols if f not in PHYSIO_FEATURES]

    input_values = {}

    with st.container(border=True):
        st.markdown("##### Physiological Signals")
        p_cols = st.columns(2)
        for idx, feature in enumerate(physio_cols):
            with p_cols[idx % 2]:
                input_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"input_{feature}"
                )

    with st.container(border=True):
        st.markdown("##### Environmental & Air Quality")
        e_cols = st.columns(2)
        for idx, feature in enumerate(env_cols):
            with e_cols[idx % 2]:
                input_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"input_{feature}"
                )

    st.write("")
    predict_clicked = st.button(
        "Run Wellbeing Classification",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:
        input_df = pd.DataFrame([input_values], columns=feature_cols)
        processed = preprocessor.transform(input_df)
        processed_df = pd.DataFrame(processed, columns=feature_cols)
        final_input = engineer_features(processed_df)

        prediction = int(model.predict(final_input)[0])
        predicted_label = LABELS[prediction]

        st.write("")
        with st.container(border=True):
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                st.caption("PREDICTED WELLBEING STATE")
                st.markdown(f"### {predicted_label}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(final_input)[0]
                probability_df = pd.DataFrame({
                    "Category": [LABELS[int(cid)] for cid in model.classes_],
                    "Probability (%)": probabilities * 100
                }).sort_values("Probability (%)", ascending=False)

                top_conf = probability_df.iloc[0]["Probability (%)"]
                probability_df["Highlight"] = [
                    "Top" if i == 0 else "Other"
                    for i in range(len(probability_df))
                ]

                with res_col2:
                    st.caption("MODEL CONFIDENCE")
                    st.markdown(f"### {top_conf:.1f}%")

                st.divider()
                st.caption("CLASS PROBABILITY DISTRIBUTION")

                bars = (
                    alt.Chart(probability_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=42)
                    .encode(
                        x=alt.X(
                            "Category:N",
                            sort=None,
                            title=None,
                            axis=alt.Axis(labelAngle=0, labelFontSize=12)
                        ),
                        y=alt.Y(
                            "Probability (%):Q",
                            scale=alt.Scale(domain=[0, 100]),
                            title="Probability (%)",
                            axis=alt.Axis(gridColor="#F1F5F9")
                        ),
                        color=alt.Color(
                            "Highlight:N",
                            scale=alt.Scale(
                                domain=["Top", "Other"],
                                range=["#10B981", "#CBD5E1"]
                            ),
                            legend=None
                        ),
                        tooltip=[
                            "Category",
                            alt.Tooltip("Probability (%):Q", format=".2f")
                        ]
                    )
                )

                labels = bars.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-5,
                    fontSize=11,
                    color="#334155"
                ).encode(
                    text=alt.Text("Probability (%):Q", format=".1f")
                )

                chart = (
                    (bars + labels)
                    .properties(height=270)
                    .configure_view(strokeWidth=0)
                )

                st.altair_chart(chart, use_container_width=True)

                with st.expander("View exact probability table"):
                    st.dataframe(
                        probability_df[["Category", "Probability (%)"]].style.format(
                            {"Probability (%)": "{:.2f}%"}
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

# ==========================================
# 7. PERFORMANCE TAB
# ==========================================

with performance_tab:

    st.markdown("##### Held-Out Group Test Split")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "97.32%")
    m2.metric("Macro-F1", "96.73%")
    m3.metric("Macro ROC-AUC", "0.999")

    st.write("")
    st.markdown("##### 5-Fold Group Cross-Validation")
    cv1, cv2 = st.columns(2)
    cv1.metric("Mean Macro-F1", "96.34%")
    cv2.metric("Std. Deviation", "±0.21%")

    st.write("")
    with st.container(border=True):
        st.markdown("##### Metric Definitions")
        st.markdown(
            """
            - **Accuracy:** Overall proportion of correctly classified test instances.
            - **Macro-F1 Score:** Unweighted mean F1 across all five classes, giving equal weight to minority and majority wellbeing states.
            - **Macro ROC-AUC:** One-vs-rest area under the ROC curve averaged across all five classes.
            - **Group Cross-Validation:** Leakage-aware evaluation ensuring subject/group independence across folds.
            """
        )

    st.caption(
        "Note: Metrics reflect offline validation on the DigitalExposome benchmark split."
    )

# ==========================================
# 8. METHODOLOGY TAB
# ==========================================

with about_tab:

    with st.container(border=True):
        st.markdown("##### Pipeline Architecture")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Model Specifications**")
            st.markdown(
                f"""
                - **Classifier:** {model_name}
                - **Class Balancing:** {imbalance_method}
                - **Raw Features:** {len(feature_cols)} sensors
                - **Engineered Features:** 5 domain indices
                - **Total Input Space:** 16 features
                """
            )
        with col_b:
            st.markdown("**Target Classes**")
            for cid, label in LABELS.items():
                st.markdown(f"- `{cid}` — {label}")

    with st.container(border=True):
        st.markdown("##### Engineered Domain Features")
        st.markdown(
            """
            1. **Physiological Index** — Mean normalized response across `IBI`, `HR`, `BVP`, and `EDA`.
            2. **Environmental Index** — Mean exposure across `NO2`, `NH3`, `CO`, and `Noise`.
            3. **PM1 / PM10 Ratio** — Fine-to-coarse particulate matter ratio.
            4. **PM2.5 / PM10 Ratio** — Respirable-to-coarse particulate matter ratio.
            5. **Physio-Environmental Interaction** — Multiplicative cross-term capturing combined autonomic and ambient load.
            """
        )

# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("---")
st.caption("DigitalExposome Wellbeing AI · Leakage-Aware Multi-Class Research Prototype")
