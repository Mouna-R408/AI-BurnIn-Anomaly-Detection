
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import ollama
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "burnin_random_forest.pkl"
SHAP_PATH = BASE_DIR / "shap_feature_importance.csv"

OLLAMA_MODEL = "llama3.2:3b"

st.set_page_config(
    page_title="Burn-In Intelligence System",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# 2. LOAD THE TRAINED MODEL AND SHAP RESULTS
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_importance():
    return pd.read_csv(SHAP_PATH)


st.title("🔬 AI-Driven Burn-In Screening & Decision Support")

st.caption(
    "Machine-learning anomaly screening, measurement trends, "
    "explainability, and local GenAI technical reporting"
)

st.info(
    "Research prototype using synthetic data. Model predictions "
    "and AI-generated reports are advisory, not certified hardware "
    "acceptance decisions."
)

try:
    model = load_model()
    importance = load_importance()

except Exception as error:
    st.error(f"Could not load the model or SHAP results: {error}")
    st.stop()


# ============================================================
# 3. COMPONENT MEASUREMENT INPUT FORM
# ============================================================

st.header("1. Component Measurements")

with st.form("screening_form"):

    col_id, col_lot = st.columns(2)

    with col_id:
        component_id = st.text_input(
            "Component ID",
            value="DEMO-001",
        )

    with col_lot:
        lot_id = st.text_input(
            "Lot ID",
            value="DEMO-LOT",
        )

    col1, col2 = st.columns(2)

    with col1:
        value_0h = st.number_input(
            "Measurement at 0 hours",
            value=10.0,
            format="%.3f",
        )

        value_24h = st.number_input(
            "Measurement at 24 hours",
            value=10.5,
            format="%.3f",
        )

    with col2:
        value_96h = st.number_input(
            "Measurement at 96 hours",
            value=11.5,
            format="%.3f",
        )

        value_168h = st.number_input(
            "Measurement at 168 hours",
            value=12.5,
            format="%.3f",
        )

    submitted = st.form_submit_button(
        "Analyze Component",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 4. PREDICTION AND SCREENING ANALYSIS
# ============================================================

if submitted:

    if not component_id.strip() or not lot_id.strip():
        st.error("Please enter both Component ID and Lot ID.")

    else:
        features = pd.DataFrame(
            [
                {
                    "Value_0h": value_0h,
                    "Value_24h": value_24h,
                    "Value_96h": value_96h,
                    "Value_168h": value_168h,
                }
            ]
        )

        prediction = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            anomaly_score = float(
                model.predict_proba(features)[0][1]
            )
        else:
            anomaly_score = None

        values = [
            value_0h,
            value_24h,
            value_96h,
            value_168h,
        ]

        change = value_168h - value_0h

        if change > 0:
            trend = "Increasing"
        elif change < 0:
            trend = "Decreasing"
        else:
            trend = "Stable"

        predicted_label = (
            "Anomaly" if prediction == 1 else "Normal"
        )

        # Store the latest analysis for the report assistant.
        st.session_state["burnin_report_context"] = {
            "component_id": component_id.strip(),
            "lot_id": lot_id.strip(),
            "measurements": {
                "0 hours": value_0h,
                "24 hours": value_24h,
                "96 hours": value_96h,
                "168 hours": value_168h,
            },
            "predicted_label": predicted_label,
            "anomaly_score": anomaly_score,
            "trend": trend,
            "change": change,
        }

        # ----------------------------------------------------
        # SCREENING RESULT
        # ----------------------------------------------------

        st.divider()
        st.header("2. Screening Result")

        result_col, score_col, trend_col = st.columns(3)

        with result_col:
            st.metric(
                "Predicted Class",
                predicted_label.upper(),
            )

        with score_col:
            if anomaly_score is not None:
                st.metric(
                    "Model Anomaly Score",
                    f"{anomaly_score:.1%}",
                )
            else:
                st.write("Anomaly score unavailable")

        with trend_col:
            st.metric(
                "Measurement Trend",
                trend,
            )
            st.caption(
                f"Change from 0h to 168h: {change:+.3f}"
            )

        if prediction == 1:
            st.warning(
                "The model flagged this measurement pattern "
                "for further engineering review."
            )
        else:
            st.success(
                "The model classified this measurement pattern "
                "as Normal. This does not guarantee reliability."
            )

        # ----------------------------------------------------
        # MEASUREMENT TREND CHART
        # ----------------------------------------------------

        st.header("3. Measurement Trend")

        time_points = [0, 24, 96, 168]

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(
            time_points,
            values,
            marker="o",
            linewidth=2,
        )

        ax.set_title("Component Measurements During Burn-In")
        ax.set_xlabel("Burn-in Time (hours)")
        ax.set_ylabel("Recorded Measurement")
        ax.set_xticks(time_points)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # GLOBAL SHAP FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.header("4. Explainability: SHAP Feature Importance")

        st.write(
            "This chart shows global feature importance from the "
            "saved model analysis. It describes overall model "
            "behavior, not the individual cause of this prediction."
        )

        required_columns = {
            "Feature",
            "Mean_Absolute_SHAP",
        }

        if required_columns.issubset(importance.columns):

            display_importance = importance.sort_values(
                "Mean_Absolute_SHAP",
                ascending=True,
            )

            fig, ax = plt.subplots(figsize=(9, 4.5))

            ax.barh(
                display_importance["Feature"],
                display_importance["Mean_Absolute_SHAP"],
            )

            ax.set_title("Global SHAP Feature Importance")
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_ylabel("Feature")
            ax.grid(axis="x", alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning(
                "The SHAP CSV does not contain the expected "
                "Feature and Mean_Absolute_SHAP columns."
            )

        # ----------------------------------------------------
        # TECHNICAL INTERPRETATION
        # ----------------------------------------------------

        st.header("5. Technical Interpretation")

        if prediction == 1:
            st.write(
                "The model classified this measurement pattern "
                "as anomalous. Review the measurement progression "
                "and compare it with approved limits for the actual "
                "component and test procedure."
            )
        else:
            st.write(
                "The model classified this measurement pattern "
                "as normal. This prediction does not rule out "
                "failure modes that are absent from the synthetic "
                "training dataset."
            )

        st.write(f"**Observed trend:** {trend}")
        st.write(
            f"**Change over the recorded interval:** {change:+.3f}"
        )

        # ----------------------------------------------------
        # ENGINEERING CHECKS
        # ----------------------------------------------------

        st.header("6. Recommended Engineering Checks")

        recommendations = [
            "Verify the entered values against the original test record.",
            "Confirm measurement-system and sensor calibration.",
            "Compare each reading with approved component-specific limits.",
            "Review the complete burn-in history and repeat tests if required.",
            "Escalate deviations according to the applicable engineering procedure.",
        ]

        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")

        st.caption(
            "These are general verification steps, not a confirmed "
            "physical root cause. Measurement units and validated "
            "engineering failure thresholds are not specified by "
            "the synthetic dataset."
        )

        # ----------------------------------------------------
        # CSV SCREENING REPORT
        # ----------------------------------------------------

        st.header("7. Export Screening Report")

        report = pd.DataFrame(
            [
                {
                    "Component_ID": component_id.strip(),
                    "Lot_ID": lot_id.strip(),
                    "Value_0h": value_0h,
                    "Value_24h": value_24h,
                    "Value_96h": value_96h,
                    "Value_168h": value_168h,
                    "Measurement_Change": change,
                    "Measurement_Trend": trend,
                    "Predicted_Label": predicted_label,
                    "Model_Anomaly_Score": anomaly_score,
                }
            ]
        )

        st.download_button(
            "Download Screening Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name=f"{component_id.strip()}_screening_report.csv",
            mime="text/csv",
        )


# ============================================================
# 5. LOCAL GENAI TECHNICAL REPORT ASSISTANT
# ============================================================

st.divider()
st.header("8. GenAI Technical Report Assistant")

st.write(
    "Generate an engineering-oriented report using the local "
    "Llama language model and the latest component analysis. "
    "No OpenAI API credits are required."
)

report_context = st.session_state.get("burnin_report_context")

if report_context is None:

    st.info(
        "Enter component measurements and click Analyze Component "
        "before generating an AI technical report."
    )

else:

    st.write(
        f"**Current component:** "
        f"{report_context['component_id']}"
    )

    st.write(
        f"**Current model result:** "
        f"{report_context['predicted_label']}"
    )

    if st.button(
        "Generate AI Technical Report",
        type="primary",
        use_container_width=True,
    ):

        prompt = f"""
You are a technical report assistant for a research prototype
that screens electronic components using synthetic burn-in data.

Use only the following supplied component analysis evidence:

{report_context}

Write a professional, concise technical report using these headings:

1. Screening Summary
2. Measurement Trend
3. Interpretation of the Model Prediction
4. Recommended Engineering Checks
5. Limitations and Conclusion

Requirements:
- Refer to the supplied component and lot identifiers.
- Describe the four recorded measurement values accurately.
- Explain the direction and amount of change without inventing units.
- Explain the prediction as a machine-learning classification.
- Do not claim that the model has established a physical root cause.
- Do not invent component specifications, failure mechanisms, or limits.
- Do not describe the anomaly score as a calibrated failure probability.
- Do not claim that the component is definitely safe or defective.
- Recommend checking original test records and approved engineering limits.
- State that the dataset is synthetic and the result is advisory.
- Use clear, professional engineering language.
"""

        with st.spinner(
            "Generating the report with your local AI model..."
        ):

            try:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                report_text = response["message"]["content"]

                st.subheader("Generated AI Technical Report")
                st.markdown(report_text)

                st.download_button(
                    "Download AI Technical Report (TXT)",
                    data=report_text,
                    file_name=(
                        f"{report_context['component_id']}"
                        "_AI_Technical_Report.txt"
                    ),
                    mime="text/plain",
                )

            except Exception as error:

                st.error(
                    "The local AI report could not be generated. "
                    "Check that Ollama is running and that the "
                    "llama3.2:3b model is installed."
                )

                st.code(str(error))


# ============================================================
# 6. FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype using synthetic data. Validate with real, "
    "appropriately labelled hardware data before operational use."
)

