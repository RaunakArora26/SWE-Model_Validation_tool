import streamlit as st
import requests
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="AI Model Validation Tool", page_icon="🩺", layout="wide")

st.title("🩺 AI Model Validation Tool for Healthcare Diagnostics")
st.write("Upload your trained AI model and test dataset to evaluate its performance on real-world healthcare data.")

# File uploaders
model_file = st.file_uploader("📦 Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])
data_file = st.file_uploader("📄 Upload test dataset (.csv)", type=["csv"])

# Run Validation button
if st.button("🚀 Run Validation"):
    if not model_file or not data_file:
        st.warning("⚠️ Please upload both model and dataset files.")
    else:
        with st.spinner("⏳ Validating your model... please wait..."):
            try:
                backend_url = "http://127.0.0.1:8000/validate_model/"
                files = {
                    "model_file": (model_file.name, model_file, "application/octet-stream"),
                    "data_file": (data_file.name, data_file, "text/csv")
                }

                response = requests.post(backend_url, files=files)
                result = response.json()

                if result.get("status") == "success":
                    st.success("✅ Validation Successful!")

                    # Display Key Metrics
                    st.subheader("📊 Model Performance Metrics")
                    metrics = {
                        "Accuracy": round(result["accuracy"] * 100, 2),
                        "Precision": round(result["precision"] * 100, 2),
                        "Recall": round(result["recall"] * 100, 2),
                        "F1 Score": round(result["f1_score"] * 100, 2),
                        "ROC-AUC": (
                            round(result["roc_auc"], 3)
                            if result["roc_auc"] and not pd.isna(result["roc_auc"])
                            else "N/A"
                        ),
                    }
                    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).set_index("Metric"))

                    # Confusion Matrix
                    st.subheader("🔢 Confusion Matrix")
                    cm = pd.DataFrame(result["confusion_matrix"])
                    st.dataframe(cm.style.highlight_max(axis=1, color="#b3e6b3"))

                    # Classification Report
                    st.subheader("🧾 Classification Report")
                    report_df = pd.DataFrame(result["classification_report"]).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

                else:
                    st.error(f"❌ Validation Failed: {result.get('error', 'Unknown error occurred')}")

            except Exception as e:
                st.error(f"💥 Error connecting to backend: {e}")

# Footer
st.markdown("---")
st.markdown("💡 *Developed for healthcare model performance validation — supports accuracy, precision, recall, F1, ROC-AUC & more!*")
