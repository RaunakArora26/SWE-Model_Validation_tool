from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import io
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

app = FastAPI()

# Allow Streamlit to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend is running successfully!"}


@app.post("/validate_model/")
async def validate_model(model_file: UploadFile = File(...), data_file: UploadFile = File(...)):
    try:
        # --- Load model ---
        model_bytes = await model_file.read()
        model = joblib.load(io.BytesIO(model_bytes))

        # --- Load dataset ---
        data_bytes = await data_file.read()
        df = pd.read_csv(io.BytesIO(data_bytes))

        # --- Split X and y ---
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # --- Predict ---
        y_pred = model.predict(X)

        # --- Compute metrics ---
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        # --- ROC-AUC (safe for binary only) ---
        try:
            if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
            else:
                roc_auc = np.nan
        except Exception:
            roc_auc = np.nan

        report = classification_report(y, y_pred, output_dict=True)

        return {
            "status": "success",
            "accuracy": round(acc * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": None if np.isnan(roc_auc) else round(roc_auc, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}
