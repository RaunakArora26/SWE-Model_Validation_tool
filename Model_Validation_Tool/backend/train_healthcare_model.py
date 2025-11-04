# train_healthcare_model.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import pandas as pd

# 1️⃣ Load real-world healthcare dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# 2️⃣ Split data into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Healthcare model trained with accuracy: {acc*100:.2f}%")
print("Confusion Matrix:\n", cm)

# 5️⃣ Save model and test data for frontend upload
joblib.dump(model, "healthcare_model.pkl")
pd.DataFrame(X_test, columns=feature_names).assign(label=y_test).to_csv("healthcare_test_data.csv", index=False)

print("💾 Saved healthcare_model.pkl and healthcare_test_data.csv")
