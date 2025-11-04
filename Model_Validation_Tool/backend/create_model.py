import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Create a simple dataset (same structure as your CSV)
data = pd.DataFrame({
    'age': [25, 30, 45, 50, 35],
    'cholesterol': [180, 200, 240, 260, 190],
    'blood_pressure': [120, 130, 140, 150, 125],
    'target': [0, 0, 1, 1, 0]
})

# Separate features and target
X = data[['age', 'cholesterol', 'blood_pressure']]
y = data['target']

# Train a simple model
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'sample_model.pkl')
print("✅ sample_model.pkl created successfully!")
