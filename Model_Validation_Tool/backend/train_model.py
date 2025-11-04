import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Create some simple sample data
data = {
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': [2, 1, 2, 3, 4, 3],
    'label':    [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Split features and labels
X = df[['feature1', 'feature2']]
y = df['label']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {acc * 100:.2f}%")

# Step 6: Save the trained model
joblib.dump(model, "trained_model.pkl")
print("💾 Model saved as trained_model.pkl")

# Step 7: Also save test data for validation later
test_data = X_test.copy()
test_data['label'] = y_test
test_data.to_csv("test_data.csv", index=False)
print("💾 Test data saved as test_data.csv")
