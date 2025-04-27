# Credit Default Risk Prediction Project

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# --- Step 1: Generate or Load Example Dataset ---
# For simplicity, we create a small synthetic dataset
np.random.seed(42)

data = pd.DataFrame({
    'Age': np.random.randint(21, 70, 500),
    'Income': np.random.randint(25000, 120000, 500),
    'LoanAmount': np.random.randint(1000, 40000, 500),
    'CreditScore': np.random.randint(300, 850, 500),
    'Defaulted': np.random.choice([0, 1], size=500, p=[0.85, 0.15])
})


# --- Step 2: Prepare Features and Target ---
X = data[['Age', 'Income', 'LoanAmount', 'CreditScore']]
y = data['Defaulted']


# --- Step 3: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --- Step 4: Train Logistic Regression Model ---
model = LogisticRegression()
model.fit(X_train, y_train)


# --- Step 5: Evaluate Model ---
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- Step 6: Confusion Matrix Visualization ---
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# --- Step 7: Simple Probability Predictions ---
probs = model.predict_proba(X_test)
data_probs = pd.DataFrame(probs, columns=['No Default', 'Default'])

print("\nExample predicted probabilities (first 5 cases):")
print(data_probs.head())
