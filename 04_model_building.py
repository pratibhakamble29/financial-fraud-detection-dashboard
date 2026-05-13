import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

print("Loading engineered dataset...")

# Load dataset
df = pd.read_csv('../outputs/fraud_engineered.csv')

print("Dataset loaded!")
print(df.shape)

# -----------------------------------
# Drop unnecessary columns
# -----------------------------------

drop_cols = ['TransactionID']

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# -----------------------------------
# Encode categorical columns
# -----------------------------------

label_encoder = LabelEncoder()

cat_cols = df.select_dtypes(
    include=['object']
).columns

for col in cat_cols:
    df[col] = label_encoder.fit_transform(
        df[col].astype(str)
    )

print("Categorical columns encoded!")

# -----------------------------------
# Define Features and Target
# -----------------------------------

X = df.drop('isFraud', axis=1)

y = df['isFraud']

# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split completed!")

# -----------------------------------
# Handle Class Imbalance using SMOTE
# -----------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE applied!")
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_smote.value_counts())

# -----------------------------------
# Logistic Regression Model
# -----------------------------------

log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(
    X_train_smote,
    y_train_smote
)

y_pred_log = log_model.predict(X_test)

y_prob_log = log_model.predict_proba(X_test)[:,1]

print("\n===== Logistic Regression =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("Precision:",
      precision_score(y_test, y_pred_log))

print("Recall:",
      recall_score(y_test, y_pred_log))

print("F1 Score:",
      f1_score(y_test, y_pred_log))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# -----------------------------------
# Random Forest Model
# -----------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

y_pred_rf = rf_model.predict(X_test)

y_prob_rf = rf_model.predict_proba(X_test)[:,1]

print("\n===== Random Forest =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("Precision:",
      precision_score(y_test, y_pred_rf))

print("Recall:",
      recall_score(y_test, y_pred_rf))

print("F1 Score:",
      f1_score(y_test, y_pred_rf))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# -----------------------------------
# Feature Importance
# -----------------------------------

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

print("\nModel building completed successfully!")