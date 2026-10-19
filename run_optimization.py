import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import os

# Set style
sns.set_theme(style="whitegrid")

# 1. Data Loading
print("Loading data...")
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 2. Baseline Model and Cross-Validation
print("\nEstablishing baseline model and performing cross-validation...")
baseline_rf = RandomForestClassifier(random_state=42)
baseline_rf.fit(X_train, y_train)

# Baseline performance on test set
y_pred_baseline = baseline_rf.predict(X_test)
y_prob_baseline = baseline_rf.predict_proba(X_test)[:, 1]

baseline_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_baseline),
    "Precision": precision_score(y_test, y_pred_baseline),
    "Recall": recall_score(y_test, y_pred_baseline),
    "F1-Score": f1_score(y_test, y_pred_baseline),
    "ROC-AUC": roc_auc_score(y_test, y_prob_baseline)
}
print("Baseline Test Metrics:")
for k, v in baseline_metrics.items():
    print(f"{k}: {v:.4f}")

# Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(baseline_rf, X_train, y_train, cv=skf, scoring='f1')

print(f"\\nCross-Validation F1-Scores: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 3. Hyperparameter Tuning
print("\nStarting hyperparameter tuning with GridSearchCV...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=skf,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

optimized_rf = grid_search.best_estimator_

# 4. Comparison
print("\nComparing Baseline vs. Optimized Model...")
y_pred_opt = optimized_rf.predict(X_test)
y_prob_opt = optimized_rf.predict_proba(X_test)[:, 1]

optimized_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_opt),
    "Precision": precision_score(y_test, y_pred_opt),
    "Recall": recall_score(y_test, y_pred_opt),
    "F1-Score": f1_score(y_test, y_pred_opt),
    "ROC-AUC": roc_auc_score(y_test, y_prob_opt)
}
print("Optimized Test Metrics:")
for k, v in optimized_metrics.items():
    print(f"{k}: {v:.4f}")

# Create comparison DataFrame
comparison_df = pd.DataFrame([baseline_metrics, optimized_metrics], index=['Baseline', 'Optimized'])
print("\nComparison Table:")
print(comparison_df)

# Save comparison
os.makedirs('docs', exist_ok=True)
comparison_df.to_csv('docs/optimization_comparison.csv')

# 5. Feature Importance
print("\nCalculating feature importance...")
importances = optimized_rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance (Optimized Random Forest)')
plt.tight_layout()
os.makedirs('docs/figures', exist_ok=True)
plt.savefig('docs/figures/feature_importance.png')
print("Feature importance plot saved to docs/figures/feature_importance.png")

print("\nOptimization completed successfully.")
