import nbformat as nbf
import pandas as pd

def create_optimization_notebook():
    nb = nbf.v4.new_notebook()

    # Markdown: Title
    nb.cells.append(nbf.v4.new_markdown_cell("# Titanic Survival Prediction: Model Optimization\n\nThis notebook focuses on improving the performance of the selected Random Forest model through cross-validation and hyperparameter tuning."))

    # Code: Imports
    nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
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
%matplotlib inline"""))

    # Markdown: Overfitting/Underfitting
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Understanding Overfitting and Underfitting\n\n**Overfitting** occurs when a model learns the noise in the training data rather than the underlying pattern, leading to high training accuracy but poor generalization to new data (high variance).\n\n**Underfitting** occurs when a model is too simple to capture the underlying structure of the data, resulting in poor performance on both training and testing sets (high bias).\n\nHyperparameter tuning and cross-validation are used to find the 'sweet spot' between these two extremes."))

    # Markdown: Data Loading
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Data Loading\n\nThe training and testing datasets prepared in the previous stage are loaded."))

    # Code: Loading
    nb.cells.append(nbf.v4.new_code_cell("""X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")"""))

    # Markdown: Baseline Model and Cross-Validation
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Baseline Model and Cross-Validation\n\nA baseline Random Forest model is established. Stratified K-Fold cross-validation is applied to assess model stability and estimate generalization performance."))

    # Code: Baseline & CV
    nb.cells.append(nbf.v4.new_code_cell("""# Baseline Random Forest
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
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"""))

    # Markdown: Hyperparameter Tuning
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Hyperparameter Tuning\n\n`GridSearchCV` is used to optimize the hyperparameters of the Random Forest model to improve performance and reduce overfitting."))

    # Code: Grid Search
    nb.cells.append(nbf.v4.new_code_cell("""param_grid = {
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

optimized_rf = grid_search.best_estimator_"""))

    # Markdown: Comparison
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Comparison: Baseline vs. Optimized Model\n\nThe performance of the optimized model is compared against the baseline model on the test set."))

    # Code: Comparison
    nb.cells.append(nbf.v4.new_code_cell("""# Optimized model performance
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
display(comparison_df)

# Save comparison
comparison_df.to_csv('docs/optimization_comparison.csv')"""))

    # Markdown: Feature Importance
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. Feature Importance\n\nVisualizing the most important features used by the optimized Random Forest model."))

    # Code: Feature Importance
    nb.cells.append(nbf.v4.new_code_cell("""importances = optimized_rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance (Optimized Random Forest)')
plt.tight_layout()
plt.savefig('docs/figures/feature_importance.png')
plt.show()"""))

    # Markdown: Final Report
    nb.cells.append(nbf.v4.new_markdown_cell("## 7. Final Model Selection and Summary\n\nBased on the comparison, the optimized model is evaluated for its generalization capability. All decisions, including the selected hyperparameters, are documented here."))

    with open('notebooks/model_optimization.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_optimization_notebook()
