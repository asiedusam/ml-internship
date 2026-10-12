# Machine Learning Problem Explanation

## Problem Statement: Titanic Survival Prediction

The objective of this machine learning task is to predict whether a passenger survived the Titanic shipwreck based on several features such as their age, sex, passenger class, and embarkation point.

### Type of Machine Learning Problem
This is a **Supervised Learning** problem, specifically a **Classification** task.

- **Supervised Learning**: A dataset is used that contains both the input features and the correct output (the labels). The model learns the relationship between the features and the labels.
- **Classification**: The goal is to predict a discrete class label. In this case, the labels are binary:
    - `1`: Survived
    - `0`: Did not survive

### Key Concepts Used

#### 1. Features and Target Variable
- **Target Variable ($y$)**: This is the prediction target. For this problem, it is the `Survived` column.
- **Features ($X$)**: These are the independent variables used to make the prediction. In the processed dataset, these include `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and the one-hot encoded `Embarked` columns.

#### 2. Training and Testing Datasets
To evaluate how well a model generalizes to new, unseen data, the dataset is split into two parts:
- **Training Set**: This subset of data is used to "teach" the model. The algorithm looks for patterns in this data to learn how the features relate to survival.
- **Testing Set**: This subset is held back during training. Once the model is trained, this set is used to make predictions and compare them against the actual known outcomes to measure the model's accuracy.

#### 3. Evaluation
By splitting the data, the model's performance on real-world, unseen data can be estimated, ensuring that it hasn't simply "memorized" the training data (a phenomenon known as overfitting).
