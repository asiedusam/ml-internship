# Data Preprocessing Documentation

## Project Overview
This document details the data preprocessing steps taken for the Titanic dataset to prepare it for machine learning models. The goal was to transform the raw data into a clean, numerical format suitable for algorithms.

## Dataset Description
The Titanic dataset contains information about passengers on the Titanic, including their survival status, age, gender, class, etc.

## Preprocessing Steps

### 1. Exploratory Data Analysis (EDA)
Before cleaning, the dataset was inspected to identify:
- **Missing Values**: Found missing values in `Age`, `Cabin`, and `Embarked`.
- **Duplicates**: Checked for any duplicate passenger records (none found).
- **Data Types**: Identified categorical (Name, Sex, Ticket, Cabin, Embarked) and numerical (Age, Fare, Pclass, etc.) columns.

### 2. Handling Missing Values
To ensure the dataset was complete, the following imputation strategies were used:
- **Age**: Imputed missing values with the **median** age to avoid the influence of outliers.
- **Embarked**: Imputed missing values with the **mode** (the most frequent value, 'S').
- **Cabin**: Since a large portion was missing, missing values were filled with the string **'Unknown'** to retain the information that the cabin was not recorded.
- **Fare**: Imputed missing values with the **median** fare.

### 3. Feature Selection and Engineering
To simplify the model and focus on predictive features, the following steps were performed:
- **Removed Unnecessary Columns**: `PassengerId`, `Name`, `Ticket`, and `Cabin` were dropped as they are unique identifiers or contain too much unstructured text for this basic task.
- **Categorical Encoding**:
    - **Sex**: Converted from text ('male', 'female') to numerical values (0 for male, 1 for female).
    - **Embarked**: Used **One-Hot Encoding** to convert the categorical 'Embarked' column into three separate binary columns: `Embarked_C`, `Embarked_Q`, and `Embarked_S`.

### 4. Data Splitting
The cleaned dataset was split into:
- **Features (X)**: The input variables used for prediction.
- **Target (y)**: The variable to be predicted (`Survived`).

The data was then split into **Training** and **Testing** sets:
- **Training Set**: 80% of the data, used to train the machine learning model.
- **Testing Set**: 20% of the data, used to evaluate the performance of the model on unseen data.

## Final Cleaned Dataset
The resulting cleaned dataset is saved as `data/titanic_cleaned.csv`.
