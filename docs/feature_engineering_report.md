# Feature Engineering Report - Titanic Dataset

## Overview
This document outlines the feature engineering process applied to the Titanic dataset to prepare it for machine learning models.

## New Features Created

| Feature | Description | Rationale |
| :--- | :--- | :--- |
| `FamilySize` | `SibSp` + `Parch` + 1 | Captures the total number of family members on board. |
| `IsAlone` | 1 if `FamilySize` == 1, else 0 | Binary indicator of whether a passenger was traveling alone. |
| `AgeBin_Code` | Binned version of `Age` | Converts continuous `Age` into categorical bins (Child, Teen, Adult, etc.) to capture non-linear relationships. |

## Transformations Applied

1.  **Scaling**:
    *   `Age` and `Fare` were scaled using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1. This helps gradient-based models converge faster.
2.  **Categorical Encoding**:
    *   `Sex` and `Embarked` were already pre-processed (One-Hot Encoded/Integer Encoded) in the previous step.
    *   `AgeBin` was converted to numerical codes via `AgeBin_Code`.

## Selected Features for Machine Learning
The following features are included in `data/titanic_features.csv`:
- `Survived` (Target)
- `Pclass`
- `Sex`
- `Age` (Scaled)
- `SibSp`
- `Parch`
- `Fare` (Scaled)
- `Embarked_C`
- `Embarked_Q`
- `Embarked_S`
- `FamilySize`
- `IsAlone`
- `AgeBin_Code`

## Data-Driven Insights (from EDA)
1.  **Gender Priority**: Females (Sex=0) had a significantly higher survival rate than males.
2.  **Socio-economic Status**: Passengers in `Pclass` 1 had a much higher survival probability compared to `Pclass` 2 and 3.
3.  **Fare Skewness**: The `Fare` distribution is highly right-skewed, indicating most passengers paid low fares while a few paid very high fares.
4.  **Age Distribution**: The passenger population was primarily composed of young adults (20-40 range).
5.  **Correlation**: There is a notable correlation between `Pclass` and `Fare`, as expected, and between `Sex` and `Survived`.
