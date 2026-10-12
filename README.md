# Titanic Data Preparation Project

This repository contains the first task of the Machine Learning Internship: **Python, Data Preparation & Machine Learning Fundamentals**.

## Objective
The goal of this project is to build a strong foundation in Python, data handling, data preprocessing, and core machine learning concepts using the Titanic dataset.

## Project Structure
- `data/`: Contains the raw (`titanic.csv`) and cleaned (`titanic_cleaned.csv`) datasets.
- `notebooks/`: Contains the Python script used for data processing (`data_preparation.py`).
- `docs/`: Contains documentation regarding the preprocessing process and the machine learning problem description.

## Tasks Completed
1. [x] Review Python fundamentals for data science.
2. [x] Learn the basics of NumPy and Pandas.
3. [x] Load and inspect a real-world dataset (Titanic).
4. [x] Identify missing values, duplicate records, and inconsistent data.
5. [x] Clean and preprocess the dataset.
6. [x] Implement machine learning concepts:
    - Features and target variables.
    - Training and testing datasets.
    - Classification (Supervised learning).
7. [x] Split the dataset into training and testing sets.
8. [x] Document the complete data preparation process.

## How to Run
To reproduce the data preparation, ensure you have a Python environment with `pandas`, `numpy`, and `scikit-learn` installed. Then run:

```bash
# Create a virtual environment and install dependencies
python3 -m venv venv
./venv/bin/pip install pandas numpy scikit-learn

# Run the preprocessing script
./venv/bin/python3 notebooks/data_preparation.py
```

## Deliverables
- [x] Cleaned dataset (`data/titanic_cleaned.csv`)
- [x] Python script for data preparation (`notebooks/data_preparation.py`)
- [x] Data preprocessing documentation (`docs/preprocessing_documentation.md`)
- [x] Short explanation of the machine learning problem (`docs/problem_explanation.md`)
