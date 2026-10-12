import pandas as pd

df = pd.read_csv('data/titanic_cleaned.csv')
print("--- Head ---")
print(df.head())
print("\n--- Info ---")
print(df.info())
print("\n--- Describe ---")
print(df.describe())
print("\n--- Columns ---")
print(df.columns.tolist())
