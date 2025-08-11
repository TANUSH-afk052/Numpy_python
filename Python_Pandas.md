# Pandas Tutorial

## Introduction to Pandas
Pandas is a powerful Python library for data manipulation and analysis, providing data structures like Series and DataFrame.

```
import pandas as pd

# Create a simple Series
s = pd.Series([1,][2][3][4][5]
print(s)
```

---

## Creating DataFrames
A DataFrame is a two-dimensional tabular data structure with labeled axes (rows and columns).

```
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25,  'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)
print(df)
```

---

## Reading Data from Files
Pandas can read data from various file types such as CSV, Excel, and JSON.

```
import pandas as pd

# Load data from a CSV file
df = pd.read_csv('data.csv')
print(df.head())
```

---

## Exploring Data
Common methods to explore and understand your data include `.head()`, `.info()`, `.describe()`, and `.shape`.

```
print(df.head())         # First 5 rows
print(df.info())         # Summary of DataFrame
print(df.describe())     # Statistical summary
print(df.shape)          # Rows and columns count
```

---

## Selecting Data
Access data by column, row, or both using labels or integer locations with `.loc[]` and `.iloc[]`.

```
# Select column "Age"
ages = df['Age']

# Select rows by labels using loc
row_0 = df.loc

# Select rows by position using iloc
row_first = df.iloc
```

---

## Filtering Data
Filter rows based on conditions.

```
# Get rows where Age > 25
filtered = df[df['Age'] > 25]
print(filtered)
```

---

## Adding and Modifying Columns
Create new columns or modify existing ones.

```
df['Senior'] = df['Age'] > 30
print(df)
```

---

## Handling Missing Data
Detect, remove or fill missing data.

```
# Check for missing values
print(df.isnull())

# Fill missing values
df.fillna(value=0, inplace=True)
```

---

## Grouping and Aggregation
Summarize data using groupby.

```
grouped = df.groupby('City')['Age'].mean()
print(grouped)
```

---

## Sorting Data
Sort data by values or index.

```
sorted_df = df.sort_values(by='Age')
print(sorted_df)
```

---

## Exporting Data
Save DataFrame to file.

```
df.to_csv('output.csv', index=False)
```
# Pandas - Cleaning Data (W3Schools)

I have completed the following **Cleaning Data** modules from W3Schools' Pandas tutorial.  
This section focuses on preparing and fixing datasets before analysis.

## Modules Completed

### 1. Clean Data
- Understanding the importance of data cleaning.
- Identifying messy data and making it ready for analysis.

### 2. Clean Empty Cells
- Handling missing values using:
  - `dropna()` to remove rows with null values.
  - `fillna()` to replace null values with default or computed values.

### 3. Clean Wrong Format
- Converting columns to correct formats (e.g., date, numeric).
- Using `pd.to_datetime()` and `astype()` for conversions.

### 4. Clean Wrong Data
- Identifying and fixing incorrect entries in datasets.
- Replacing or removing invalid values.

### 5. Remove Duplicates
- Using `drop_duplicates()` to remove duplicate rows.
- Understanding when and why duplicates should be removed.

## Summary
Cleaning data ensures better analysis, avoids misleading results, and improves model accuracy.  
These techniques are essential for any data preprocessing workflow.

---
**Source:** [W3Schools Pandas Cleaning Data](https://www.w3schools.com/python/pandas/pandas_cleaning.asp)
