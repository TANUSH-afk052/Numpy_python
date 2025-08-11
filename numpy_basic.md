
# 🐍 NumPy Notes (W3Schools)

[![NumPy](https://img.shields.io/badge/Library-NumPy-blue?logo=python)](https://numpy.org/)
[![Level](https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green)]()
[![Source](https://img.shields.io/badge/Source-W3Schools-orange)](https://www.w3schools.com/python/numpy_intro.asp)

This document contains my completed **NumPy** modules from W3Schools,  
including **Basics**, **Random**, and **ufunc (Universal Functions)**.

---

## 📌 ![Basics Badge](https://img.shields.io/badge/Section-Basics-blue) NumPy Basics Cheat Sheet

### 🔹 Introduction
NumPy is a popular Python library for numerical computing. It provides support for arrays, matrices, and many mathematical functions.

---

### 📥 Importing NumPy
```python
import numpy as np
````

---

### 🛠 Creating Arrays

```python
arr = np.array([1, 2, 3, 4])         # 1D array
arr2d = np.array([[1, 2], [3, 4]])   # 2D array
zeros = np.zeros((3, 3))             # array of zeros
ones = np.ones((2, 4))               # array of ones
range_arr = np.arange(0, 10, 2)      # range array
linspace_arr = np.linspace(0, 1, 5)  # evenly spaced values
```

---

### 📊 Array Attributes

```python
arr.shape   # shape of array
arr.size    # number of elements
arr.dtype   # data type
arr.ndim    # number of dimensions
```

---

### ➕➖✖️➗ Array Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b
a - b
a * b
a / b

np.sqrt(a)
np.exp(a)
np.sin(a)
```

---

### 🔍 Indexing and Slicing

```python
arr[0]       # first element
arr[1:4]     # slice from index 1 to 3
arr2d[0, 2]  # 3rd element in 1st row
arr2d[:, 1]  # 2nd column from all rows
```

---

### 🔄 Reshaping Arrays

```python
arr = np.arange(1, 10)
arr.reshape((3, 3))
```

---

### 🎲 Random Numbers

```python
np.random.rand()
np.random.randint(1, 10, size=(3, 3))
np.random.randn(3, 3)
np.random.seed(42)
```

---

## 🎲 ![Random Badge](https://img.shields.io/badge/Section-Random-purple) NumPy Random

### 🎯 Random Intro

```python
np.random.randint(0, 10)
```

---

### 📈 Data Distribution

```python
np.random.choice([1, 2, 3], size=(10,))
```

---

### 🔄 Random Permutation

```python
np.random.permutation(np.array([1, 2, 3]))
```

---

### 📊 Seaborn Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt
data = np.random.normal(0, 1, 1000)
sns.histplot(data, kde=True)
plt.show()
```

---

### 📉 Distributions

* **Normal:** `np.random.normal(0, 1, 10)`
* **Binomial:** `np.random.binomial(10, 0.5, 10)`
* **Poisson:** `np.random.poisson(3, 10)`
* **Uniform:** `np.random.uniform(0.0, 1.0, 10)`
* **Logistic:** `np.random.logistic(0.0, 1.0, 10)`
* **Multinomial:** `np.random.multinomial(6, [1/6.]*6, 10)`
* **Exponential:** `np.random.exponential(2.0, 10)`
* **Chi-Square:** `np.random.chisquare(2, 10)`
* **Rayleigh:** `np.random.rayleigh(1.0, 10)`
* **Pareto:** `np.random.pareto(2.0, 10)`
* **Zipf:** `np.random.zipf(2.0, 10)`

---

## ⚡ ![Ufunc Badge](https://img.shields.io/badge/Section-ufunc-red) NumPy ufunc (Universal Functions)

### 📜 ufunc Intro

* Fast, element-wise operations on arrays.
* Better performance than Python loops.

---

### 🛠 ufunc Create Function

```python
def myadd(x, y):
    return x + y
myadd = np.frompyfunc(myadd, 2, 1)
```

---

### ➗ ufunc Simple Arithmetic

```python
np.add()
np.subtract()
np.multiply()
np.divide()
```

---

### 🔢 ufunc Rounding Decimals

```python
np.trunc()
np.fix()
np.around()
np.floor()
np.ceil()
```

---

### 📈 ufunc Logs

```python
np.log2()
np.log10()
np.log()
```

---

### ➕ ufunc Summations

```python
np.sum()
np.cumsum()
```

---

### ✖️ ufunc Products

```python
np.prod()
np.cumprod()
```

---


