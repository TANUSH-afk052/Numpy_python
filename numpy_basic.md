

---

````markdown
# NumPy Basics Cheat Sheet

## Introduction
NumPy is a popular Python library for numerical computing. It provides support for arrays, matrices, and many mathematical functions.

---

## Importing NumPy

```python
import numpy as np
````

---

## Creating Arrays

```python
# 1D array
arr = np.array([1, 2, 3, 4])

# 2D array
arr2d = np.array([[1, 2], [3, 4]])

# Array of zeros
zeros = np.zeros((3, 3))

# Array of ones
ones = np.ones((2, 4))

# Array with a range of values
range_arr = np.arange(0, 10, 2)  # from 0 to 10 step 2

# Array with evenly spaced values
linspace_arr = np.linspace(0, 1, 5)  # 5 values between 0 and 1
```

---

## Array Attributes

```python
arr.shape      # shape of array
arr.size       # number of elements
arr.dtype      # data type of elements
arr.ndim       # number of dimensions
```

---

## Array Operations

```python
# Arithmetic operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b     # array([5, 7, 9])
a - b     # array([-3, -3, -3])
a * b     # array([4, 10, 18])
a / b     # array([0.25, 0.4, 0.5])

# Universal functions
np.sqrt(a)     # square root
np.exp(a)      # exponentiation
np.sin(a)      # sine
```

---

## Indexing and Slicing

```python
arr = np.array([10, 20, 30, 40, 50])

arr[0]       # 10
arr[1:4]     # array([20, 30, 40])
arr[:3]      # array([10, 20, 30])
arr[-1]      # 50

# 2D indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr2d[0, 2]  # 3
arr2d[:, 1]  # array([2, 5])
```

---

## Reshaping Arrays

```python
arr = np.arange(1, 10)    # array([1, 2, ..., 9])
arr.reshape((3, 3))
```

---

## Random Numbers

```python
# Random float between 0 and 1
np.random.rand()

# Random array of floats
np.random.rand(3, 2)

# Random integers between low (inclusive) and high (exclusive)
np.random.randint(1, 10, size=(3, 3))

# Random samples from a normal (Gaussian) distribution
np.random.randn(3, 3)

# Seed random number generator (for reproducibility)
np.random.seed(42)
```
## Useful Tips

* Use `np.copy()` to make a copy of an array.
* Use `np.concatenate()` to join arrays.
* Use `np.where()` for conditional selection.

---

## References

* [NumPy Documentation](https://numpy.org/doc/)
* [W3Schools NumPy Tutorial](https://www.w3schools.com/python/numpy_intro.asp)

---

*End of Cheat Sheet*

---

---


