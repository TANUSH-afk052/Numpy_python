

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
***


# NumPy Random

## Random Intro
NumPy provides random number generation via `numpy.random`, for use in simulations and analysis.

```
import numpy as np

# Generate a random integer between 0 and 9
print(np.random.randint(0, 10))
```

---

## Data Distribution
You can easily model statistical distributions and visualize them using NumPy.

```
import numpy as np
data = np.random.choice([1,ize=(10,))[1]
print(data)
```

---

## Random Permutation
Shuffle or permute elements using `np.random.permutation`.

```
import numpy as np
arr = np.array([1, 2,huffled = np.random.permutation(arr)
print(shuffled)
```

---

## Seaborn Module
Seaborn is a data visualization library that works well with random distributions.

```
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, 1000)
sns.histplot(data, kde=True)
plt.show()
```

---

## Normal Distribution
Generate numbers from a standard normal (Gaussian) distribution.

```
import numpy as np
data = np.random.normal(loc=0, scale=1, size=10)
print(data)
```

---

## Binomial Distribution
Model number of successes in n trials with probability p.

```
import numpy as np
data = np.random.binomial(n=10, p=0.5, size=10)
print(data)
```

---

## Poisson Distribution
Number of events occurring in a fixed interval.

```
import numpy as np
data = np.random.poisson(lam=3, size=10)
print(data)
```

---

## Uniform Distribution
All values have equal probability.

```
import numpy as np
data = np.random.uniform(low=0.0, high=1.0, size=10)
print(data)
```

---

## Logistic Distribution
Similar to normal, often used in classification.

```
import numpy as np
data = np.random.logistic(loc=0.0, scale=1.0, size=10)
print(data)
```

---

## Multinomial Distribution
More than two outcomes per trial.

```
import numpy as np
# 6 trials, probabilities for each event
data = np.random.multinomial(6, [1/6.]*6, size=10)
print(data)
```

---

## Exponential Distribution
Time between events in Poisson process.

```
import numpy as np
data = np.random.exponential(scale=2.0, size=10)
print(data)
```

---

## Chi Square Distribution
Used in hypothesis testing.

```
import numpy as np
data = np.random.chisquare(df=2, size=10)
print(data)
```

---

## Rayleigh Distribution
Often used in signal processing.

```
import numpy as np
data = np.random.rayleigh(scale=1.0, size=10)
print(data)
```

---

## Pareto Distribution
Used to model wealth distribution.

```
import numpy as np
data = np.random.pareto(a=2.0, size=10)
print(data)
```

---

## Zipf Distribution
Frequency of events follows power law.

```
import numpy as np
data = np.random.zipf(a=2.0, size=10)
print(data)
```

---
##Platforms Used
W3 School
---

