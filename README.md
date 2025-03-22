# Intro

<img width="554" alt="Screenshot 2025-03-21 at 4 59 17 p m" src="https://github.com/user-attachments/assets/305b00e2-415e-4904-9893-d5a79c3b1f94" />

<img width="555" alt="Screenshot 2025-03-21 at 5 04 33 p m" src="https://github.com/user-attachments/assets/4fbfb89f-0ea4-4abb-8a92-2a970dc63db6" />

<img width="710" alt="Screenshot 2025-03-21 at 5 05 57 p m" src="https://github.com/user-attachments/assets/39de8529-62f5-4622-aa04-3abb30396b99" />

<img width="659" alt="Screenshot 2025-03-21 at 5 07 22 p m" src="https://github.com/user-attachments/assets/40a03d92-dc26-45ed-b3c3-063d4d238b2e" />

<img width="829" alt="Screenshot 2025-03-21 at 5 14 05 p m" src="https://github.com/user-attachments/assets/fa555ee9-e78e-4f26-add5-1538833972c7" />

<img width="738" alt="Screenshot 2025-03-21 at 5 17 18 p m" src="https://github.com/user-attachments/assets/1e8cf9f3-245a-48a1-8a17-7fa8365a0165" />

## Getting started with Python

- https://colab.google/
- New Notebook

- Anaconda
- VSCode

## Complete Python

```
conda create -p venv python==3.12
```

```
conda activate venv/
```

For running Jupyter Notebooks:

```
pip install ipykernel
```

### Lambda Functions

Anonymous Function, any number of arguments but only one expression

```python
#Basic Function
def addition(a,b):
  return a+b

addition(2,3)

#Lambda Function
adition = lambda a,b:a+b
type(addition)
addition(5,6)
```
```python
def even(num):
  if even%2==0:
    return True

even(24)
```

```python
even1 =  num:num%2==0
even1(12)
```

```python
def adition(x,y,z):
  return x+y+z

addition(12,13,14)

addition1=lambda x,y,z:x+y+z
addition1(12,13,14)
```

```python
## map() - applies a function to all items in a list
numbers = [1,2,3,4,5,6]
def square(number):
  return number**2

square(2)

list(map(lambda x:x**2, numbers))
```


### Map Functions

Applies funtion to all items in an input list or iterable and return a map object (an iterator)

```python
def square(x):
  return x*x

square(x)

numbers=[1,2,3,4,5,6,7,8]

list(map(square, numbers))
```

```python
## Lambda function with map
numbers=[1,2,3,4,5,6,7,8]
list(map(lambda x:x*x, numbers))
```

```python
## Map Multiple iterables
numbers1 = [1,2,3]
numbers2 = [4,5,6]

added_numbers=list(map(lambda x,y:x+y, numbers1, numbers2))
print(added_numbers)
```

```python
# map() to convert a list of strings to integers
str_numbers = ['1', '2', '3', '4', '5']
int_numbers = list(map(int, str_numbers))

print(int_numbers)
```

```python
words=['apple', 'banana', 'cherry']
upper_words=list(map(str.upper, words))
print(upper_words)
```

```python
def get_name(person):
  return person['name']

people = [
  {'name':'Krish', 'age':32},
  {'name':'Jack', 'age':33},
]
list(map(get_name, people))
```

### Filter Function in Python

Constructs an iterator from elements of an iterable for which a function return true

```python
def even(num):
  if num%2==0:
    return True

even(24)

lst=[1,2,3,4,5,6,7,8,9,10,11,12]

even_numbers = list(filter(even, lst))
```

```python
## filter with lambda function
numbers = [1,2,3,4,5,6,7,8,9]
greater_than_five=list(filter(lambda x:x>5, numbers))
print(greater_than_five)
```

```python
## filter with lambda function and multiple conditions
numbers = [1,2,3,4,5,6,7,8,9]
even_and_greater_than_five=list(filter(lambda x:x>5 and x%2==0, numbers))
print(even_and_greater_than_five)
```

```python
## filter to check if the age is greater than 25 in dictionaries
people = [
  {'name':'Krish', 'age':32},
  {'name':'Jack', 'age':33},
  {'name':'John', 'age':25},
]

def age_greater_than_25(person)
  return person['age']>25

list(filter(age_greater_than_25, people))
```

## Data Analysis

### Numpy

### NumPy Basics

NumPy is a fundamental library for scientific computing in Python. It provides support for arrays and matrices, along with a collection of mathematical functions to operate on these data structures. In this lesson, we will cover the basics of NumPy, focusing on arrays and vectorized operations.

### Installation

```python
!pip install numpy
```

### Creating Arrays

#### Basic Array Creation
```python
import numpy as np

# Create a 1D array
arr1 = np.array([1,2,3,4,5])
print(arr1)
print(type(arr1))
print(arr1.shape)
```
Output:
```
[1 2 3 4 5]
<class 'numpy.ndarray'>
(5,)
```

#### Reshaping Arrays
```python
# 1D array reshaped to 2D
arr2 = np.array([1,2,3,4,5])
arr2.reshape(1,5)  # 1 row and 5 columns
```
Output:
```
array([[1, 2, 3, 4, 5]])
```

#### Shape Verification
```python
arr2 = np.array([[1,2,3,4,5]])
arr2.shape
```
Output:
```
(1, 5)
```

#### 2D Arrays
```python
# Create a 2D array
arr2 = np.array([[1,2,3,4,5],[2,3,4,5,6]])
print(arr2)
print(arr2.shape)
```
Output:
```
[[1 2 3 4 5]
 [2 3 4 5 6]]
(2, 5)
```

#### Array Creation Functions

##### Using arange
```python
np.arange(0,10,2).reshape(5,1)
```
Output:
```
array([[0],
       [2],
       [4],
       [6],
       [8]])
```

##### Creating Arrays of Ones
```python
np.ones((3,4))
```
Output:
```
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
```

##### Identity Matrix
```python
# Create an identity matrix
np.eye(3)
```
Output:
```
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

### Array Attributes

```python
# Demonstrating array attributes
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Array:\n", arr)
print("Shape:", arr.shape)  # Output: (2, 3)
print("Number of dimensions:", arr.ndim)  # Output: 2
print("Size (number of elements):", arr.size)  # Output: 6
print("Data type:", arr.dtype)  # Output: int32
print("Item size (in bytes):", arr.itemsize)  # Output: 4
```
Output:
```
Array:
 [[1 2 3]
 [4 5 6]]
Shape: (2, 3)
Number of dimensions: 2
Size (number of elements): 6
Data type: int32
Item size (in bytes): 4
```

### NumPy Vectorized Operations

```python
arr1 = np.array([1,2,3,4,5])
arr2 = np.array([10,20,30,40,50])

# Element-wise addition
print("Addition:", arr1 + arr2)

# Element-wise subtraction
print("Subtraction:", arr1 - arr2)

# Element-wise multiplication
print("Multiplication:", arr1 * arr2)

# Element-wise division
print("Division:", arr1 / arr2)
```
Output:
```
Addition: [11 22 33 44 55]
Subtraction: [ -9 -18 -27 -36 -45]
Multiplication: [ 10  40  90 160 250]
Division: [0.1 0.1 0.1 0.1 0.1]
```

