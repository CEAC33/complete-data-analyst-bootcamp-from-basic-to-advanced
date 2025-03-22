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

## Pandas - DataFrame And Series

Pandas is a powerful data manipulation library in Python, widely used for data analysis and data cleaning. It provides two primary data structures: Series and DataFrame. A Series is a one-dimensional array-like object, while a DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns).

### Basic Setup

```python
import pandas as pd
```

### Pandas Series

A Pandas Series is a one-dimensional array-like object that can hold any data type. It is similar to a column in a table.

#### Creating a Basic Series
```python
import pandas as pd
data = [1,2,3,4,5]
series = pd.Series(data)
print("Series \n", series)
print(type(series))
```
Output:
```
Series 
 0    1
1    2
2    3
3    4
4    5
dtype: int64
<class 'pandas.core.series.Series'>
```

#### Creating a Series from Dictionary
```python
data = {'a':1, 'b':2, 'c':3}
series_dict = pd.Series(data)
print(series_dict)
```
Output:
```
a    1
b    2
c    3
dtype: int64
```

#### Creating a Series with Custom Index
```python
data = [10,20,30]
index = ['a','b','c']
pd.Series(data, index=index)
```
Output:
```
a    10
b    20
c    30
dtype: int64
```

### Pandas DataFrame

#### Creating DataFrame from Dictionary of Lists
```python
data = {
    'Name': ['Krish', 'John', 'Jack'],
    'Age': [25, 30, 45],
    'City': ['Bangalore', 'New York', 'Florida']
}
df = pd.DataFrame(data)
print(df)
print(type(df))
```
Output:
```
    Name  Age       City
0  Krish   25  Bangalore
1   John   30   New York
2   Jack   45    Florida
<class 'pandas.core.frame.DataFrame'>
```

#### Creating DataFrame from List of Dictionaries
```python
data = [
    {'Name':'Krish', 'Age':32, 'City':'Bangalore'},
    {'Name':'John', 'Age':34, 'City':'Bangalore'},
    {'Name':'Bappy', 'Age':32, 'City':'Bangalore'},
    {'Name':'Jack', 'Age':32, 'City':'Bangalore'}
]
df = pd.DataFrame(data)
print(df)
print(type(df))
```
Output:
```
    Name  Age       City
0  Krish   32  Bangalore
1   John   34  Bangalore
2  Bappy   32  Bangalore
3   Jack   32  Bangalore
<class 'pandas.core.frame.DataFrame'>
```

#### Sample Sales Data DataFrame
```python
# Sample sales data
sales_data = {
    'Transaction ID': [10001, 10002, 10003, 10004],
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    'Product Category': ['Electronics', 'Home Appliances', 'Clothing', 'Books'],
    'Product Name': ['iPhone 14 Pro', 'Dyson V11 Vacuum', "Levi's 501 Jeans", 'The Da Vinci Code'],
    'Units Sold': [2, 1, 3, 4],
    'Unit Price': [999.99, 499.99, 69.99, 15.99],
    'Total Revenue': [1999.98, 499.99, 209.97, 63.96],
    'Region': ['North America', 'Europe', 'Asia', 'North America'],
    'Payment Method': ['Credit Card', 'PayPal', 'Debit Card', 'Credit Card']
}

df_sales = pd.DataFrame(sales_data)
print(df_sales)
```

This creates a more complex DataFrame with sales data including:
- Transaction details
- Product information
- Sales metrics
- Regional data
- Payment information

The DataFrame provides a structured way to work with this tabular data, making it easy to perform operations like:
- Data analysis
- Filtering
- Grouping
- Aggregations
- Statistical calculations

## Data Manipulation and Analysis with Pandas

Data manipulation and analysis are key tasks in any data science or data analysis project. Pandas provides a wide range of functions for data manipulation and analysis, making it easier to clean, transform, and extract insights from data. In this lesson, we will cover various data manipulation and analysis techniques using Pandas.

### Setup and Data Loading

```python
import pandas as pd

# Load the sample dataset
df = pd.read_csv('data.csv')
```

### Basic Data Exploration

#### Viewing First Few Rows
```python
# Fetch the first 5 rows
df.head(5)
```
Output:
```
         Date Category  Value   Product  Sales Region
0  2023-01-01        A   28.0  Product1  754.0   East
1  2023-01-02        B   39.0  Product3  110.0  North
2  2023-01-03        C   32.0  Product2  398.0   East
3  2023-01-04        B    8.0  Product1  522.0   East
4  2023-01-05        B   26.0  Product3  869.0  North
```

#### Viewing Last Few Rows
```python
df.tail(5)
```
Output:
```
          Date Category  Value   Product  Sales Region
45  2023-02-15        B   99.0  Product2  599.0   West
46  2023-02-16        B    6.0  Product1  938.0  South
47  2023-02-17        B   69.0  Product3  143.0   West
48  2023-02-18        C   65.0  Product3  182.0  North
49  2023-02-19        C   11.0  Product3  708.0  North
```

### Data Structure Overview

The sample dataset contains the following columns:
- `Date`: Transaction date
- `Category`: Product category (A, B, C)
- `Value`: Numeric value associated with the transaction
- `Product`: Product identifier (Product1, Product2, Product3)
- `Sales`: Sales amount
- `Region`: Geographic region (East, North, South, West)

## Common Data Manipulation Tasks

#### 1. Basic Information
```python
# Display basic information about the dataset
df.info()
```

#### 2. Statistical Summary
```python
# Generate summary statistics
df.describe()
```

#### 3. Filtering Data
```python
# Filter rows based on conditions
east_region = df[df['Region'] == 'East']
high_sales = df[df['Sales'] > 500]
```

#### 4. Grouping and Aggregation
```python
# Group by Region and calculate mean sales
region_sales = df.groupby('Region')['Sales'].mean()

# Group by multiple columns
category_region = df.groupby(['Category', 'Region']).agg({
    'Sales': 'mean',
    'Value': 'sum'
})
```

#### 5. Sorting Data
```python
# Sort by Sales in descending order
df_sorted = df.sort_values('Sales', ascending=False)

# Sort by multiple columns
df_multi_sort = df.sort_values(['Region', 'Sales'], ascending=[True, False])
```

#### 6. Data Transformation
```python
# Add new calculated column
df['Revenue'] = df['Sales'] * df['Value']

# Apply date transformations
df['Month'] = pd.to_datetime(df['Date']).dt.month
```

### Best Practices for Data Manipulation

1. **Data Integrity**
   - Always keep a copy of the original data
   - Verify data types are correct
   - Handle missing values appropriately

2. **Performance**
   - Use vectorized operations instead of loops
   - Chain operations efficiently
   - Consider memory usage for large datasets

3. **Documentation**
   - Document data transformations
   - Keep track of filtering criteria
   - Note any assumptions made

4. **Quality Checks**
   - Verify results after transformations
   - Check for unexpected values
   - Validate aggregations

This overview covers the fundamental aspects of data manipulation with Pandas, providing a foundation for more advanced analysis techniques.
