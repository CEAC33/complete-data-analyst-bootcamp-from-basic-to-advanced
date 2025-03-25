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


## Reading Data From Different Sources in Pandas

This guide demonstrates various methods to read and write data using Pandas, including working with JSON data and CSV files from different sources.

### 1. Working with JSON Data

#### Reading JSON Data
```python
import pandas as pd
from io import StringIO

# Sample JSON data
Data = '{"employee_name": "James", "email": "james@gmail.com", "job_profile": [{"title1":"Team Lead", "title2":"Sr. Developer"}]}'

# Read JSON data into a DataFrame
df = pd.read_json(StringIO(Data))
```

Output:
```
  employee_name            email                                        job_profile
0         James  james@gmail.com  {'title1': 'Team Lead', 'title2': 'Sr. Developer'}
```

#### Writing JSON Data

Pandas provides multiple ways to convert DataFrames to JSON format using different orientations:

1. Default orientation:
```python
df.to_json()
```
Output:
```
{"employee_name":{"0":"James"},"email":{"0":"james@gmail.com"},"job_profile":{"0":{"title1":"Team Lead","title2":"Sr. Developer"}}}
```

2. Index orientation:
```python
df.to_json(orient='index')
```
Output:
```
{"0":{"employee_name":"James","email":"james@gmail.com","job_profile":{"title1":"Team Lead","title2":"Sr. Developer"}}}
```

3. Records orientation (most readable):
```python
df.to_json(orient='records')
```
Output:
```
[{"employee_name":"James","email":"james@gmail.com","job_profile":{"title1":"Team Lead","title2":"Sr. Developer"}}]
```

### 2. Reading Data from URLs

Pandas can directly read data from URLs, which is particularly useful when working with publicly available datasets:

```python
# Reading CSV data from a URL
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
```

Sample output (first few rows):
```
   0      1     2     3     4    5     6     7     8     9     10    11    12    13
0  1  14.23  1.71  2.43  15.6  127  2.80  3.06  0.28  2.29  5.64  1.04  3.92  1065
1  1  13.20  1.78  2.14  11.2  100  2.65  2.76  0.26  1.28  4.38  1.05  3.40  1050
2  1  13.16  2.36  2.67  18.6  101  2.80  3.24  0.30  2.81  5.68  1.03  3.17  1185
3  1  14.37  1.95  2.50  16.8  113  3.85  3.49  0.24  2.18  7.80  0.86  3.45  1480
```

### Best Practices for Reading Data

1. **Error Handling**
   - Always include error handling when reading from external sources
   - Check for missing or malformed data
   - Verify the data types of columns after reading

2. **Performance Considerations**
   - For large CSV files, consider using `chunksize` parameter
   - Use appropriate data types to optimize memory usage
   - Consider using `nrows` parameter to read only needed rows

3. **Data Validation**
   - Verify the number of columns and rows
   - Check for missing values
   - Validate data types and ranges
   - Confirm the data matches expected format

4. **URL Data**
   - Ensure URL is accessible
   - Handle timeouts and connection errors
   - Consider caching data for frequently used sources

### Common Parameters for read_csv()

- `header`: Specify row number(s) to use as column names
- `sep` or `delimiter`: Specify the separator/delimiter
- `dtype`: Specify data types for columns
- `na_values`: Specify additional strings to recognize as NA/NaN
- `encoding`: Specify the file encoding
- `nrows`: Number of rows to read
- `skiprows`: Lines to skip at the start of the file
- `parse_dates`: Specify columns to parse as dates

This guide covers the basics of reading data from different sources using Pandas. The library supports many more formats including Excel, SQL databases, HTML tables, and more.

## Data Visualization With Matplotlib

Matplotlib is a powerful plotting library for Python that enables the creation of static, animated, and interactive visualizations. It is widely used for data visualization in data science and analytics.

```python
!pip install matplotlib
```

```python
import matplotlib.pyplot as plt
```

```python
x=[1,2,3,4,5]
y=[1,4,9,16,25]

##create a line plot
plt.plot(x,y)
plt.xlabel('X axis')
plt.ylabel('Y Axis')
plt.title("Basic Line Plot")
plt.show()
```

![image](https://github.com/user-attachments/assets/73b41c05-f7b2-4884-8b76-09325941e771)

```python
x=[1,2,3,4,5]
y=[1,4,9,16,25]

##create a customized line plot

plt.plot(x,y,color='red',linestyle='--',marker='o',linewidth=3,markersize=9)
plt.grid(True)
```

![image](https://github.com/user-attachments/assets/63410faa-2f01-4112-a393-74219b11b633)

```python
## Multiple Plots
## Sample data
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 2, 3, 4, 5]

plt.figure(figsize=(9,5))

plt.subplot(2,2,1)
plt.plot(x,y1,color='green')
plt.title("Plot 1")

plt.subplot(2,2,2)
plt.plot(y1,x,color='red')
plt.title("Plot 2")

plt.subplot(2,2,3)
plt.plot(x,y2,color='blue')
plt.title("Plot 3")

plt.subplot(2,2,4)
plt.plot(x,y2,color='green')
plt.title("Plot 4")
```

```python
###Bar Plor
categories=['A','B','C','D','E']
values=[5,7,3,8,6]

##create a bar plot
plt.bar(categories,values,color='purple')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.show()
```

### Histograms

Histograms are used to represent the distribution of a dataset. They divide the data into bins and count the number of data points in each bin.

```python
# Sample data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

##create a histogram
plt.hist(data,bins=5,color='orange',edgecolor='black')
```

![image](https://github.com/user-attachments/assets/4fa0f998-6cc9-48bb-91a9-01f6d261714f)

```python
##create a scatter plot
# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]

plt.scatter(x,y,color="blue",marker='x')
```

![image](https://github.com/user-attachments/assets/d4fc4ea3-1218-4618-bd0a-896af66ae1b6)

```python
### pie chart

labels=['A','B','C','D']
sizes=[30,20,40,10]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0.2,0,0,0) ##move out the 1st slice

##create apie chart
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%",shadow=True)
```

![image](https://github.com/user-attachments/assets/cd905923-af94-4850-8675-b0c121e0d659)

```python
## Sales Data Visualization
import pandas as pd
sales_data_df=pd.read_csv('sales_data.csv')
sales_data_df.head(5)
```

<img width="731" alt="Screenshot 2025-03-21 at 6 49 41 p m" src="https://github.com/user-attachments/assets/e20e1fc6-3f88-46b2-96a6-7fc7556927bf" />

```python
sales_data_df.info()
```

Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 240 entries, 0 to 239
Data columns (total 9 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Transaction ID    240 non-null    int64  
 1   Date              240 non-null    object 
 2   Product Category  240 non-null    object 
 3   Product Name      240 non-null    object 
 4   Units Sold        240 non-null    int64  
 5   Unit Price        240 non-null    float64
 6   Total Revenue     240 non-null    float64
 7   Region            240 non-null    object 
 8   Payment Method    240 non-null    object 
dtypes: float64(2), int64(2), object(5)
memory usage: 17.0+ KB
```

```python
## plot total sales by products
total_sales_by_product=sales_data_df.groupby('Product Category')['Total Revenue'].sum()
print(total_sales_by_product)
```

Output:
```
Product Category
Beauty Products     2621.90
Books               1861.93
Clothing            8128.93
Electronics        34982.41
Home Appliances    18646.16
Sports             14326.52
Name: Total Revenue, dtype: float64
```

```python
total_sales_by_product.plot(kind='bar',color='teal')
```

![image](https://github.com/user-attachments/assets/4c12ca44-3705-4274-860b-f588eee6b03e)

```python
## plot sales trend over time
sales_trend=sales_data_df.groupby('Date')['Total Revenue'].sum().reset_index()
plt.plot(sales_trend['Date'],sales_trend['Total Revenue'])
```

![image](https://github.com/user-attachments/assets/de0c44bb-d182-447a-a055-03e250b4282e)

## Data Visualization With Seaborn

Seaborn is a Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. Seaborn helps in creating complex visualizations with just a few lines of code. In this lesson, we will cover the basics of Seaborn, including creating various types of plots and customizing them. 

```python
!pip install seaborn
```

```python
import seaborn as sns
```

```python
### Basic Plotting With Seaborn
tips=sns.load_dataset('tips')
tips
```

<img width="436" alt="Screenshot 2025-03-21 at 6 55 57 p m" src="https://github.com/user-attachments/assets/ae2ff494-3830-4e1f-8c53-2854dae490f9" />

```python
##create a scatter plot
import matplotlib.pyplot as plt

sns.scatterplot(x='total_bill',y='tip',data=tips)
plt.title("Scatter Plot of Total Bill vs Tip")
plt.show()
```

![image](https://github.com/user-attachments/assets/6eb1ae27-4b2e-41ca-a7cc-2d0e343d14f8)


```python
## Line Plot

sns.lineplot(x='size',y='total_bill',data=tips)
plt.title("Line Plot of Total bill by size")
plt.show()
```

![image](https://github.com/user-attachments/assets/4a34a873-a381-4bc7-a7fd-9e4b09a23b69)

```python
## Categorical Plots
## BAr Plot
sns.barplot(x='day',y='total_bill',data=tips)
plt.title('Bar Plot of Total Bill By Day')
plt.show()
```

![image](https://github.com/user-attachments/assets/e106cd3c-81ec-4dde-8095-1606725655e2)

```python
## Box Plot
sns.boxplot(x="day",y='total_bill',data=tips)
```

![image](https://github.com/user-attachments/assets/aa25505b-37cf-48fa-bc3c-41eb85f8fe93)

```python
## Violin Plot

sns.violinplot(x='day',y='total_bill',data=tips)
```

![image](https://github.com/user-attachments/assets/7e65e8fa-92cf-49e2-b09d-161fabb88777)


```python
### Histograms
sns.histplot(tips['total_bill'],bins=10,kde=True)
```

![image](https://github.com/user-attachments/assets/a0948ced-af69-4cb8-87ed-fd4b7b0ae7a7)


```python
## KDE Plot
sns.kdeplot(tips['total_bill'],fill=True)
```

![image](https://github.com/user-attachments/assets/f76016cb-dccd-42f3-8a01-99813f8189c6)

```python
# Pairplot
sns.pairplot(tips)
```

![image](https://github.com/user-attachments/assets/ea64ca6f-03e2-4f31-9c32-29dfc7747f77)

```python
tips 
```

<img width="419" alt="Screenshot 2025-03-21 at 6 59 59 p m" src="https://github.com/user-attachments/assets/9745a21c-f9e7-47f2-a2b7-566e6ffd3f71" />

```python
## HEatmap
corr=tips[['total_bill','tip','size']].corr()
corr
```

<img width="302" alt="Screenshot 2025-03-21 at 7 00 50 p m" src="https://github.com/user-attachments/assets/7d989e9c-29da-4d00-88a6-e62810952cb5" />

```python
sns.heatmap(corr,annot=True,cmap='coolwarm') 
```

![image](https://github.com/user-attachments/assets/767cadb1-f001-4b23-b9b4-3bdc1a35089c)

```python
import pandas as pd
sales_df=pd.read_csv('sales_data.csv')
sales_df.head()
```

<img width="740" alt="Screenshot 2025-03-21 at 7 01 51 p m" src="https://github.com/user-attachments/assets/ad5257ee-5622-4e40-9978-6e3975164c6c" />

```python
## Plot total sales by product
plt.figure(figsize=(10,6))
sns.barplot(x='Product Category',y="Total Revenue",data=sales_df,estimator=sum)
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.show() 
```

![image](https://github.com/user-attachments/assets/6f1bd428-7fd0-4ea3-8655-e1ddc83cd5aa)

```python
## Plot total sales by Region
plt.figure(figsize=(10,6))
sns.barplot(x='Region',y="Total Revenue",data=sales_df,estimator=sum)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()
```

## Getting started with Statistics

### Intro to Statistics

<img width="830" alt="Screenshot 2025-03-21 at 7 24 58 p m" src="https://github.com/user-attachments/assets/5452f190-782d-4ecf-bc7e-b600aead6d72" />

<img width="747" alt="Screenshot 2025-03-21 at 7 27 21 p m" src="https://github.com/user-attachments/assets/b533ece9-5881-404c-81be-263ce4d84220" />

<img width="637" alt="Screenshot 2025-03-21 at 8 16 09 p m" src="https://github.com/user-attachments/assets/a044ce7e-bc8b-44d6-854d-c68f3b08d54c" />

### Types of Statistics

<img width="824" alt="Screenshot 2025-03-21 at 8 17 42 p m" src="https://github.com/user-attachments/assets/eae0ef1a-d534-42fe-a746-61bc31429746" />

Descriptive:

<img width="420" alt="Screenshot 2025-03-21 at 8 21 17 p m" src="https://github.com/user-attachments/assets/77b09a22-5722-4f6f-a5ce-203323acc31e" />

<img width="260" alt="Screenshot 2025-03-21 at 8 21 59 p m" src="https://github.com/user-attachments/assets/f27b810c-538a-451d-8920-14321198dbdf" />

Inferential:

<img width="270" alt="Screenshot 2025-03-21 at 8 24 07 p m" src="https://github.com/user-attachments/assets/6905355f-7d8a-4d4a-955b-36e2d1a8f85a" />

<img width="260" alt="Screenshot 2025-03-21 at 8 24 28 p m" src="https://github.com/user-attachments/assets/57771a1b-2a6f-40b7-8a5c-833b16f1b688" />

<img width="544" alt="Screenshot 2025-03-21 at 8 25 19 p m" src="https://github.com/user-attachments/assets/468dd50f-4c46-4c9c-8dca-c686a362c215" />

<img width="838" alt="Screenshot 2025-03-21 at 8 28 12 p m" src="https://github.com/user-attachments/assets/ffc6163c-bdf5-4d1c-a7a1-d9e1852b56eb" />

### Population and Sample Data

<img width="823" alt="Screenshot 2025-03-21 at 8 30 15 p m" src="https://github.com/user-attachments/assets/92e6f515-81a3-4750-bb75-5559e7413fc8" />

