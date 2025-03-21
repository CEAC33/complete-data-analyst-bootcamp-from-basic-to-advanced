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

```
#Basic Function
def addition(a,b):
  return a+b

addition(2,3)

#Lambda Function
adition = lambda a,b:a+b
type(addition)
addition(5,6)
```
```
def even(num):
  if even%2==0:
    return True

even(24)
```

```
even1 = lambda num:num%2==0
even1(12)
```

```
def adition(x,y,z):
  return x+y+z

addition(12,13,14)

addition1=lambda x,y,z:x+y+z
addition1(12,13,14)
```

```
## map() - applies a function to all items in a list
numbers = [1,2,3,4,5,6]
def square(number):
  return number**2

square(2)

list(map(lambda x:x**2, numbers))
```






