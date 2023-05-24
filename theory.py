# Generators and Coroutines
# Generator and Couroutines
# Generator and Couroutines

# File I/O
# File Input/Output
# File Input/Output

# Functional Programming
# Functional Programming
# Functional Programming

# Comprehensions
numbers = [1, 2, 3, 4, 5]

# List comprehension
squared_numbers = [num**2 for num in numbers]
print(squared_numbers) # Output: [1, 4, 9, 16, 25]

# Set comprehension
even_numbers = {num for num in numbers if num % 2 == 0}
print(even_numbers) # Output: {2, 4}

# Dictionary Comprehension
number_dict = {num: num**2 for num in numbers}
print(number_dict) # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Concurrency
import concurrent.futures

def square_numbers(num):
    return num**2

numbers = [1, 2, 3, 4, 5]

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(square_number, numbers)

print(list(results)) # Output: [1, 4, 9, 16, 25]

# Dictionaries
person = {
    "name": "John",
    "age": 25,
    "city": "New York"
}

print(person["name"]) # Output: John
print(person.get("age")) # Output: 25

person["occupation"] = "Engineer"
print(person)

del person["age"]
print(person)

# Modules and Imports
import math

print(math.sqrt(16)) # Output: 4.0

from random import randint

print(randint(1, 10)) # Output: Random number between 1 and 10

# Regular Expression Usage
import re

pattern = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
email = "example@example.com"

if re.match(pattern, email, re.IGNORECASE):
    print("Valid email")
else:
    print("Invalid email")

# Classes
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.rasius**2
    
    def circumference(self):
        return 2 * math.pi * self.radius

my_circle = Circle(5)
print("Area:",my_circle.area())
print("Circumference:", my_circle.circumference())


# Control Flow
num = 10
if num > 0:
    print("Number is positive")
elif num < 0:
    print("Number is negative")
else:
    print("Number is zero")

for i in range(5):
    print(i)

while num > 0:
    print(num)
    num -= 1

# Exception Handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Division by zero")

try:
    value = int("abc")
except ValueError:
    print("Error: Invalid integer")

# Strings
message = "Hello, World!"

print(message.upper()) # HELLO, WORLD!
print(message.lower()) # hello, world!
print(message.startwith("Hello")) # True
print(message.endwith("World!")) # True
print(message.split(".")) # ['Hello', 'World!']
print(len(message)) # 13


# Precedence and Associativity
result = 2 + 3 * 4 # Multiplication has higher precendence than addition
print(result) # Output: 14

# Formatting
name = "John"
age = 25
print("My name is {} and I'm {} years old.".format(name, age)) # Output: name is John and I'm 25 years old.

# Decorators
def decorator(func):
    def wrapper():
        print("Before function execution")
        func()
        print("After function execution")
    return wrapper

@decorator
def hello():
    print("Hello, world!")

hello() # Output: Before function execution
        # Hello, world!
        # After function execution

# Encapsulation
class Car:
    def __init__(self, make, model):
        self._make = make # Protected attribute
        self.__model = model # Private attribute
    
    def get_make(self):
        return self._make
    
    def get_model(self):
        return self.__model

    def set_model(self, model):
        self,__model = model

my_car = Car("Toyata", "Corolla")
print(my_car.get_make()) # Output: Toyota
print(my_car.get_model()) # Output: Corolla

my_car.set_model("Camry")
print(my_car.get_model()) # Output: Camry


# Inheritance
# Functions
# Generators and Coroutines
# File I/O
# Functional Programming
# Comprehensions
# Concurrency
# Dictionaries
# Modules and Imports
# Regular Expression Usage
# Classes
# Control Flow
# Exception Handling
# Strings
# Precedence and Associativity
# Formatting and Decorators
# Encapsulation


# Regular Expression Usage:
import re

pattern = r"\b[A-ZO-9.%+-]+@[A-ZO-9.-]+\.[A-Z]{2,}\b""
text = "Contact us at info@example.com or support@example.org."
matches = re.findall(pattern, text, re.IGNORECASE)
print(matches)

# Classes:
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
rectangle = Rectangle(5, 10)
print(rectangle.area())

# Control Flow:
x = 5

if x > 10:
    print("x is greater than 10")
elif x < 5:
    print("x is less than 5")
else:
    print("x is between 5 and 10")

# Exception Handling:
try:
    num - int(input("Enter a number: "))
    result = 10 / num
    print("Result:", result)
except ZeroDivisionError:
    print("Error: Cannot divided by zero")
except ValueError:
    print("Error: Invalid input:")

# Strings
greeting = "Hello, world!"
print(greeting[0]) # print "H"
print(greeting[7:12]) # print "world"
print(greeting.upper()) # prints "HELLO, WORLD!""

# Precedence and Associativity
x = 2 + 3 * 4
print(x) # prints 14

y = (2+ 3) * 4
print(y) # prints 20

# Formatting and Decorators
class MyClass:
    def __init__(self, value):
        self._value = value
    
    @staticmethod
    def my_static_method():
        print("This is a static method.")
    
    @ property
    def value(self):
        return self._value

my_object = MyClass(42)
MyClass.my_static_method()
print(my_object.value)


'''
Regular Expression Usage:
Regular expressions are patterns used to match character combinatin in strings. They
can be used for a variety of tasks, such as validating input or searching for specfic
patterns in text. Regular expressions are typically written using a special syntax that includes metacharacters,
which represent sets of characters or behaviors.

Classes:
Classes are a way to define custom data types in object-oriented programming. They allow
you to encapsulate data and behaviour into a single unit, which can then be instantiated
and manipulated as objects. Classes can contain attributes, which represent data 
associated with the class, and methods, which represent the behvaours or actions that the class can perform

Control Flow:
Control flow refers to the order in which statements are executed in a program. Control flow statements, such as if/else statements and loops, allow you to change
the oreder of execution based on certain contions or criteria. For exanmple, an if statement might execute one block of code if a certain condition is true, and a differnt block of code
if the condition is false.

Exception Handling:
Exception handling is a mechanism for dealing with errors or other unexpected events
that occur during the execution of a program. When an exception is thrown, the program
can catch it and take appropriate action, such as displaying an error message or trying to
recover from the error. Exception handling is an important part of writing robust and
reliable software.

Strings:
String are a data type used to represent text in programming. They can contain any cobination of letters,
numbers and symbols, and can be manipulated using a variety of string methods. Some common string operations
include concatenation (joining two strings together), slicing (selecting a portion of a string), and searching
(finding the location of a specific substring within a larger string)

Precedence and Assosiativity:
Precedence and associativity are rules that determine the order in which operators are
evaludated in an expression. Precedence refers to the orer in which operators of different levels
of precedence are evaluated, while associativity determines the orer in which
operators of the same precedence are evaludated. For example, in the expression 2+3*4,
the multiplication operator (*) has higher precedence than the addition operator (+), so
the expression is evaluated as 2 + (3*4) = 14

Formatting and Decorators:
Formatting and decorators are ways to modify the behavior or output of functions an objects
in Python. Formatting allows you to control the way that data is displayed or formatted,
while decorators allow you to add additional functionality or behaviour to a 
function or object without modifying its underlying code. Examples of decorators include
@statismethod, which allows you to define a method that doesn't require an instance of the class,
and @property, which allows you to define a method that is accessed like an attribute.

Encapsulation:
Encapsulation is a concept in object-oriented programming that refers to the idea of bundling data and behaviour into a single unit, 
and then restricting access to that unit from outside the class. This allows you to control how the data is manipulated, and helps prevent
accidental or malicious modificaiton of the data. Encapsulation is typically achieved using access modifiers, such as private
or protected, which limit the visibility of class members from outside the class
'''
# Inheritance
# Functions
# Generators and Coroutines
# File I/O
# Functional Prgramming
# Comprehensions
# Concurrency
# Dictionaries
# Modules and Imports
# Regular Expression Usage
# Classes
# Control Flow
# Exception Handling
# Strings
# Precedence and Associativity
# Formatting and Decorators
# Encapsulation



'''
Inheritance,
Functions,
Generators and Coroutines,
File I/O
Lists and Tuples
Functional Programming
Comprehensions
Concurrency
Dictionaries
Modules and Imports
Regular Expression Usage
Classes
Control Flow
Exception Handling
Strings
Precedence and Associativity
Formatting and Decorators
Encapsulation
'''

'''
Regular expression (or regex) are patterns used to match character combinations in strings.
Python has a built-in module called 're' that provides support for regular expressions.
-> Regular expressions are patterns to match characters
'''
import re
text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"

matches = re.findall(pattern, text)
print(matches)

'''
Control flow refers to the order in which statements are executed in a program. 
Python provides several control flow statements to help you controlt he flow of your program's execution.
These include if/else statements, for loops, while loops, and try/except statements.
'''
x = 5

if x > 10:
    print("x is greater than 10")
else:
    print("x is less than or equal to 10")


# Walrus operators
'''
Walrus operator, :=, is a new operator that was introduced in Python 3.8.
It allows you to assign a value to a variable as part of an expression.
The walrus operator is useful when you want to evaluate an expression and assign the result to a variable.
And then use that variable in a subsequent expression. 
'''
import random
while (guess := input("Guess a number between 1 and 10: ")) != str(random.randint(1,10)):
    print("Incorrect guess. Try again.")
print("Congratulations! You guess the number!")

# fh write a line containing (abcdef)
fh.write("abcdef" + "\n")
print("abcdef", file = fh)

# The child class acquires the properties and can access all the data members and functions defined in the parent class

# Precedence and Associativity
# Example of operator precedence and associativty
x = 10
y = 5

# Example 1
result = x + y * 2 # Multiplication has higher precedence than addition
print(result) # 20

# Example 2
result = (x + y) * 2 # Parentheses have highest precedence
print(result) # 30

# Example 3
result = x / y ** 2 # Exponentiation has higher precedence than division
print(result) # 0.4

# Example 4
result = 10 / 5/ 2 # Division has left-to-right associativity
print(result) # 1.0

# Example 5
result = 2 ** 3 ** 2 # Exponentiation has right-to-left associativity
print(result) # 512



'''
Encapsulation is a technique used to hide the internal details of an object from the outside world. In Python, we can use underscores to indicate whether a method or attribute
should be considered private or not.
Note: The underscores indicate that the make and model attributes should be considered private. This provides getter and setter methods to allow access to 
these attributes from outside the class
'''
class Car:
    def __init__(self, make, model):
        self._make = make
        self._model = model
    def get_make(self):
        return self._make
    def get_model(self):
        return self._model
    def set_make(self, make):
        self._make = make
    def set_model(self, model):
        self._model = model
'''
Formatting and decorators:
Python has a built-in formation function that can be used to format strings. Decorators are a way to modify the behavior of a function or class.
'''
def format_output(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return wrapper
@format_output
def add(a, b):
    return a + b
print(add(2, 3)) # Output: The reuslt is: 5


'''
Yield keyword is used in a function to create a generator object. When a generator function is called, it returns a generator object without actually executing the body
of the function. The yield keyword is used to produce a value from the generator functin and pause the function's execution until the next value is requested
Yield -> Pause until the next value is requested
'''
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        yield a
    a, b = b, a + b

for num in fibonacci(10):
    print(num)


# Web scraping with Beautiful Soup
# Beautiful Soup is a Python library that makes it easy to extract data from HTML and XML documents
import requests
from bs4 import BeautifulSoup

# Send a request to the website and get its HTML content
url = "https://www.nytimes.com/"
reponse = requests.get(url)
html_content = response.content

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(html_content, "html.parser")

# Find all the articles on the page
articles = soup.find_all("article")

# Extract the titles and links of each article
for article in articles:
    title = article.find("h2").text.strip()
    link = article.find("a")["href"]
    print(f"Title: {title}")
    print(f"Link: {link}")

# Data analysis with Pandas
import pandas as pd

# Read in a CSV file
df = pd.read_csv("data.csv")

# Print the first 5 rows of the data
print(df.head())

# Calculate some basic statistics on the data
print(df.describe())

# Machine learning with scikit-learn:
# scikit-learn is a python library for maching learning
# Here sciki-learn is used to train a linear regression model on some data
from sklearn.linear_model import LinearRegression
import numpy as np

# Create some sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# Train a linear regression model on the data
model = LinearRegression().fit(x, y)

# Make a prediction with the model
prediction = model.predct(np.array([[6]]))
print(prediction)

# Regular expressions (regex) are a way to match patterns in strings.
# Regex syntax can be complex and difficult to understand, especially for those who are not familiar with it
import re

text = "The quick brown fox junmps over the lazy dog."

# Match any word starting with the letter "q"
pattern = r"\bq\w+"

matches = re.findall(pattern, text)
print(matches) # Output: ['quick']


import asyncio

async def my_coroutine(id):
    print(f"Starting coroutine {id}")
    await asyncio.sleep(2)
    print(f"Ending coroutine {id}")

async def main():
    # Create a list of coroutines
    coroutines = [my_coroutine(1), my_coroutine(2), my_coroutine(3)]

    # Run the coroutines concurrently
    await asyncio.gather(*coroutines)

# Run the main function asynchronously
asyncio.run(main())

# Error Handling
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Threading
import threading

def print_numbers():
    for i in range(1, 11):
        print(i)

def print_letters():
    for letter in "ABCDEFGHIJK":
        print(letter)

# Create two threads that will execute the print_numbers and print_letters functions simultaneously
thread1 = threading.Thread(target=print_numbers)
thread2 = threading.Thread(target=print_letters)

# Start both threads
thread1.start()
thread2.start()

# Wait for both threads to finish before exiting the program
thread1.join()
thread2.join()

# Multiprocessing
from multiprocessing import Process

def print_numbers():
    for i in range(1, 11):
        print(i)

def print_letters():
    for letter in "ABCDEFGHIJK":
        print(letter)

# Create two processes that will execute the print_numbers and print_letters functions simultaneously
process1 = Process(target=print_numbers)
process2 = Process(target=print_letters)

# Start both processes
process1.start()
process2.start()

# Wait for both processes to finish before exiting the program
process1.join()
process2.join()


# Arbitrary number of arguments using the *args and **kwargs syntax
def my_function(*args, **kwargs):
    print(args)
    print(kwargs)
my_function(1, 2, 3, name="Alice", age=30)
# Output:
# (1, 2, 3)
# {'name': 'Alice', 'age', 30}

# Deleting in dictionaries
# Elements can be deleted from a dictionary using the del keyword or the pop() method
# Using the del keyword
my_dict = {"apple": 1, "banana": 2, "orange": 3}
del my_dict["banana"]
print(my_dict) # Output: {"apple": 1, "orange": 3}

# Using the pop() method
my_dict = {"apple": 1, "banana": 2, "orange": 3}
my_dict.pop("banana")
print(my_dict) # Output: {"apple": 1, "orange": 3}

# Note that when using 'pop()', the default value to be returned if the key is not found in the dictionary
my_dict = {"apple": 1, "banana": 2, "orange": 3}
my_dict.pop("mango", 0) # Returns 0 because "mango" is not in the dictionary

#clear() method can be used to remove all elements from a dictionary
my_dict = {"apple": 1, "banana": 2, "orange": 3}
my_dict.clear()
print(my_dict) # Output: {}

# LinkedLists
# delete_node(): removes a node from a linked list
def delete_node(head, val):
    # Case 1: head node contains the value to delete
    if head.val == val:
        head = head.next
        return head
    # Case 2: value to delete is not in head node
    curr = head
    prev = None
    while curr:
        if curr.val == val:
            prev.next = curr.next
            return head
        prev = curr
        curr = curr.next
    return head
# reverse_list()

# The XOR Operator is represented by the caret('^') symbol.
# The XOR operator returns a '1' in each bit position where the corresponding bits of either but not both operands are '1'
a = 0b1010 # binary representation of 10
b = 0b1100 # binary representation of 12

c = a ^ b # XOR operation

print(bin(c)) # Output: 0b0110 (binary representation of 6)

# Dictionaires
# get(): returns the value for a given key. If the key is not found, it returns a default value
d = { 'a': 1, 'b': 2, 'c': 3 }
print(d.get('a')) # Output: 1
print(d.get('d')) # Output: None
print(d.get('d'), 0) # Output: 0

# keys(): returns a view object that contains the keys of the dictionary
d = {'d': 1, 'b': 2, 'c': 3}
keys = d.keys()
print(keys) # Output: dict_keys(['d', 'b', 'c'])

# values(): returns a view object that contains the values of the dictionary
d = {'d': 1, 'b': 2, 'c': 3}
values = d.values()
print(values) # Output: dict_values([1, 2, 3])

# items(): returns a view object that contains the key-value pairs of the dictionary
d = {'d': 1, 'b': 2, 'c': 3}
items = d.items()
print(items) # Output: dict_items([('d', 1), ('b', 2), ('c', 3)])

# update(): updates the dictionary with the key-value pairs from another dictionary
d1 = {'a': 1, 'b': 2}
d2 = {'c': 3, 'd': 4}
d1.update(d2)
print(d1) # Output: {'a': 1, 'b': 2, 'c': 3, 'd': 4}




# Lists
# append(): Adds an element to the end of the list
lst = [1, 2, 3]
lst.append(4)
print(lst) # Output: [1, 2, 3, 4]

# extend(): Adds elements from an iterable to the end of the list
lst = [1, 2, 3]
lst.extend(4, 5)
print(lst) # Output: [1, 2, 3, 4, 5]

# insert(): Inserts an element at a specific index
lst = [1, 2, 3]
lst.insert(1, 5)
print(lst) # Output: [1, 5, 2, 3]

# remove(): Removes the first occurence of a specific element from the list
lst = [1, 2, 3, 4]
lst.remove(2)
print(lst) # Output: [1, 3, 4]

# pop(): Removes and returns the last element of the list
lst = [1, 2, 3]
x = lst.pop()
print(x) # Output: 3
print(lst)  # Output: [1, 2]



# A Function to check if a string is a palindrome or not
def is_palindrome(s):
    return s == s[::-1]

# A function to remove duplicates from a list
def remove_duplicate(lst):
    return list(set(lst))

# A function to reverse a linked list
class Node:
    def __init__ (self, value = None):
        self.value = value
        self.next = None

# A function to reverse a linked list
def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev

# A function to find the maximum subarray sum
def max_subarray_sum(arr):
    max_sum = float(-'inf')
    current_sum = 0
    for num in arr:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# A function to check if a binary tree is balanced
class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.value = value
        self. left = left
        self.right = right

def is_balanced(root):
    if not root:
        return True
    left_height = height(root.left)
    right_height = height(root.right)
    if abs(left_height - right_height) <= 1 and is_balanced(root.left) and is_balanced(root.right):
        return True
    return False

def height(root):
    if not root:
        return 0
    return 1 + max(height(root.left), height(root.right))




'''
x = random.randint(0, 9)
x = random.random()
x = random.uniform(5, 10)
random.shuffle(numbers)
x = random.choice(numbers)
'''
# The random library in Python provides a set of functions for generating random numbers and sequences. 
# Generate a random integer between two values:
import random
# Generate a random integer between 0 and 9
x = random.randint(0, 9)
print(x) # Output: 4

# Generate a random floating-point number between two values:
x = random.random()
print(x)
# Generate a random floating-point number between 5 and 10
x = random.uniform(5, 10)
print(x) # Output: 7.329728936997814

# Shuffle a list:
# Shuffle a list in place
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
print(numbers)

# Choose a random element from a list:
fruits = ['apple', 'banana', 'cherry']
x = random.choice(fruits)
print(x) # Output: "banana"


# List comprehension: List comprehensions provide a concise way to create lists based on exisiting lists.
numbers = [1, 2, 3, 4, 5]
squares = [x ** 2 for x in numbers]
print(squares) # Output: [1, 4, 9, 16, 25]

# Generators are a type of iterable, like lists or tuples, but thye don't store all the values in memory.
# Instead, they generate the values on the fly as you iterate over them.
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        yield a
        a, b = b, a + b
for num in fibonacci(10):
    print(num)

# Decorators are a way to modify or enhance the behavior of a function.
# They are often used to add additional functionality to functions without modifying their source code.
def my_decorator(func):
    def wrapper():
        print("Before the function is called.")
        func()
        print("After the function is called.")
    return wrapper
@my_decorator
def say_hello():
    print("Hello1")
say_hello()

# Context managerss are a way to manage resources (like files or database connections) that need to be cleaned up after they are used
with open('file.txt', 'w') as f:
    f.write('Hello, world!')

# Error handling is the process of handling errors that may occur during the execution of a program.
# In Python, you can use try-except blocks to handle errors.
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Cannot divide by zero.")

'''
In Python, you can define multiple __init__ methods in a single class, each with a different set of parameters. This is known as method overloading, which is the ability to define multiple methods with the same name but different parameters.
'''
class Person:
    def __init__(self):
        self.name = "Unknown"
        self.age = 0
    def __init__(self, name, age):
        self.name = name
        self.age = age
person1 = Person()
print(person1.name) # Output: "Unknown"
print(person1.age) # Output: 0

person2 = Person("Alice", 25)
print(person2.name) # Output: "Alice"
print(person2.age) # Output: 25

'''
__init__ is a special method in Python classes that is automatically called when an object is created from the class.
It is used to initialize the attributes (variables) of the object with default values, or with values passed as arguments.
___________________________________________________________________________________________________________________________
The class Person with an __init__ method takes two parameters name and age. Inside the method, we set the values of two instance variables
(self.name and self.age) to the values passed as arguments
When a new Person object was created and the values name and age was passed to the constructor,
The constructor automatically calls the __init__ method to initialize the object's attributes
In the __init__ is used to set up all the initial state of an object when it is created. 
It is one of the most commonly used methods in Python classes, and it is called automatically when a new object is created.
___________________________________________________________________________________________________________________________
__init__ is similar to constructor in C++ and Java. It is run as soon as an object of a class is instantiated.
Languages like Java or C++, a constructor is a special method that is called when an object is created and it is used to initialize the object's attributes
In other languages like Java or C++, a constructor is a special method that is called when an object is created and it is used to initialize the object's instance variables (or attributes).
The main difference between Python __init__ and constructors in other languages is that in Python, you can have multiple __init__ methods with different parameters, but in
other languages, you can only have one constructor with the same name as the class.

'''
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person1 = Person("John", 36)
print(person1.name) # Output: "John"
print(person1.age) # Output: 36

person2 = Person("Jane", 25)
print(person2.name) # Output: "Jane"
print(person2.age) # Output: 25



'''
lstrip() and rstrip() are string methods in Python that are similar to strip(),
but they only remove whitespace characters from the left (beginning) or right (end) side of a string, respectively.
'''
my_string = "    hello     "
lstripped_string = my_string.lstrip()
print(lstripped_string) # Output: "hello     "

rstripped_string = my_string.rstrip()
print(rstripped_string) # Output: "    hello"

my_string2 = "*****hello*****"
lstripped_string2 = my_string2.lstrip("*")
print(lstripped_string2) # Output: "hello*****"

rstripped_string2 = my_string2.rstrip("*")
print(rstripped_string2) # Output: "*****hello"


'''
lstrip() is used to remove whitespace characters from the begining of a string.
rstrip() is used to remove whitespace characters from the end of a string.
strip() is used to remove whitespace characters from the begining and/or end of a strip
'''

'''
strip() is used to remove whitespace characters (spaces, tabs, and newlines) from the begining
and/or end of a string. By default, it removes all whitespace characters, but you can also pass a string
argument to remove specific characters.
'''
# -> removes all whitespace characters from the begining and end of the string
my_string = "   hello   "
stripped_string = my_string.strip()
print(stripped_string) # Output: "hello"

my_string2 = "0000000hello0000000"
stripped_string2 = my_string2.strip('0')
print(stripped_string2) # Output: "hello"




'''
split() is used to split a string into a list of substrings based on a delimiter.
The delimiter is passed as an argument to the 'split()' method. By default, the delimiter
is a whitespace character, but it can be any string.
'''
# -> creates a list of substrings based on delimiter
my_string = "hello world"
my_list = my_string.split()
print(my_list) # Output: ['hello', 'world']


# Variables and data types:
x = 10
y = "hello"
z = True
my_list = [1, 2, 3, 4, 5]
my_dict = {"name": "John", "age": 36}

# Control flow:
if x > 5:
    print("x is greater than 5")
elif x < 5:
    print("x is less than 5")
else:
    print("x is equal to 5")
for i in range(10):
    print(i)

while x < 20:
    print(x)
    x += 1

# Functions:
def add_numbers(x, y):
    return x + y
result = add_numbers(3, 4)
print(result)

# Classes and objects
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return self.length * self.width

my_rectangle = Rectangle(4, 5)
print("Length:", my_rectangle.length)
print("Width:", my_rectangle.width)
print("Area:", my_rectangle.area())

# Modules and packages
import math
x = math.sqrt(25)
print(x)

# File I/O
with open("myfile.txt", "w") as f:
    f.write("Hello, world!")

with open("myfile.txt", "r") as f:
    contents = f.read()
    print(contents)

# Exception handling
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Iterators and generators
my_list = [1, 2, 3, 4, 5]

# Using an iterator
my_iterator = iter(my_list)
print(next(my_iterator))
print(next(my_iterator))

# Using a generator
def my_generator():
    for i in range(10):
        yield i
for i in my_generator():
    print(i)


# Import a module and use one of its functions
import math
x = math.sqrt(25)
print(x)

# Define a class with a constructor and instance variables
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
    def area(self):
        return self.length  * self.width

# Create an object of the class and call its methods
my_rectangle = Rectangle(4, 5)
print("Length:", my_rectangle.length)
print("Width:", my_rectangle.width)
print("Area:", my_rectangle.area())

# Use a conditional statement to check a condition
if x > 5:
    print("x is greater than 5")
else:
    print("x is less than or equal to 5")

# Use a for loop to iterate over a list
my_list = [1, 2, 3, 4, 5]
for item in my_list:
    print(item)

# Hadnle an exception
try:
    y = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")


# Define a variable and print it
x = 10
print(x)

# Define a list and print its contents
my_list = [1, 2, 3]
print(my_list)

# Define a function taht takes two arguments and returns their sum
def add_numbers(x, y):
    return x + y

# Call the function and print the result
result = add_numbers(3,4)
print(result)

# Define a class with a method that prints a message
class MyClass:
    def say_hello(self):
        print("Hello from MyClass!")

# Create an object of the class and call its method
my_object = MyClass()
my_object.say_hello()


'''
Lists -> Mutable
String -> Immutable
Tuples -> Immutable
Sets -> Mutable
Dictionaries -> Mutable
'''

'''
Difference between class and object
Class is a blueprint for creating objects. An object is an instance of a class. 

A class defines a set of attributes and methods that describe the behavior and properties
of objects created from it. It defines the structure of an object, including its data members,
and member functions. Classes can be seen as a blueprint or a cookie cutter, which is used to create objects that share common characteristics.

On the other hand, an object is an instance of a class, which means that it is a specific realization of a class.
It represents a unique occurence of a class, with its own unique state and behavior. Objects are created from
a class, with its own unique state and behavior Objects are created from class using the constructor method,
which initializes the object with initial values

To summarize, a class is an abstract entity that defies the attributes and methods of a group of objects,
while an object is a specific instance of a class with its own unique state and behavior. 
In other words, a class is a general concept, while an object is a specific realizaiton of that concept. 





_________________________________________________________________________
Python is an object-oriented programming (OOP) language. It supports thae fundamental principles of OOP,
such as encapsulation, inheritance, and polymorphism. 

In Python, everything is an object, including variables, functions, and data structures like
lists and dictionares. Objects are created from classes, which define their properties and behaviours. Classes in Python can inherit from other classes,
allowing for code reuse and a hierarchical organization of code.

Python's support for OOP makes it a powerful language for building complex applications, especially those that
require modularity, maintainability, and extensibility. Additionally, Python's syntax and built-in functions ,ake it easy to
write and read code that follow OOP principles.


'''

'''
1. Variables: Variables are used to store data in Python.
They can hold different types of data, such as strings, integers, and floating-point numbers.
2. Lists: Lists are one of the most commonly used data structures in Python.
They are used to store an ordered sequence of elements, which can be of any data type.
Lists are mutable, meaning their contents can be changed.
3. Tuples: Tuples are similar to lists, but they are immutable, meaning
their contents cannot be changed once they are defined.
4. Dictionaries: Dictionaries are used to store key-value pairs.
They are commonly used to represent structured data, such as JSON data or configuration settings.
5. Sets: Sets are used to store unique values in Python. 
They are commonly used to perform set operations, such as union, intersection, difference.
6. Loops: Loops are used to iterate over a sequence of elements in Python. 
There are two types of loops in Python: for loops and while loops.
7. Conditionals: Conditionals are used to execute different blocks of code based on whether
a condition is true or false. The most common conditional statements in Python are if, elif, and else.
8. Functions: Functions are used to encapsulate a block of code that can be reused multiple times.
They can take parameters and return values.
9. Classes: Classes are used to define custom data types in Python.
They encapsulate data and behavior into a single object, making it easier to organize and manage complex code.
10. Exceptions: Exceptions are used to handle errors and other exceptional conditions in Python.
They allow you to gracefully handle errors and prevent your program from crashing.

'''



'''
Mutables -> Lists, Dictionaries, Sets (Can be changed)
Immutable -> Tuples, Strings, Numbers (Cannot be changed)
1. Dictionaries: A dictionary is a collection of key-value pairs. Each key is unique and is assocaited with a value. 
Dictionaries are mutable, which means their contents can be changed after they are created.
2. Tuples: A tuple is an immutable collection of ordered elemnts. Tuples are similair to lists, but
they cannot be modified once they are created.
3. Sets: A set is a collection of unique elements. Sets are mutable and can be modified after they are created.
4. Queues: A queue is a collection of elements that supports adding elements to the end of the queue and removing elements from the front of the queue.
5. Stacks: A stack is a collection of elements that supports adding elements to the top of the stack
and removing elements from the top of the stack
'''
# Dictionaries are mutable
my_dict = {"apple": 1, "banana": 2, "orange": 3}
print(my_dict["banana"]) # Output: 2

# Tuples are immutable
my_tuple = (1, 2, 3)
print(my_tuple[1]) # Output: 2

# Sets are mutable
my_set = {1, 2, 3}
print(2 in my_set) # Output: True

# Queues are mutable
from queue import Queue
my_queue = Queue()
my_queue.put(1)
my_queue.put(2)
print(my_queue.get()) # Output: 1

# Stack are mutable
my_stack = []
my_stack.append(1)
my_stack.append(2)
print(my_stack.pop()) # Output: 2





'''
Important Things
1. Clear and concise code: The could should be easy to read, understand and maintain.
Use comments and docstrings to document your code and make it more undestandable.
2. Robust error handling: Make sure the code handles all possible errors gracefully.
3.Efficient Performance: Optimize the code to ensure it runs as fast as possible,
especially for computationally intensive tasks.
4. Proper use of variables and data structures: Use variables and data structures
appropriately and avoid unnessary memory usage.
5. Proper indentation and formatting: Follow consistent formatting guidelines for the code including
indentation, whitespace, line length
6. Good naming conventions: Use descriptive and meaningful names for variables, functions,
and classes taht accurately reflect their purpose and usage.
7.Proper documentation: Provide detailed documentation for the code including function signatures and parameter descriptions
8. Version control: Use version control tools like Git to keep track of changes to the code and collaborate with others
9. Testing and debugging: Write unit tests to verify that the code works as intended and debug any issues that arive
10. Security: Be aware of security concerns and implement appropriate measures to protect sensitive data and prevent attacks.
'''
'''
The code uses a for loop to iterate through the numbers 1 to 20.
The if statement is to check if each numbers is even. (If it is divisible by 2 with no remainder)
If the number is even, it is printed to the console. The if i == 10: break statement
is used to exit the loo[ after the first 10 even numbers have been printed.
'''
# Prints our the first 10 even numbers
for i in range(1, 21):
    if i % 2 == 0:
        print(i)
    if i == 10:
        break


# open() function to open files for reading
with open('filename.txt', 'r') as file:
    contents = file.read()
    print(contents)
'''
1. open('filename.txt', 'r') is the syntax to open a file fo reading mode 'r'
if the file does not exist, Python will raise a FileNotFoundError
2. with - This line of code starts a context in which the file is opened.
The advantage of using a with statement is that it will automatically 
close the file for you once you are done with it
3. 'as file' - This line of code assigns the file object to the variable 'file'
4. contents = file.read() - This line of code reads the entire contents of the file
and stores it in the variable 'contents'
5. print(contents) - This line of code prints the contents of the file to the console
'''
# Open a file for writing, appending, or in binary mode
# Open a file for writing
with open('output.txt', 'w') as file:
    file.write('Hello, world!')

# Open a file for appending
with open('log.txt', 'a') as file:
    file.write('An error occurred.')

# Open a binary file
with open('image.png', 'rb') as file:
    data = file.read()




''' In Python, dictionaries are mutable.
This means that the contents of a dictionary can be changed after it has been created.
For example, you can add, remove, or modify key-value pairs in a dictionary using various
methods like update(), pop(), del, etc
'''
# Create a dictionary
person = {"name": "Mirage", "age": 27}

# Modify a key-value pair
person["age"] = 24

# Add a new key-value pair
person["city"] = "New York"

# Remove a key-value pair
del person["age"]

print(person)





# Loop through a string:
word = "Python"
for letter in word:
    print(letter)

# List are mutable
# Loop through a list:
numbers = [1, 2, 3, 4, 5]
for num in numbers: 
    print(num)

# Tuple are immutable
# Loop through a tuple:
numbers = (1, 2, 3, 4, 5)
for num in numbers:
    print(num)

# Dictionary are mutable
# Loop through a dictionary:
person = {"name": "John", "age": 30, "city": "New York"}
for key, value in person.items():
    print(key + ":", value)

# Loop through multiple lists at once using the zip() function:
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(name, "is", age, "years old. ")




'''
In Python, 'upper()' and 'lower()' are string methods used to convert the case of the letters in a string
'''
'''
upper() is used to convert all the lowercase letters in a string to uppercase
It returns  a new string with all the lowercase letters replaced by their uppercase counterparts.
'''
string = "hello world"
upper_string = string.upper()
print(upper_string) # Output: "HELLO WORLD"

'''
lower() is used to convert all the uppercase letters in a string to lowercase
It returns a new string with all the uppercase letters replaced by their lowercase counterparts
'''
string = "HELLO WORLD"
lower_string = string.lower()
print(lower_string) # Output: "hello world"




import openpyxl

def generate_excel_sheet(filename, data):
    # Create a new workbook
    workbook = openpyxl.Workbook()

    # Select the new sheet
    sheet = workbook.active
    sheet.title = "Sheet Report"

    # Write the data to the seet
    for row in data:
        sheet.append(row)

    # Save the workbook to a file
    workbook.save(filename)

data = [
    ["Name", "Age", "Gender"],
    ["Alice", 25, "Female"],
    ["Bob", 30, "Male"],
    ["Charlie", 35, "Male"]
]

generate_excel_sheet("report.xlsx", data)


'''
In Python, strip() and split() are string methods that operate on a string and return a modified version of it
'''
'''
strip() is used to remove leading and trailing whitespace characters (spaces, tabs, newlines) from a string.
It returns a new string with the leadering and trailing whitespace removed.
'''
# strip() method removes the leadering and trailing whitespace
# from the string "  hello  \n"
string_with_whitespace = "   hello    \n"
stripped_string = string_with_whitespace.strip()
print(stripped_string)

'''
split() is used to split a string into a list of substrings based on a specified delimiter character.
By default, the delimiter is a space character
But any character can be specified as the delimiter
'''
# split() method splits a string into a list of substrings
sentence = "This is a sentence."
words = sentence.split()
print (words) # Output: ["This", "is", "a", "sentence"]





'''
Python, a list is mutable, which means that you can modify its contents by adding, remove, or changing elements.
A tuple is immutable, which means that you cannot modify its contents
Tuples make it useful for situations where the stored collection of items can't be changed, such as coodinates or database records. 
'''
myTuple = (1, 2, 3, 4, 5)
print(myTuple)
print(myTuple[0])
myList = [1, 2, 3, 4, 5]
print(myList)
print(myList[0])
myList[0] = 6
print(myList)

# Dictionaries
myDictionary = {'name': 'John', 'age': 30, 'city': 'New York'}
print(myDictionary)