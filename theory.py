'''
__init__ is a special method in Python classes that is automatically called when an object is created from the class.
It is used to initialize the attributes (variables) of the object with default values, or with values passed as arguments.
___________________________________________________________________________________________________________________________
The class Person with an __init__ method takes two parameters name and age. Inside the method, we set the values of two instance variables
(self.name and self.age) to the values passed as arguments
When a new Person object was created and the values name and age was passed to the constructor,
The constructor automatically calls the __init__ method to initialize the obect's attributes
In the __init__ is used to set up all the initial state of an object when it is created. 
It is one of the most commonly used methods in Python classes, and it is called automatically when a new object is created.
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