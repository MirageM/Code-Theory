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