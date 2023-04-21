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