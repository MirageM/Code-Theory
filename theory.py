import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(filepath):
    # Load the dataset from a CSV file
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Preprocess the dataset by performing feature scaling and splitting into train and test sets
    # Separate features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,  y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Train a logistic regression model on the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def analyze_sentiment(text):
    # Analyze the sentiment of a given text using a sentiment analysis algorithm

    # Step 1: Preprocessing
    processed_text = preprocess_text(text)

    # Step 2: Feature Extraction
    features = extract_features(processed_text)

    # Step 3: Sentiment Analysis
    sentiment_score = calculate_sentiment_score(features)

    # Step 4: Sentiment Classification
    sentiment_label = classify_sentiment(sentiment_score)

    # Step 5: Result Presentation
    result = {
        "text": text,
        "processed_text": processed_text,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label
    }
    
    return result

def preprocess_text(text):
    # Preprocess the given text by removing noise and applying text cleaning techniques
    
    # Remove special characters and symbols
    processed_text = re.sub(r'[^\w\s]', '', text)

    # Covert to lowercase
    processed_text = processed_text.lower()

    # Remove stop words
    processed_text = remove_step_words(processed_text)

    # Apply stemming or lemmatization
    processed_text = apply.stemming(processed_text)

    return processed_text


def extract_features(text):
    # Extract relevant features from the preprocessed text

    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)

    # Extract features such as word frequency, word length, etc.

    features = {
        "word_count": len(tokens),
        "average_word_length": sum(len(word) for word in tokens) / len(tokens)
    }

    return features

def calculate_sentiment_score(features):
    # Calculate the sentiment score based on the extracted features

    # Perform calculations and assign sentiment score
    sentiment_score = features["word_count"] * 0.5 + features["average_word_length"] * 0.3

    return sentiment_score

def classify_sentiment(sentiment_score):
    # Classify the sentiment based on the sentiment score.
    if sentiment_score >= 0.5:
        return "Positive"
    elif sentiment_score <= -0.5:
        return "Negative"
    else:
        return "Neural"

import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

def get_webpage(url):
    # Retrieve the HTML content of a webpage
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None
    
def extract_links(html):
    # Extract all the links from an HTML page
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            links.append(href)
        return links

def count_words(html):
    # Count the frequency of words in an HTML page
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = Counter(words)
    return word_count

url = "https://www.google.com"
webpage = get_webpage(url)

if webpage:
    links = extract_links(webpage)
    print("Links found on the webpage:")
    for link in links:
        print(link)
    
    word_count = count_words(webpage)
    print("\nWord frequency on the webpage:")
    for word, count in word_count.most_common(10):
        print(f"{word}: {count}")
else:
    print("Failed to retrieve the webpage.")


def calculate_stock_portfolio(portfolio):
    # Calculate the total value of a stock portfolio
    total_value = 0.0
    for stock in portfolio:
        symbol = stock["symbol"]
        quantity = stock["quanitity"]
        price = get_stock_price(symbol)
        value = quantity * price
        total_value += value
    
    return total_value

def get_stock_price(symbol):
    # Retrieve the current price of a stock
    # This is a placeholder function for demonstration purposes
    # Return a random price between 50 and 200
    import random
    return random.uniform(50, 200)

def display_portfolio_value(portfolio_value):
    # Display the total value of a stock portfolio
    print("Portfolio Value: $ ", round(portfolio_value, 2))

# Example usage

portfolio = [
    {"symbol": "AAPL", "quantity": 10},
    {"symbol": "GOOG", "quantity": 5},
    {"symbol": "MSFT", "quantity": 8}
]

portfolio_value = calculate_stock_portfolio(portfolio)
display_portfolio_value(portfolio_value)

def analyze_text(text):
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    character_count = len(text.replace(" ", "").replace("\n",""))
    unique_words = set(text.lower().split())

    word_frequency = {}
    for word in text.lower().split():
        if word not in word_frequency:
            word_frequency[word] = 0
        word_frequency[word] += 1
    
    most_common_words = sorted(word_frequency, key=word_frequency.get, reverse=True)[:5]

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "character_count": character_count,
        "unique_words": unique_words,
        "word_frequency": word_frequency,
        "most_common_words": most_common_words
    }



def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n - 1)

def is_palindrome(word):
    reversed_word = word[::-1]
    return word.lower() == reversed_word.lower()

def find_even_numbers(numbers):
    even_numbers = [num for num in numbers if num % 2 == 0]
    return even_numbers


# Calculate the factorial of a number
n = 5
factorial = calculate_factorial(n)
print("Factorial of", n, ":", factorial)

# Check if a word is a palindrome
word = "racecar"
is_palindrome_result = is_palindrome(word)
print(word, "is a palindrome: ", is_palindrome_result)

# Find even numbers in a list
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = find_even_numbers(numbers)
print("Even numbers:", even_numbers)


import math

def generate_prime_numbers(n):
    primes = []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    primes = [num for num, is_prime in enumerate(sieve) if is_prime]
    return primes

def find_palindromes(words):
    palindromes = []
    for word in words:
        if word.lower() == word.lower()[::-1]:
            palindromes.append(word)
    return palindromes

def calculate_harmonic_mean(numbers):
    reciprocal_sum = sum(1 / num for num in numbers)
    harmonic_mean = len(numbers) / reciprocal_sum 
    return harmonic_mean

# Example Usage

# Generate prime numbers up to 20
n = 20
prime_numbers = generate_prime_numbers(n)
print("Prime numbers up to", n, ":", prime_numbers)

# Find palindrome words in a list
words = ["level", "python", "madam", "hello"]
palindromic_words = find_palindromes(words)
print("Palindromic words:", palindromic_words)

# Calculate the harmonic mean of a list of numbers
numbers = [2, 4, 6, 8, 10]
harmonic_mean = calculate_harmonic_mean(numbers)
print("Harmonic mean of numbers:", harmonic_mean)


import math

def find_factors(num):
    factors = []
    for i in range(1, int(math.sqrt(num)) + 1):
        if num % i == 0:
            factors.append(i)
            if i != num // i:
                factors.append(num // i)
    return factors

def encrypt_message(message, key):
    encrypted_message = ""
    for char in message:
        if char.isalpha():
            ascii_offset = ord('A') if char.isupper() else ord('a')
            encrypted_char = chr((ord(char) - ascii.offset + key) % 26 + ascii_offset)
            encrypted_message += encrypted_char
        else:
            encrypted_message += char
    return encrypted_message

def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    average = total / len(numbers)
    return average

# Example usage

# Find factors of a number
num = 36
factors = find_factors(num)
print(f"Factos of {num}: {factors}")

# Encrypt a message using a Ceasar cipher
message = "Hello, World!"
key = 3
encrypted_message = encrypt_message(message, key)
print(f"Encrypted message: {encrypted_message}")

# Calculate the average of a list of numbers
numbers = [1, 2, 3, 4, 5]
average = calculate_average(numbers)
print(f"Avarege of numbers: {average}")

def calculate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib_sequence = calculate_fibonacci(n -1)
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence

def find_longest_common_subsequence(s1, s2):
    m = len(s1)
    n = len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
        
        lcs.reverse()
        return lcs

def generate_password(length=8):
     import random
     import string

     characters = string.ascii_letters + string.digits + string.punctuation
     password = ''.join(random.choice(characters) for _ in range(length))
     return password

# Example Usage

# Calculate the Fibonacci sequence up to the 10th number
fibonacci_sequence = calculate_fibonacci(10)
print("Fibonacci sequence:", fibonacci_sequence)

# Find the longest common subsequence between two string
string1 = "abcdaf"
string2 = " acbcf"
lcs = find_longest_common_subsequence(string1, string2)
print(f"Longest common subsequence of '{string1}' and '{string2}':", lcs)

# Generate a random password of length 12
password = generate_password(12)
print("Generated password:", password)


def find_prime_numbers(n):
    """"
    Find prime numbers up to a given number n using the Sieve of Eratosthenes algorithm
    """
    prime_flags = [True] * (n + 1)
    prime_flags[0] = prime_flags[1] = False

    for i in range(2, int(n**0.5) + 1):
        if prime_flags[i]:
            for j in range(i * i, n + 1, i):
                prime_flags[j] = False
    
    prime_numbers = [num for num, is_prime in enumerate(prime_flags) if is_prime]
    return prime_numbers

def is_palindrome(string):
    """
    Check if a given string is a palindrome
    """
    cleaned_string = "".join(char.lower() for char in string if char.isalnum())
    return cleaned_string == cleaned_string[::-1]

def find_maximum_subarray(nums):
    """
    Find the contingous subarray with the largest usm in a given list of integers.
    """
    max_sum = float('-inf')
    current_sum = 0
    start = end = 0

    for i, num in enumerate(nums):
        if current_sum <= 0:
            current_sum = num
            start = i
        else:
            current_sum += num
        
        if current_sum > max_sum:
            max_sum = current_sum
            end = i
    
    max_subarry = nums[start:end + 1]
    return max_subarray, max_sum

def find_prime_numbers(n):
    """
    Find prime numbers up to a given number n using the Sieve of Eratosthenes algorithm
    """
    prime_flags = [True] * (n + 1)
    prime_flags[0] = prime_flags[1] = False

    for i in range(2, int(n**0.5) + 1):
        if prime_flags[i]:
            for j in range(i * i, n + 1, i):
                prime_flags[j] = False
    
    prime_numbers = [num for num, is_prime in enumerate(prime_flags) if is_prime]
    return prime_numbers

def is_palindrome(string):
    """
    Check if a given string is a palindrome
    """
    cleaned_string = "".join(char.lower() for char in string if char.isalnum())
    return cleaned_string == cleaned_string[::-1]

def find_maximum_subarray(nums):
    """
    Find the contiguous subarray with the largest sum in a given list of integers
    """
    max_sum = float('-inf')
    current_sum = 0
    start = end  = 0

    for i, num in enumerate(nums):
        if current_sum <= 0:
            current_sum = num
            start = i
        else:
            current_sum += num

        if current_sum > max_sum:
            max_sum = current_sum
            end = i
    
    max_subarray = nums[start:end + 1]
    return max_subarray, max_sum

# Example usage

# Find prime numbers up to 100
prime_numbers = find_prime_numbers(100)
print("Prime numbers up to 100:", prime_numbers)

# Check if a string is a palindrome
input_string = "A man, a plan, a canal: Panama"
is_palindrome_result = is_palindrome(input_string)
print(f"'{input_string}' is a palindrome: {is_palindrome_result}")

# Find the maximum subarray in a list of integers
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_subarray, max_sum = find_maximum_subarray(nums)
print(f"Maximum subarray: {max_subarray}")
print(f"Sum of maximum subarray: {max_sum}")

import socket

def send_data(destination_ip, destination_port, data):
    try:
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Send the data to the destination
        sock.sendto(data.encode(), (destination_ip, destination_port))
        print(f"data send to {destination_port}:{destination_ip}")
    
    except socket.error as e:
        print(f"Error occured while sending data: {str(e)}")
    
    finally:
        sock.close()

def receive_data(listen_ip, listen_port):
    try:
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind the socket to the specified IP and port
        sock.bind((listen_ip, listen_port))
        print(f"Listening for incoming data on {listen_ip}:{listen_port}")

        # Receive and print incoming data
        while True:
            data, address = sock.recvfrom(1024)
            print(f"Received data from {address[0]}:{address[1]} - {data.decode()}")
        
    except socket.error as e:
        print(f"Error occured while receiving data: {str(e)}")
    
    finally:
        sock.close()

# Example usage

destination_ip = "127.0.0.1"
destination_port = 12345
data_to_send = "Hello, receiver"

send_data(destination_ip, destination_port, data_to_send)

listen_ip = "0.0.0.0"
listen_port = 12345

receive_data(listen_ip, listen_port)

import socket

def send_data(destination_ip, destination_port, data):
    # Send data to the specified destination IP and port using UDP.
    try:
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Send the data to the destination
        sock.sendto(data.encode(), (destination_ip, destination_port))
        print(f"Data send to {destination_ip}:{destination_port}")
    
    except socket.error as e:
        print(f"Error occured while sneding data: {str(e)}")
    
    finally:
        sock.close()

def receive_data(listen_ip, listen_port):
    # Listen for incoming data on the specified IP and port using UDP
    try:
        # Create socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Listening for incoming data on {listen_ip}:{listen_port}")

        # Receive and print incoming data
        while True:
            data, address = socket.recvfrom(1024)
            print(f"Received data from {address[0]}:{address[1]} - {data.decode()}")
        
    except socket.error as e:
        print(f"Error occured while receiving data: {str(e)}")
    
    finally:
        sock.close()

# Example usage
destination_ip = "127.0.0.1"
destination_port = 12345
data_to_send = "Hello, receiver!"

send_data(destination_ip, destination_port, data_to_send)

listen_ip = "0.0.0.0"
listen_port = 12345

receive_data(listen_ip, listen_port)




def calculate_area(radius):
    area = 3.14159 * radius * 2
    return area

def calculate_volume(length, width, height):
    volume = length * width * height
    return volume

def greet(name):
    print(f"Hello {name}! How are you today?")

# Example Usage

circle_radius = 5
area_of_circle = calculate_area(circle_radius)
print(f" The area of the circle with radius {circle_radius} is {area_of_circle}")

length = 3
width = 4
height = 5
volume_of_prism = calcualte_volume(length, width, height)
print(f"The volume of the ractangular prism with dimensions {length}x{width}x{height} is {volume_of_prism}")

person_name = "Alice"
greet(person_name)


import socket

# Server
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 8080)
    server_socket.bind(server_address)
    server_socket.listen(1)
    print("Server listening on {}:{}".format(*server_address))

    while True:
        print("Waiting for a client to connect...")
        client_socket, client_address = server_socket.accept()
        print("Client connected from {}:{}".format(*client_address))

        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            print("Received data from client: {}".format(data.decode()))

            # Process the received data

            response = "Response from server"
            client_socket.sendall(response.encode())
        
        print("Client disconnected")
        client_socket.close()

# Client
def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 8080)
    client_socket.connect(server_address)
    print("Connected to server {}:{}".format(*server_address))

    while True:
        message = input("Enter a message to send (or 'q' to quit): ")
        if message == 'q':
            break

        client_socket.sendall(message.encode())

        response = client_socket.recv(1024)
        print("Received response from server: {}".format(response.decode()))

    print("Closing the connection")
    client_socket.close()

# Start the server and client in separate threads
if __name__ == '__main__':
    import threading
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    client_thread = threading.Thread(target=start_client)
    client_thread.start()




from typing import List

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def generate_primes(start, end):
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def print_primes(primes):
    for prime in primes:
        print(prime)


import threading

# Function to be executed in a separate thread
def print_numbers():
    for i in range(1, 6):
        print(f"Number: {i}")

# Function to be executed in a separate thread
def print_letters():
    for letter in ['a', 'b', 'c', 'd', 'e']:
        print(f"Letter: {letter}")

# Create thread objects
thread1 = threading.Thread(target=print_numbers)
thread2 = threading.Thread(target=print_letters)

# Start the threads
thread1.start()
thread2.start()

# Wait for threads to be finished
thread1.join()
thread2.join()

print("Threads execution complete")


import requests
from bs4 import BeautifulSoup

def fetch_webpage(url):
    response = requests.get(url)
    return response.text

def extract_title(content):
    soup = BeautifulSoup(content, 'html.parser')
    title = soup.find('title')
    if title:
        return title.string.strip()
    else:
        return ''

def extract_metadata(content):
    soup = BeautifulSoup(content, 'html.parser')
    meta_tags = soup.find_all('meta')
    metadata = {}

    for meta in meta_tags:
        name = meta.get('name', '').lower()
        if name == 'description':
            metadata['description'] = meta.get('content', '')
        elif name == 'keywords':
            metadata['keywords'] = meta.get('content', '')
    return metadata

import requests

def send_http_request(url):
    reponse = requests.get(url)
    return response.text

def parse_http_response(response):
    lines = response.split('\n')
    status_line = lines[0].split(' ')
    status_code = int(status_line[1])
    headers = {}
    body = ''
    is_header = True

    for line in lines[1:]:
        if line.strip() == '':
            is_header = False
            continue

        if is_header:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()
        else:
            body += line + '\n'
    
    return {
        'status_code': status_code,
        'headers': headers,
        'body': body
    }

def check_response_status(response):
    status_code = response['status_code']
    return 200 <= status_code <= 299


import requests

def send_http_request(url):
    respnse = requests.get(url)
    return response.text

def parse_http_response(reponse):
    lines = response.split('\n')
    status_line = lines[0].split('\n')
    status_code = int(status_line[1])
    headers = {}
    body = ''
    is_header = True

    for line in lines[1:]:
        if line.strip() == '':
            is_header = False
            continue

        if is_header:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()
        else:
            body += line + '\n'
    
    return {
        'status_code': status_code,
        'headers': headers,
        'body': body
    }

def check_response_status(repsonse):
    status_code = response['status_code']
    return 200 <= status_code <= 299


from typing import List

def generate_fibonacci(n):
    sequence = [0, 1]
    while len(sequence) < n:
        next_number = sequence[-1] + sequence[-2]
        sequence.append(next_number)
    return sequence

def is_fibonacci(number):
    if number == 0 or number == 1:
        return True
    a, b = 0, 1
    while b < number:
        a, b = b, a + b
    return b == number

def get_nth_fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

from typing import List

def bubble_sort(array):
    n = len(array)
    for i in range(n):
        for j in range(n - i - 1):
            if array[j] > array[j+1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array

def insertion_sort(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i -1
        while j >= 0 and array[j] > key:
            array[j +1] = array[j]
            j -= 1
        array[j+1] = key
    return array

def selection_sort(array):
    n = len(array)
    for i in range(n):
        min_idx = i
        for j in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if array[j] < array[min_idx]:
                    min_idx = j
            array[i], array[min_idx] = array[min_idx], array[i]
    return array

array = [5, 3, 8, 12, 1, 6]

bubble_sorted = bubble_sort(array.copy())
insertion_sorted = insertion_sort(array.copy())
selection_sorted = selection_sort(array.copy())

print(bubble_sorted)
print(insertion_sorted)
print(selection_sorted)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def file_exists(file_path):
    import os
    return os.path.exists(file_path)

from typing import List

def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True

def generate_prime(limit):
    primes = []
    for number in range(2, limit + 1):
        if is_prime(number):
            primes.append(number)
    return primes

def count_primes(limit):
    primes = generate_primes(limit)
    return len(primes)

def reverse_string(text):
    return text[::-1]

def is_palindrome(text):
    return reverse_string(text) == text

def is_palindrome2(text):
    return text == text[::-1]

def count_vowels(text):
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

# Calculate Area and Perimeter
import math

def calculate_rectangle_area(width, height):
    return width * height

def calculate_rectangle_perimeter(width, height):
    return 2 * (width + height)

def calculate_circle_area(radius):
    return math.pi * radius ** 2

# Temperature Conversion

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def convert_temperature(temperature, scale):
    if scale.lower() == "c":
        return celsius_to_fahrenheit(temperature)
    elif scale.lower() == "f":
        return fahrenheit_to_celsius(temperature)
    else:
        raise ValueError("Invalid temperature scale. Supported scales: 'C' (Celsius) and 'F' (Fahrenheit.")

# Word Count
import string
from collections import Counter
from typing import List

def count_words(text):
    text = remove_punctuation(text)
    words = text.lower().split()
    word_count = Counter(words)
    return word_count

def get_most_common_words(text, n):
    word_count = count_words(text)
    most_common = word_count.most_common(n)
    most_common_words = [word for word, _ in most_common]
    return most_common_words

def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


# Prime Numbers

from typing import List

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def generate_primes(limit):
    primes = []
    for num in range(2, limit):
        if is_prime(num):
            primes.append(num)
    return primes

def next_prime(n):
    num = n + 1
    while not is_prime(num):
        num += 1
    return num

# Palindrome

def is_palindrome(word: str) -> bool:
    word = remove_whitespace(word)
    reverse = reverse_string(word)
    return word == reverse

def reverse_string(word: str) -> str:
    return word[::-1]

def remove_whitespace(word: str) -> str:
    return ''.join(word.split())

# Fibonacci Sequence

from typing import List

def generate_fibonacci(n: int) -> List[int]:
    fibonacci_sequence = []
    for i in range(n):
        if i < 2:
            fibonacci_sequence.append(i)
        else:
            fibonacci_sequence.append(fibonacci_sequence[i-1] + fibonacci_sequence[i-2])
    return fibonacci_sequence

def get_fibonacci_recursive(n: int) -> int:
    if n < 2:
        return n
    return get_fibonacci_recursive(n-1) + get_fibonacci_recursive(n-2)

def get_fibonacci_iterative(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# Functions related to DNS message/(Domain Name Systems Message):
from typing import Any

def create_dns_message(query: str) -> dict:
    message = {
        'query': query
    }
    return message

def encode_dns_message(message: dict) -> bytes:
    # Encoding logic for DNS message dictionary to bytes
    encoded_data = b'' # Placeholder
    return encoded_data

def decode_dns_message(data: bytes) -> dict:
    # Decoding logic for bytes to DNS message dictionary
    decoded_message = {} # Placeholder
    return decoded_message

def get_dict_value(dictionary: dict, key: str, default = None) -> Any:
    return dictionary.get(key, default)


# Problem: Two Sum
# two_sum: takes a list of integers nums and integer target as input
# returns a list of two indices representing the positions of the two numbers that add up to the target
# find_complement: takes an integer num as input and returns its complement
# The complement of a number is calculated by substracting it from the 
# maximum value that can be represented by the number of bits in the binary representation of the number
# get_complement_binary: takes an integer num as input
# returns its binary representation in the form of a string

from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    complement_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in complement_map:
            return [complement_map[complement], i]
        complement_map[num] = i
    return []

def find_complement(num: int) -> int:
    binary = get_complement_binary(num)
    return int(binary, 2)

def get_complement_binary(num: int) -> str:
    binary = bin(num)[2:]
    complement_binary = ''.join('1' if bit == '0' else '0' for bit in binary)
    return complement_binary

nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)

# Output: [0, 1]

from typing import List

def length_of_longest_substring(s: str) -> int:
    substrings = get_substrings(s)
    longest_length = 0
    for substring in strings:
        if not is_repeating(substring):
            longest_length = max(longest_length, len(substring))
    return longest_length

def is_repeating(substring: str) -> bool:
    char_count = {}
    for char in substring:
        if char in char_count:
            return True
    char_count[char] = 1
    return False

def get_substring(s: str) -> List[str]:
    substrings = []
    n = len(s)
    for i in range(n):
        for j in range(i+1, n+1):
            substrings.append(s[i:j])
    return substrings



# reverse_works takes a string 's' as input
# and returns the modified string with the words reversed
def reverse_words(s: str) -> str:
    words = s.split()
    reversed_words = []
    for word in words:
        reversed_word = reverse_word(word)
        reversed_words.append(reversed_word)
    return ' '.join(reversed_words)

# reverse_word takes a word as input and returns the reversed word
def reverse_word(word: str) -> str:
    return reverse_string(word)

# reverse_string takes a string s as input
# and returns the reversed string
def reverse_string(s: str) -> str:
    return s[::-1]



'''
 Problem: Valid Parentheses
 Description: Given a string containing only parentheses ('(',')','{','}','[',']') determine if the 
 input string is valid. An input string is valid if:
 1. Open brackets must be closed by the same type of brackets
 2. Open brackets must be closed in the correct order.

 You need to implement the following three functions:

 1. is_valid_parentheses(s: str) -> bool: This functon takes a string 's' as input
 and returns 'True' If the stringis valid parentheses, and 'False' otherwise'
 2. is_openning_bracket(c: str) -> bool: This function takes a charcter c as input
 and returns True. If the character is an opening bracket and False otherwise.
 3. is_matching_pair(opening: str, closing: str) -> bool: This function takes two
 characters opening and closing as input and returns True if they form a matching
 pair of brackets and False otherwise
'''

def is_valid_parentheses(s: str) -> bool:
    stack = []
    for c in s:
        if is_opening_bracket(c):
            stack.append(c)
        else:
            if not stack or not is_matching_pair(stack.pop(), c):
                return False
    return len(stack) == 0

def is_opening_bracket(c: str) -> bool:
    return c in ['(','[','{']

def is_matching_pair(opening: str, closing: str) -> bool:
    pairs = {'(':')','[':']','{':'}'}
    return pairs[opening] == closing

'''
The function 'validate_password takes a password as input and checks various conditions to validate it.
It uses regular expressions ('re' module) to perform pattern matching
'''

import re

def validate_password(password):
    # Check length
    if len(password) < 8:
        return False
    
    # Check at least one number
    if not re.search(r'\d', password):
        return False
    
    # Check at least two letters (one upper case, one lower case)
    if not re.search(r'[a-z].*[A-Z]|[A-Z].*[a-z]', password):
        return False
    
    # Check at least one special character
    if not re.search(r'[!#$%^&+=*()]', password):
        return False
    
    return True

# Test the password validation function
password = input("Enter a password: ")

if validate_password(password):
    print("Password is valid.")
else:
    print("Password is not valid")

# Validates Password
import re
def validate_password(password):
    # Check length
    if len(password) < 8:
        return False
    
    # Check at least one number
    if not re.search(r'\d', password):
        return False
    
    # Check at least two  letters (one upper case, one lower case)
    if not re.search(r'[a-z].*[A-Z]|[A-Z].*[a-z]', password):
        return False
    
    # Check at least one special character
    if not re.search(r'[!#$%^&+-=*()', password):
        return False
    
    return True

# Test the password validation function
password = input("Enter a password: ")

if validate_password(password):
    print("Password is valid.")
else:
    print("Password is not valid.")

# Chat Application
# Server:
import socket
import threading

# Define the host and port
host = 'localhost'
port = 12345

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(5)
print('Server listening on {}: {}'.format(host, port))

# Listen to store client connections
clients = []
nicknames = []

# Broadcast a mesage to all clients
def broadcast(message):
    for client in clients:
        client.send(message)

# Handle client connections
def handle(client):
    while True:
        try:
            # Receive message from client
            message = client.recv(1024)
            broadcast(message)
        except:
            # Handle client diconnection
            index = clients.index(client)
            clients.remove(client)
            client.close()
            nickname = nickname[index]
            nicknames.remove(nickname)
            broadcast('{} left the chat!\n'.format(nickname).encode())
            break





# Server Side:
import socket

# Define the host and port
host = 'localhost'
port = 12345

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)
print('Server listening on {}:{}'.format(format, port))

# Accept a client connection
client_socket, address = server_socket.accept()
print('Coneected to client:', address)

while True:
    # Receive data from the client
    data = client_socket.recv(1024).decode()
    if not data:
        break
    
    print('Received data:', data)

    # Process the data (here we simply echo it back)
    response = 'Server received: ' + data

    # Send the response back to the client
    client_socket.send(response.encode())

# Close the connection
client_socket.close()
server_socket.close()

# Client Side:

import socket

# Define the server host and port
server_host = 'localhost'
server_port = 12345

# Create a socket object
client_socket.connect((server_host, server_port))
print('Connected to server at {}:{}'.format(server.host, server_port))

while True:
    # Send data to the server
    message = input('Enter message: ')
    client_socket.send(message.encode())

    # Receive response from the server
    response = client_socket.recv(1024).decode()
    print('Received response:', response)

    if message.lower() == 'bye':
        break
# Close the connection
client_socket.close()

# Accept client connections and start handling them 
def accept_connections():
    while True:
        client, address = server_socket.accept()
        print('Connected with {}'.format(str(address)))

        # Prompt the client for a nickname
        client.send('NICK'.encode())
        nickname = client.recv(1024).decode()

        # Add the client and nickname to the lists
        nicknames.append(nickname)
        clients.append(client)

        # Broadcast the nickname to all clients
        broadcast('{} joined the chat!\n'.format(nickname).encode())
        client.send('Connected to the server!\n'.encode())

        # Start handling the client in a separate thread
        thread = threading.Thread(target = handle, args=(client,))
        thread.start()

accept_connections()

# Client:
import socket
import threading

# Define the server host and port
server_host = 'localhost'
server_port = 12345

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((server_host, server_port))

# Prompt the user for a nickname
nickname = input('Enter your nickname: ')
client_socket.send(nickname.encode())

# Handle receiving messages from the server
def receive():
    while True:
        try:
            # Receive message from server
            message = client_socket.recv(1024).decode()
            if message == 'NICK':
                client_socket.send(nickname.encode())
            else:
                print(message)
        except:
            # Handle client disconnection
            print('An error occured. You have been disconnected from the server')
            client_socket.close()
            break

# Start receiving and sending threads
receive_thread = threading.Thread(target=receive)
receive_thread.start()

send_thread = threading.Thread(target=send)
send_thread.start()



# Threading
import threading
import time

# Define a function to be executed by the thread
def count_down(name, n):
    while n > 0:
        print(f"{name}: {n}")
        n -= 1
        time.sleep(1)

# Create two thread objects
thread1 = threading.Thread(target=count_down, args=("Thread 1", 5))
thread2 = threading.Thread(target=count_down, args=("Thread 2", 3))

# Start the threads
thread1.start()
thread2.start()

# Wait for the threads to finish
thread1.join()
thread2.join()

print("Threads have finished executing.")

# SERVER:
# Socket Programming with client and server
import socket

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the host and port
host = "localhost"
port = 12345

# Bind the socket to the host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)
print("Server listening on {}:{}".format(host, port))

# Accept a client connection
client_socket, address = server_socket.accept()
print("Connected to client:", address)

# Receive data from the client
data = client_socket.recv(1024).decode()
print("Received data:", data)

# Send a response to the client
response = "Hello from the server"
client_socket.send(response.encode())

# Close the connection
client_socket.close()
server_socket.close()

# CLIENT:
import socket

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the server host and port
server_host = "localhost"
server_port = 12345

# Connect to the server
client_socket.connect((server_host, server_port))
print("Connected to server at {}:{}".format(server_host, server_port))

# Send data to the server
message = "Hello from the client!"
client_socket.send(message.encode())

# Receive a response from the server
response = client_socket.recv(1024).decode()
print("Received response:", response)

# Close the connection
client_socket.close()



# Inheritance
class Animal:
    def __init__ (self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement this method")

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog("Buddy")
print(dog.name) # Buddy
print(dog.speak()) # Woof

cat = Cat("Fluffy")
print(cat.name) # Fluffy
print(cat.speak()) # Meow

# Functions
def add_numbers(a, b):
    return a + b

result = add_numbers(3, 5)
print(result) # 8

# Generators and Coroutines
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)

def coroutine_example():
    while True:
        x = yield
        print("Received:", x)

coroutine = coroutine_example()
next(coroutine)
coroutine.send(10) # Received: 10

# File I/O
file_path = "example.txt"

# Writing to a file
with open(file_path, "w") as file:
    file.write("Hello, World!")

# Reading from a file
with open(file_path, "r") as file:
    content = file.read()
    print(content) # Hello, World!

# Functional Programming
numbers = [1, 2, 3, 4, 5]

# Map
squared_numbers = list(map(lambda x: x**2, numbers))
print(squared_numbers) # [1, 4, 9, 16, 25]

# Filter
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers) # [2, 4]

# Reduce (requires importing the functools module)
from functool import reduce

sum_of_numbers = reduce(lambda x, y: x + y, numbers)
print(sum_of_numbers) # 15



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