import threading


# Function to calculate the factorial of a given integer using recursion
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


# Function to call the factorial function using threads
def threaded_factorial(n):
    # Create a thread for the factorial function
    t = threading.Thread(target=factorial, args=[n])
    # Start the thread
    t.start()
    # Wait for the thread to finish
    t.join()
    # Print the result
    print("Factorial of", n, "is", factorial(n))


# Example usage
threaded_factorial(5)
