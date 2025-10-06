""" Python Basics & Setup refers to the fundamental process of installing Python, configuring the development environment, and understanding core programming concepts. This includes:

Installation: Downloading and installing Python interpreter

Environment Setup: Configuring IDEs (VS Code, PyCharm), Jupyter notebooks, or text editors

Syntax Fundamentals: Variables, data types, operators, control structures

Basic Operations: Input/output, basic calculations, string operations

Module Management: Using pip for package installation and import statements

The essence is establishing a working Python environment and grasping the foundational syntax that enables problem-solving through code."""

# Python Basics & Setup - Toy Example
# Basic Calculator with User Input

def simple_calculator():
    """A simple calculator demonstrating Python basics"""
    
    # Get user input
    print("=== Simple Python Calculator ===")
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    operation = input("Enter operation (+, -, *, /): ")
    
    # Perform calculation based on operation
    if operation == '+':
        result = num1 + num2
    elif operation == '-':
        result = num1 - num2
    elif operation == '*':
        result = num1 * num2
    elif operation == '/':
        if num2 != 0:
            result = num1 / num2
        else:
            result = "Error: Division by zero!"
    else:
        result = "Error: Invalid operation!"
    
    # Display result
    print(f"Result: {num1} {operation} {num2} = {result}")

# Demonstrate basic Python concepts
def demonstrate_basics():
    """Show various Python basic concepts"""
    
    # Variables and data types
    name = "Python Learner"
    age = 25
    height = 5.9
    is_beginner = True
    
    # Lists and loops
    numbers = [1, 2, 3, 4, 5]
    squared_numbers = [x**2 for x in numbers]
    
    # Dictionary
    student = {
        "name": name,
        "age": age,
        "courses": ["Math", "Science", "Programming"]
    }
    
    print("\n=== Python Basics Demonstration ===")
    print(f"Name: {name}, Age: {age}, Height: {height}")
    print(f"Is Beginner: {is_beginner}")
    print(f"Numbers: {numbers}")
    print(f"Squared Numbers: {squared_numbers}")
    print(f"Student Info: {student}")

# Run the examples
if __name__ == "__main__":
    simple_calculator()
    demonstrate_basics()