
import os

class Calculator:
    """Basic math calculator class."""
    def add(self, a: float, b: float) -> float:
        return a + b
    
    def multiply(self, a: float, b: float) -> float:
        return a * b

def main():
    calc = Calculator()
    print("Add 2+3:", calc.add(2, 3))

if __name__ == "__main__":
    main()
