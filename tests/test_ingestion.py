import os
from packages.ingestion import run_ingestion_pipeline, cleanup_repository

def create_synthetic_py_lib(target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    
    # pyproject.toml
    with open(os.path.join(target_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "mathlib"\nversion = "0.1.0"\ndependencies = ["numpy>=1.20.0"]')
    
    # src/calculator.py
    os.makedirs(os.path.join(target_dir, "src"), exist_ok=True)
    with open(os.path.join(target_dir, "src", "calculator.py"), "w") as f:
        f.write('''
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
''')

def test_ingestion_synthetic():
    target = "./evals/datasets/python_library"
    create_synthetic_py_lib(target)
    
    manifest, files = run_ingestion_pipeline(target)
    assert manifest.file_count >= 2
    assert "Python" in manifest.languages
    assert manifest.archetype.value in ("LIBRARY", "UNKNOWN")
    cleanup_repository(manifest.local_path)
