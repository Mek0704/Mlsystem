"""
Machine Learning GUI APP

This application provides an interface where users can train different machine learning models on CSV datasets and visualize the results.


Usage:
    python main.py

REQUIREMENTS:
    pip install -r requirements.txt
"""

#import app from GUI module
from gui.app_window import run_app


def main():
    """
    The main function that launches the application.
    """
    print("="*50)
    print("Machine Learning GUI App is being launched...")
    print("="*50)
    print()
    print("Supported Models:")
    print("  - Perceptron")
    print("  - Multi-Layer Perceptron (MLP)")
    print("  - Decision Tree")
    print()
    print("Supported Preprocesses:")
    print("  - One-Hot Encoding")
    print("  - StandardScaler Normalizasyonu")
    print("  - MinMaxScaler Normalizasyonu")
    print()
    print("="*50)
    
    # start GUI
    run_app()


if __name__ == "__main__":
    main()
