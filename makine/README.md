===Machine Learning GUI App===


Machine Learning GUI App is a user-friendly application that allows you to train, test, and compare machine learning models without writing a code. 
Just upload your CSV datasets, select your target column, and visualize the results!

-FEATURES:

This project provides an end-to-end Machine Learning pipeline from data preprocessing to model evaluation:

*  Easy Data Loading: Supports CSV files and automatically detects columns.
*  Advanced Preprocessing:
    * One-Hot Encoding for categorical features.
    * StandardScaler or MinMaxScaler normalization options for numerical data.
    * Automatic Label Encoding for the target variable.
*  Supported Models:
    * Perceptron
    * Multi-Layer Perceptron (Artificial Neural Networks - MLP)
    * Decision Tree Classifier.
*  Flexible Configuration:
    * Customizable Hidden Layer structure for MLP (e.g., `100,50`) directly from the UI.
    * Adjustable Train/Test ratio via a slider (50% - 90%).
*  Detailed Analysis & Reporting:
    * Metrics: Accuracy, Precision, Recall, F1-Score.
    * Visualization: Confusion Matrix with using Seaborn.
    * Comparison Mode: Train all models sequentially and compare their performance in a unified table.

-INSTALLATION:

You must install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

-USAGE:

To launch the application, run the following command in your terminal:

```bash
python main.py


-TECHNOLOGIES:

Language: Python 3.10+

GUI: Tkinter

ML & Data: Scikit-learn, Pandas, Numpy

Visualization: Matplotlib, Seaborn
