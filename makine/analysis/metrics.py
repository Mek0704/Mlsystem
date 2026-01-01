"""
Analysis and reporting module

This module calculates model performance metrics and
performs confusion matrix visualization.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_accuracy(y_test, y_pred):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Parameters:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): predicted labels
        
    Transformes to the:
        float: Accuracy score (0.0 - 1.0)
    """
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def calculate_precision(y_test, y_pred):
    """
    Precision = TP / (TP + FP)
    
    Parameters:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): predicted labels
        
    Transformes to the:
        float: Precision score (0.0 - 1.0)
    """
    # average=‘weighted’ calculates the weighted average for multiple classes
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    return precision


def calculate_recall(y_test, y_pred):
    """
    Recall = TP / (TP + FN)
    
    Parameters:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): predicted labels
        
    Transformes to the:
        float: Recall score (0.0 - 1.0)
    """
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    return recall


def calculate_f1(y_test, y_pred):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Parameters:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): predicted labels
        
    Transformes to the:
        float: F1 score (0.0 - 1.0)
    """
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return f1


def get_all_metrics(y_test, y_pred):
    """
    Gets all metrics and transformes to the dictionary
    
    Parameters:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): predicted labels
        
    Transformes to the:
        dict: all metrics
    """
    metrics = {
        'Accuracy': calculate_accuracy(y_test, y_pred),
        'Precision': calculate_precision(y_test, y_pred),
        'Recall': calculate_recall(y_test, y_pred),
        'F1-Score': calculate_f1(y_test, y_pred)
    }
    
    # printing results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f} ({metric_value*100:.2f}%)")
    print("="*50)
    
    return metrics


def create_confusion_matrix(y_test, y_pred):
    """
    Parameters:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): predicted labels
        
    Transformes to the:
        numpy.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix (Contingency Table):")
    print(cm)
    
    return cm


def plot_confusion_matrix(y_test, y_pred, class_names=None):
    """
    Visulizes Confusion Matrix
    
    Parametreler:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): predicted labels
        class_names (list): Class names (optional)
    """
    # calculate confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    
    #if there is no class names then use number
    if class_names is None:
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        class_names = [f"Class {i}" for i in unique_labels]
    
    # making figure
    plt.figure(figsize=(8, 6))
    
    # draw heatmap 
    sns.heatmap(
        cm,
        annot=True,           # display values
        fmt='d',              # integer format
        cmap='Blues',         # colour
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Labels
    plt.title('Confusion Matrix(Contingency Table)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    
    # Shows
    plt.tight_layout()
    plt.show()


def generate_report(y_test, y_pred, model_name, class_names=None):
    """
    Creates a comprehensive results report
    
    Parameters:
        y_test (numpy.ndarray): Real labels
        y_pred (numpy.ndarray): Predicted labels
        model_name (str): Choosen model name
        class_names (list): Class names
        
    Transformes to the:
        str: Report txt
    """
    # calculate metrics 
    metrics = get_all_metrics(y_test, y_pred)
    cm = create_confusion_matrix(y_test, y_pred)
    
    # Create the report text
    report = []
    report.append("="*60)
    report.append("           MODEL RESULTS REPORT")
    report.append("="*60)
    report.append(f"\nUsed Model: {model_name}")
    report.append(f"Test Data Number: {len(y_test)}")
    report.append("")
    report.append("-"*40)
    report.append("PERFORMANCE METRICS")
    report.append("-"*40)
    
    for metric_name, metric_value in metrics.items():
        percentage = metric_value * 100
        report.append(f"  {metric_name:12}: {metric_value:.4f} ({percentage:.2f}%)")
    
    report.append("")
    report.append("-"*40)
    report.append("CONFUSION MATRIX")
    report.append("-"*40)
    
    # Add Confusion matrix as string
    for row in cm:
        report.append("  " + "  ".join([f"{val:4d}" for val in row]))
    
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    return report_text, metrics
