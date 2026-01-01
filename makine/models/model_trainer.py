"""
Model training module

This module handles the initialization, training, and prediction
steps for various machine learning models.
"""

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model(model_name, X_train, y_train, **kwargs):
    """
    Trains the selected model.
    
    Parameters:
        model_name (str): Name of the model ('Perceptron', 'MLP', 'Decision Tree')
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        **kwargs: Arbitrary keyword arguments (e.g., hidden_layers for MLP)
        
    Returns:
        model: Trained sklearn model object
    """
    
    # Initialize the model variable
    model = None
    
    print(f"\n{model_name} model is being initialized...")
    
    if model_name == "Perceptron":
        # Perceptron does not use hidden_layers, so we don't pass kwargs to it to avoid TypeError
        model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
        
    elif model_name == "MLP":
        # Extract hidden_layers from kwargs, default to (100,) if not provided
        # This is the safe way to handle optional parameters
        hidden_layers = kwargs.get('hidden_layers', (100,))
        
        print(f"  -> MLP Configuration: Hidden Layers={hidden_layers}")
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
    elif model_name == "Decision Tree":
        # Decision Tree usually doesn't need hidden_layers either
        model = DecisionTreeClassifier(criterion='entropy', random_state=42)
        
    else:
        raise ValueError(f"Model not supported: {model_name}")
    
    # Train the model
    print(f"{model_name} is training...")
    model.fit(X_train, y_train)
    print("Training completed.")
    
    return model


def train_and_predict(model_name, X_train, X_test, y_train, **kwargs):
    """
    Trains the model and makes predictions on test data.
    
    Parameters:
        model_name (str): Name of the model
        X_train, X_test, y_train: Data splits
        **kwargs: Extra arguments passed to train_model
        
    Returns:
        tuple: (trained_model, y_pred)
    """
    
    # This ensures that hidden_layers is passed along only if it exists
    model = train_model(model_name, X_train, y_train, **kwargs)
    
    # Prediction
    print("Prediction is being performed on test data...")
    y_pred = model.predict(X_test)
    
    return model, y_pred