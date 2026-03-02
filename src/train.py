import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocess import load_and_preprocess_data

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    # Ensure a directory exists for saving models
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create a full pipeline that includes the preprocessor and the model
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train
        full_pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = full_pipeline.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Confusion_Matrix': cm
        }
        
        # Save model
        filename = f"models/{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(full_pipeline, filename)
        print(f"Saved to {filename}\\n")
        
    return results
    
if __name__ == "__main__":
    filepath = '../customer_churn_datasest.csv'
    if not os.path.exists(filepath):
        filepath = 'customer_churn_datasest.csv'
        
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(filepath)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    
    for name, metrics in results.items():
        print(f"=== {name} ===")
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1 Score:  {metrics['F1']:.4f}")
        print(f"Confusion Matrix:\\n{metrics['Confusion_Matrix']}")
        print()
