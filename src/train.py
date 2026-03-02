import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocess import load_and_preprocess_data

from sklearn.model_selection import GridSearchCV

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__max_depth': [3, 5, 10, None],
                'classifier__min_samples_leaf': [1, 5, 10]
            }
        },
        'MLP': {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'params': {
                'classifier__hidden_layer_sizes': [(50,), (100,)],
                'classifier__alpha': [0.0001, 0.001]
            }
        }
    }
    
    results = {}
    
    # Ensure a directory exists for saving models
    os.makedirs('models', exist_ok=True)
    
    for name, config in models.items():
        print(f"Training and Tuning {name}...")
        
        # Create a full pipeline that includes the preprocessor and the model
        base_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=config['params'],
            scoring='f1',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        # Train and tune
        grid_search.fit(X_train, y_train)
        
        # Use the best pipeline found
        best_pipeline = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        
        # Predict
        y_pred = best_pipeline.predict(X_test)
        
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
            'Confusion_Matrix': cm,
            'Best_Params': grid_search.best_params_
        }
        
        # Save best model
        filename = f"models/{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(best_pipeline, filename)
        print(f"Saved optimized model to {filename}\n")
        
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
