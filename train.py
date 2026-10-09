import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_and_evaluate():
    print("[+] Loading Dataset: Crop_recommendation.csv ...")
    df = pd.read_csv('Crop_recommendation.csv')
    
    print(f"[+] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[+] Unique Crops ({len(df['label'].unique())}): {list(df['label'].unique())}\n")
    
    # Feature & Target Separation
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Feature Statistics / Bounds for input validation
    feature_bounds = {}
    for col in X.columns:
        feature_bounds[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        }
        
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define Models to Compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'Support Vector Machine (SVM)': SVC(kernel='rbf', C=1.0),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    best_model_name = None
    best_accuracy = 0
    best_model_obj = None
    
    print("=" * 60)
    print(" MODEL COMPARISON & EVALUATION")
    print("=" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        results[name] = {
            'Test Accuracy': round(acc * 100, 2),
            'CV Accuracy Mean': round(cv_scores.mean() * 100, 2),
            'CV Accuracy Std': round(cv_scores.std() * 100, 2)
        }
        
        print(f"* {name}:")
        print(f"   - Test Accuracy : {acc * 100:.2f}%")
        print(f"   - 5-Fold CV Acc : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
        print("-" * 60)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_obj = model
            
    print(f"\n[!] Best Performing Model: {best_model_name} ({best_accuracy * 100:.2f}% Accuracy)\n")
    
    # Full Classification Report for Best Model
    y_pred_best = best_model_obj.predict(X_test)
    print("Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_best))
    
    # Export Model and Metadata
    print("[+] Saving best model to 'crop_model.pkl'...")
    with open('crop_model.pkl', 'wb') as f:
        pickle.dump(best_model_obj, f)
        
    # Save Metadata for App Input Validation & Interview references
    metadata = {
        'best_model': best_model_name,
        'accuracy': float(best_accuracy),
        'model_comparison': results,
        'feature_names': list(X.columns),
        'crop_classes': list(np.unique(y)),
        'feature_bounds': feature_bounds
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("[+] Model & Metadata successfully saved!")

if __name__ == '__main__':
    train_and_evaluate()
