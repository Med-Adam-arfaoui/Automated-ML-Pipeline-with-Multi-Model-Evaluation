from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import os
import numpy as np
from core.config import MODEL_DIR

# ----------------------------------
# Define Models and Parameter Grids
# ----------------------------------
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10], 'penalty': ['l2']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    'SVM': (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
    'XGBoost': (xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}),
    'Naive Bayes': (GaussianNB(), {})
}

# ----------------------------------
# Train a Single Model
# ----------------------------------
def train_single_model(name, X, y, test_size=0.2, random_state=42):
    if name not in models:
        raise ValueError(f"Model '{name}' is not supported.")

    model, param_grid = models[name]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features for certain models
    scaler = StandardScaler()
    if name in ['Logistic Regression', 'KNN', 'SVM']:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_model, X_test_model = X_train_scaled, X_test_scaled
    else:
        # For models that don't need scaling, convert to numpy arrays
        X_train_model, X_test_model = X_train.values, X_test.values

    # Perform grid search if parameters exist
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted')
        grid.fit(X_train_model, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        best_model = model
        best_model.fit(X_train_model, y_train)
        best_params = None

    # Predictions and probabilities
    y_pred = best_model.predict(X_test_model)
    
    # Get probability scores for ROC curve (if available)
    y_proba = None
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_test_model)
    elif hasattr(best_model, 'decision_function'):
        y_proba = best_model.decision_function(X_test_model)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.joblib")
    joblib.dump(best_model, model_path)

    # Convert X_test_model to list - handle both numpy arrays and DataFrames
    if hasattr(X_test_model, 'tolist'):
        X_test_list = X_test_model.tolist()
    else:
        # If it's a DataFrame, convert to numpy first then to list
        X_test_list = X_test_model.values.tolist() if hasattr(X_test_model, 'values') else X_test_model.tolist()

    # Prepare return dictionary
    result = {
        'model_name': name,
        'best_params': best_params,
        'accuracy': acc,
        'f1_weighted': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'path': model_path,
        'X_test': X_test_list,
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }
    
    # Add probability scores if available (convert to list for JSON serialization)
    if y_proba is not None:
        result['y_proba'] = y_proba.tolist()
    
    return result