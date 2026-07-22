import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ==========================================
# 1. Train and Evaluate Model
# ==========================================
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Trains the model and calculates all evaluation metrics."""
    
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Get probability scores for ROC AUC and PR AUC if available
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = test_pred
        
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score) 
    
    # Calculate metrics
    results = {
        "Trained Model": model,
        "Model": model_name,
        "Train Accuracy": accuracy_score(y_train, train_pred) * 100,
        "Test Accuracy": accuracy_score(y_test, test_pred) * 100,
        "Precision": precision_score(y_test, test_pred) * 100,
        "Recall": recall_score(y_test, test_pred) * 100,
        "F1 Score": f1_score(y_test, test_pred) * 100,
        "ROC AUC": roc_auc * 100,
        "PR AUC": pr_auc * 100,
        "Training Time (sec)": training_time
    }
    
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    print(f"Train Accuracy : {results['Train Accuracy']:.2f}%")
    print(f"Test Accuracy  : {results['Test Accuracy']:.2f}%")
    print(f"Precision      : {results['Precision']:.2f}%")
    print(f"Recall         : {results['Recall']:.2f}%")
    print(f"F1 Score       : {results['F1 Score']:.2f}%")
    print(f"ROC AUC        : {results['ROC AUC']:.2f}%")
    print(f"PR AUC         : {results['PR AUC']:.2f}%")
    print(f"Training Time  : {training_time:.4f} seconds\n")

    print("\nClassification Report\n")
    report =classification_report(y_test,test_pred,output_dict=True)
    report_df =pd.DataFrame(report).transpose()
    for column in ["precision","recall","f1-score"]:
        if column in report_df.columns:
            report_df[column]=(report_df[column] * 100)
    print(report_df.round(2))

    print("\nConfusion Matrix\n")
    cm=confusion_matrix(y_test,test_pred)
    print(cm)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Legitimate", "Phishing"])
    disp.plot()
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
    return results


# ==========================================
# 2. Hyperparameter Tuning
# ==========================================
def tune_model(model, parameters, X_train, y_train):
    """
    Performs Hyperparameter tuning using GridSearchCV. 
    Scoring is set to 'average_precision' (PR-AUC) due to imbalanced dataset.
    """
    grid = GridSearchCV(
        estimator=model, 
        param_grid=parameters, 
        cv=5, 
        scoring="average_precision",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    print("\nBest Parameters found:", grid.best_params_)
    print(f"Best PR-AUC CV Score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


