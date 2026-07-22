import pandas as pd
import joblib

from Data_Preprocessing import load_and_preprocess_data
from models_config import get_models_dictionary
from train_evaluate import tune_model, train_and_evaluate

def test_compare():
    print("🚀 Starting Model Training and Tuning Pipeline...\n")
    
    X_train_original, X_test_original, y_train, y_test, X_train_scaled, X_test_scaled, scaler = load_and_preprocess_data("dataset_phishing.csv")
    
    # Export expected features for the GUI
    expected_features = list(X_train_original.columns)
    joblib.dump(expected_features, 'expected_features.pkl')

    models_dict = get_models_dictionary()
    results_list = []
    
    for model_name, (base_model, parameters) in models_dict.items():
        print(f"\n{'*'*60}")
        print(f"⚙️ Processing Model: {model_name}")
        print(f"{'*'*60}")
        
        if model_name in ["Logistic Regression", "K-Nearest Neighbors"]:
            X_train_curr = X_train_scaled
            X_test_curr = X_test_scaled
            print("[Info] Using Scaled Data")
        else:
            X_train_curr = X_train_original
            X_test_curr = X_test_original
            print("[Info] Using Original Data")
            
        print(f"⏳ Tuning Hyperparameters for {model_name}...")
        tuned_model = tune_model(
            model=base_model,
            parameters=parameters,
            X_train=X_train_curr,
            y_train=y_train
        )
        
        print(f"📊 Evaluating the Tuned {model_name}...")
        metrics = train_and_evaluate(
            model=tuned_model,
            model_name=model_name,
            X_train=X_train_curr,
            X_test=X_test_curr,
            y_train=y_train,
            y_test=y_test
        )
        results_list.append(metrics)

    print("\n" + "="*90)
    print("🏆 FINAL MODELS COMPARISON 🏆")
    print("="*90)
    
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values(
        by=["PR AUC", "Recall", "F1 Score"],
        ascending=[False, False, False]
    ).round(2).reset_index(drop=True)
    
    display_df = comparison_df.drop(columns=['Trained Model'], errors='ignore')
    print(display_df.to_string())
    print("="*90 + "\n")
    
    best_model_name = comparison_df.iloc[0]['Model']
    print(f"🎉 The Winner is: {best_model_name} with a PR-AUC of {comparison_df.iloc[0]['PR AUC']}%")

    model_filename = 'best_model.pkl'
    joblib.dump(comparison_df.iloc[0]['Trained Model'], model_filename)
    print(f"Model successfully trained and saved as {model_filename}")

if __name__ == "__main__":
    test_compare()