from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, 
    AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier 

def get_models_dictionary():
    """
    Returns a dictionary of models where:
    Key = Model Name
    Value = Tuple of (Base_Model_Object, Parameter_Grid_Dictionary)
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {'C': [0.1, 1, 10]}
        ),
        
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        ),
        
        "Gaussian Naive Bayes": (
            GaussianNB(),
            {'var_smoothing': [1e-9, 1e-8]} # Minimal tuning for Naive Bayes
        ),
        
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {'max_depth': [10, 20, None], 'min_samples_split': [2, 10]}
        ),
        
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100, 150, 200], 'max_depth': [None, 20]}
        ),
        
        "Bagging Classifier": (
            BaggingClassifier(random_state=42),
            {'n_estimators': [50, 100, 150, 200]}
        ),
        
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.1, 1.0]}
        ),
        
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.1, 0.2]}
        ),
        
        "XGBoost": (
            XGBClassifier(eval_metric='logloss', random_state=42),
            {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]}
        ),
        
        # NOTE: SVM takes a VERY long time on large datasets. 
        # "SVM (Linear)": (
        #     SVC(kernel='linear', probability=True, random_state=42),
        #     {'C': [0.1, 1]}
        # ),
        # "SVM (RBF)": (
        #    SVC(kernel='rbf', probability=True, random_state=42),
        #    {'C': [0.1, 1], 'gamma': ['scale']}
        # )
    }
    
    return models