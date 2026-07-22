# 🔐 URL Phishing Detector

A Machine Learning-based system that detects whether a URL is **Legitimate** or **Phishing**.

The project analyzes URL-based features and applies multiple Machine Learning algorithms to identify suspicious URLs and help users avoid malicious websites.

---

## 📌 About the Project

Phishing attacks use malicious URLs to trick users into visiting fake or harmful websites.

This project aims to build a Machine Learning-based system that analyzes a URL and predicts whether it is:

- 🟢 Legitimate
- 🔴 Phishing

The project also compares different Machine Learning models to identify the best-performing model.

---

## 🚀 Project Workflow

1. Data Loading and Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Analysis
4. Handling Imbalanced Data
5. Data Splitting and Scaling
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation and Comparison
9. Feature Extraction
10. Final Model Integration
11. User Interface

---

## 📊 Dataset

The dataset contains legitimate and phishing URLs represented by numerical features extracted from URL characteristics.

Examples include:

- URL Length
- Domain Length
- Number of Subdomains
- Digits and Special Characters
- HTTPS Usage
- URL Entropy
- Path and Query Length
- IP Address Detection

### Target Classes

| Label | Class |
|------|------|
| 0 | Legitimate |
| 1 | Phishing |

Since the dataset is imbalanced, the project focuses on **Precision, Recall, F1-Score, and ROC-AUC**, in addition to Accuracy.

---

## 🤖 Machine Learning Models

The project experiments with:

### Baseline Models
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)

### Advanced Models
- Decision Tree
- Support Vector Machine (SVM)
  - Linear
  - RBF
  - Polynomial

### Boosting Models
- AdaBoost
- Gradient Boosting

### Ensemble Models
- Random Forest

---

## ⚙️ Model Optimization

The project uses:

- GridSearchCV
- Cross-Validation

These techniques are used to optimize hyperparameters, analyze model performance, and reduce overfitting.

---

## 📈 Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Special attention is given to detecting the **Phishing** class correctly.

---

## 🔍 Feature Extraction

When a user enters a new URL, the system extracts the same features used during model training and passes them to the trained model.

The final prediction is:

🟢 **Legitimate**

or

🔴 **Phishing**

---

## 🖥️ User Interface

The project includes a user interface that allows users to enter a URL and receive a prediction from the trained Machine Learning model.

---

## 📁 Project Structure

```text
URL-phishing-Detector/
│
├── Dataset.csv
├── Data_Preprocessing.py
├── train_evaluate.py
├── test_compare.py
├── models_config.py
├── app.py
├── best_model.pkl
├── expected_features.pkl
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Mallak-Khaled/URL-phishing-Detector.git
cd URL-phishing-Detector
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application using:

```bash
streamlit run app.py
```

Enter a URL to check whether it is **Legitimate** or **Phishing**.

---

## 📌 Project Status

🚧 **In Progress**

The project is currently under development, including model comparison, hyperparameter tuning, feature extraction, and final model integration.

---

## 🚀 Future Improvements

- Improve performance using larger datasets.
- Explore additional Machine Learning and Deep Learning models.
- Add real-time threat intelligence.
- Develop a browser extension for real-time detection.

---

## 🎓 Acknowledgments

Special thanks to **NTI (National Telecommunication Institute)** for the training and learning experience.

---

## 📩 Contact Me

For questions, suggestions, or collaboration, feel free to contact me.

**Malak Khaled**
Project Link:https://github.com/Mallak-Khaled/URL-phishing-Detector
📧 Email: mokak43@gmail.com
