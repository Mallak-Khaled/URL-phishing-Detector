import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath="dataset_phishing.csv"):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()

    print(df)

    print(df.shape)
    print(list(df.columns))
    print("\nmissing values", df.isnull().sum())
    print("\nclass counts", df["status"].value_counts())
    print("\nsummary statistics\n", df.describe())
    
    # 1. The 36 robust features (Lexical + Fast Internet APIs)
    selected_features = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 
        'nb_and', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 
        'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 
        'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host', 
        'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 'nb_subdomains', 'prefix_suffix',
        'whois_registered_domain', 'domain_registration_length', 'domain_age', 'dns_record'
    ]
    
    # 2. Keep only selected features + target status
    columns_to_keep = selected_features + ['status']
    df = df[columns_to_keep]
    

# 4. Encode Target
    if "status" in df.columns:
        le = LabelEncoder()
        df['status'] = le.fit_transform(df['status'])
    
    # 5. Split features (X) and target (y)
    X = df.drop("status", axis=1)
    y = df["status"]
    
    # 6. Train/Test Split
    X_train_original, X_test_original, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Apply Scaling for Distance-based/Linear Models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_original)
    X_test_scaled = scaler.transform(X_test_original)
    
    return X_train_original, X_test_original, y_train, y_test, X_train_scaled, X_test_scaled, scaler


load_and_preprocess_data()