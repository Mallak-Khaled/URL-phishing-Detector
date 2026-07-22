import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re
import urllib.parse
import tldextract
import whois
import socket
from datetime import datetime

# ==========================================
# 1. Page Configuration & Theme
# ==========================================
st.set_page_config(
    page_title="Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        color: #00FF41;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        text-shadow: 0 0 5px #00FF41;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00FF41;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .metric-card-danger {
        border-left: 5px solid #FF4B4B;
    }
    .safe-result {
        background-color: rgba(0, 255, 65, 0.1);
        border-left: 5px solid #00FF41;
        padding: 15px;
        border-radius: 5px;
        color: #00FF41;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin-top: 20px;
    }
    .phishing-result {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #FF4B4B;
        padding: 15px;
        border-radius: 5px;
        color: #FF4B4B;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Fast Feature Extraction Engine (Lexical + Fast API)
# ==========================================
def extract_url_features(url, expected_columns):
    features = {col: 0 for col in expected_columns}
    
    if not url.startswith('http'):
        url = 'http://' + url
        
    try:
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        
        try:
            ext = tldextract.extract(url)
            subdomain = ext.subdomain
            registered_domain = f"{ext.domain}.{ext.suffix}"
        except:
            subdomain = ""
            registered_domain = domain

        # --- Lexical Features ---
        features['length_url'] = len(url)
        features['length_hostname'] = len(domain)
        features['ip'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        features['nb_at'] = url.count('@')
        features['nb_qm'] = url.count('?')
        features['nb_and'] = url.count('&')
        features['nb_eq'] = url.count('=')
        features['nb_underscore'] = url.count('_')
        features['nb_tilde'] = url.count('~')
        features['nb_percent'] = url.count('%')
        features['nb_slash'] = url.count('/')
        features['nb_star'] = url.count('*')
        features['nb_colon'] = url.count(':')
        features['nb_comma'] = url.count(',')
        features['nb_semicolumn'] = url.count(';')
        features['nb_dollar'] = url.count('$')
        features['nb_space'] = url.count(' ') + url.count('%20')
        features['nb_www'] = 1 if re.search(r'\bwww\b', url.lower()) else 0
        features['nb_com'] = 1 if re.search(r'\bcom\b', url.lower()) else 0
        features['nb_dslash'] = url.count('//')
        features['http_in_path'] = 1 if 'http' in path.lower() else 0
        features['https_token'] = 1 if parsed_url.scheme == 'https' else 0
        
        url_len = max(1, len(url))
        domain_len = max(1, len(domain))
        features['ratio_digits_url'] = sum(c.isdigit() for c in url) / url_len
        features['ratio_digits_host'] = sum(c.isdigit() for c in domain) / domain_len
        
        features['punycode'] = 1 if 'xn--' in url else 0
        features['port'] = 1 if parsed_url.port else 0
        features['tld_in_path'] = 1 if ext.suffix and ext.suffix in path else 0
        features['tld_in_subdomain'] = 1 if ext.suffix and ext.suffix in subdomain else 0
        features['nb_subdomains'] = len(subdomain.split('.')) if subdomain else 0
        features['prefix_suffix'] = 1 if '-' in domain else 0

        # --- Fast Internet Features (DNS & WHOIS) ---
        # 1. DNS Record
        features['dns_record'] = 0 # Assume healthy by default
        try:
            socket.gethostbyname(domain)
        except:
            features['dns_record'] = 1 # 1 means no record found (suspicious)
            
        # 2. WHOIS Information
        features['whois_registered_domain'] = 0 # Assume not registered by default
        features['domain_registration_length'] = 284
        features['domain_age'] = 5079
        
        try:
            domain_info = whois.whois(registered_domain)
            if domain_info.domain_name:
                features['whois_registered_domain'] = 0
                
            creation_date = domain_info.creation_date
            expiration_date = domain_info.expiration_date
            
            if isinstance(creation_date, list): creation_date = creation_date[0]
            if isinstance(expiration_date, list): expiration_date = expiration_date[0]
            
            if creation_date:
                features['domain_age'] = (datetime.now() - creation_date).days
            if expiration_date:
                features['domain_registration_length'] = (expiration_date - datetime.now()).days
        except:
            pass # Keep defaults if WHOIS fails
            
    except Exception as e:
        pass 

    df_result = pd.DataFrame([features])
    return df_result[expected_columns]

# ==========================================
# 3. Load Model and Assets
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    expected_cols = joblib.load('expected_features.pkl')
    return model, expected_cols

model, expected_cols = load_assets()

# ==========================================
# 4. Main Dashboard UI
# ==========================================
st.markdown("<h1 class='main-title'>🛡️Phishing URL Detector</h1>", unsafe_allow_html=True)
# st.markdown("######  _Phishing URL Detector_", text_alignment='center')

st.write("---")

# ------------------------------------------
# TAB 1: Single URL Analysis
# ------------------------------------------

st.markdown("### Real-Time URL Inspection")
url_input = st.text_input("Enter the target URL to scan:", placeholder="https://www.example.com")

if st.button("🚀 Initiate Deep Scan"):
    if url_input:
        with st.spinner("Executing Lexical, DNS, and WHOIS analysis..."):
            
            # Extract Features dynamically for ALL URLs (No Whitelist)
            input_features = extract_url_features(url_input, expected_cols)
            
            if input_features is not None:
                # Predict using the XGBoost Model
                prediction = model.predict(input_features)[0]
                
                try:
                    proba = model.predict_proba(input_features)[0]
                    confidence = proba[1] if prediction == 1 else proba[0]
                    confidence_text = f"Confidence Score: {confidence * 100:.2f}%"
                except:
                    confidence_text = ""
                # Display Result
                if prediction == 1:
                    st.markdown(f"<div class='phishing-result'>🚨 THREAT DETECTED!<br>This URL exhibits phishing characteristics.<br><span style='font-size: 1rem;'>{confidence_text}</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='safe-result'>✅ SECURE<br>No malicious patterns detected.<br><span style='font-size: 1rem;'>{confidence_text}</span></div>", unsafe_allow_html=True)
                
                with st.expander("🛠️ View Extracted Feature Vector (36 Features)"):
                    st.dataframe(input_features.transpose(), use_container_width=True)
    else:
        st.warning("Please provide a valid URL to analyze.")


st.sidebar.markdown("---")
st.sidebar.write("**Model Engine:** Random Forest Classifier")