import pandas as pd
import os
import joblib

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

from tld import get_tld, is_tld
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score,confusion_matrix


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"data","phishing_url_combined.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
data = pd.read_csv(DATA_PATH)

print("Reading from:", DATA_PATH)
print(data.columns.tolist())
print(data['type'].value_counts(dropna=False) if 'type' in data.columns else "NO 'type' COLUMN YET")

data = data.rename(columns={'URL': 'url'})
#Feature Extraction
#extract top level domain
def tld(url):
    try:
        res=get_tld(url, as_object = True, fail_silently=False, fix_protocol=True)
        pri_domain=res.parsed_url.netloc
    except:
        pri_domain=None
    return pri_domain

#checks if url starts with https
def HTTPs(url):
    htp = urlparse(url).scheme
    match = str(htp)
    if match =='https':
        return 1
    else:
        return 0
#checks how many numbers are in the url
def number_count(url):
    number = 0
    for i in url:
        if i.isnumeric():
            number += 1
    return number
#checks how many letters are in the url
def letterCount(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters += 1
    return letters

def shortening(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

#Use of IP or not in domain
def IP_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return -1
    else:
        return 1
#putting everything in one function
def URLconverter(url):
    data['url_clean'] = data['url'].apply(lambda u: re.sub(r'^https?://', '', str(u), flags=re.IGNORECASE))
    data['url_len'] = data['url_clean'].apply(lambda x: len(str(x)))
    data['digits'] = data['url_clean'].apply(lambda i: number_count(i))
    data['letters'] = data['url_clean'].apply(lambda i: letterCount(i))
    data['ShorteningService'] = data['url_clean'].apply(lambda i: shortening(i))
    data['IP_address'] = data['url_clean'].apply(lambda i: IP_address(i))
    spec_char = ['@','?','-','=','.','#','%','+','$','!','*',',','//','&','/',';',':','^','~','|','<','>','{','}']
    for special in spec_char:
        data[special] = data['url_clean'].apply(lambda i: i.count(special))
    X = data.drop(['url','url_clean'],axis=1)

    data['digit_ratio'] = data['digits'] / data['url_len']
    data['letter_ratio'] = data['letters'] / data['url_len']
    data['special_char_density'] = (data['url_len'] - data['letters'] - data['digits']) / data['url_len']

    return X
    
final_data=URLconverter(data['url'])
final_data['type'] = final_data['type'].astype(str).str.strip().str.lower()
final_data = final_data.dropna(subset=['type'])
final_data = final_data[final_data['type'].isin(['benign', 'phishing'])]
final_data['type'] = final_data['type'].map({'benign': 0, 'phishing': 1}).astype('int32')

print(f"Rows after cleaning 'type' column: {len(final_data)}")

#splitting into training and testing data set
X = final_data.drop(['type'],axis=1)
Y = final_data['type']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
# Check how rare IP_address actually is
print(final_data['IP_address'].value_counts())

# Drop near-zero-importance features
low_value_features = ['#', '@', 'IP_address', ',', '*', '$', '!', '^', '|', '<', '>', '{', '}']
X = X.drop(columns=low_value_features)
feature_names = X.columns.tolist()

# Re-split with the trimmed X
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_train = X_train.to_numpy(dtype='float64')
X_test = X_test.to_numpy(dtype='float64')

print(Y_train.dtype, Y_train.isnull().sum(), Y_train.unique(), len(X_train), len(Y_train))

clf = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    colsample_bytree=0.8,      # = feature_fraction
    subsample=0.8,             # = bagging_fraction
    subsample_freq=5,          # = bagging_freq
    reg_alpha=0.1,             # = lambda_l1
    reg_lambda=0.2,            # = lambda_l2
    min_child_samples=20,      # = min_data_in_leaf
    n_estimators=1000,
    verbose=-1,
    random_state=42,
)

clf.fit(
    X_train, Y_train,
    eval_set=[(X_train, Y_train), (X_test, Y_test)],
    eval_metric=["auc", "binary_logloss"],
    callbacks=[lgb.early_stopping(50)],
)
print(clf.classes_)

pred=clf.predict(X_test)

joblib.dump(clf, os.path.join(MODEL_DIR,"url_model.joblib"))
print(f"Model saved to {os.path.join(MODEL_DIR,'url_model.joblib')}")

joblib.dump(feature_names, os.path.join(MODEL_DIR,"url_feature_names.joblib"))
print(f"Model saved to {os.path.join(MODEL_DIR,'url_feature_names.joblib')}")

print("Accuracy:", round(accuracy_score(Y_test, pred), 2))
print(classification_report(Y_test, pred))
print(confusion_matrix(Y_test, pred))

conf_matrix = confusion_matrix(Y_test, pred)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()