import pandas as pd
import os


#Get the folder this script lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"data","malicious_phish.csv")
data = pd.read_csv(DATA_PATH)

data.isnull().sum() #checks if there is any missing value
data.duplicated().sum() #check if there is any duplicate
data.drop_duplicates(inplace=True) #drops the duplicate

#Keeping only phishing & benign url
indexNames = data[ data['type'] == 'defacement' ].index
data.drop(indexNames , inplace=True)

indexNames = data[ data['type'] == 'malware' ].index
data.drop(indexNames , inplace=True)

benign = data[data['type'] == 'benign']
phish = data[data['type'] == 'phishing']

phishing_sampled = phish.sample(n=int(len(phish)/18), random_state=42)
print(len(phishing_sampled))
phishing_sampled = pd.DataFrame(phishing_sampled)

benign_sampled = benign.sample(n=int(len(benign)/85),random_state=42)
print(len(benign_sampled))
benign_sampled = pd.DataFrame(benign_sampled)

balanced_dataset = pd.concat([benign_sampled,phishing_sampled])
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42)
print(balanced_dataset.shape[0])

OUTPUT_PATH = os.path.join(BASE_DIR,"data","phishing_url.csv")
balanced_dataset.to_csv(OUTPUT_PATH, index=False)
print(f"done - saved to {OUTPUT_PATH}")