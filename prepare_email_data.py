import pandas as pd
import os


#Get the folder this script lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"data","email_spam.csv")
data = pd.read_csv(DATA_PATH)

def clean_email_data(df):
    df = df.drop_duplicates(subset='text').copy()
    df = df.dropna(subset=['text', 'spam'])
    
    # Optional: split out the subject line as its own feature
    df['subject'] = df['text'].str.extract(r'^Subject:\s*(.*?)(?:\n|$)')
    df['body'] = df['text'].str.replace(r'^Subject:\s*', '', regex=True)
    
    # normalize whitespace
    df['body'] = df['body'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    df = df.rename(columns={'spam': 'label', 'text': 'Email'})
    return df

cleaned_data = clean_email_data(data)
print(cleaned_data.shape[0])


OUTPUT_PATH = os.path.join(BASE_DIR,"data","spam_email.csv")
cleaned_data.to_csv(OUTPUT_PATH, index=False)
print(f"done - saved to {OUTPUT_PATH}")