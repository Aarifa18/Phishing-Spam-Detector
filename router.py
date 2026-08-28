"""
The two trained models are connected via router.py.

What this file does:
1. Loads the email (NLP) and URL models that have already been trained and saved.
2. Determines whether the text in is an email or a URL.
3. If it's an email, it additionally looks for links in the email body and rates them using the URL model.
4. Combines all of the information into a single "phishing probability" and a straightforward conclusion (phishing/legitimate).
"""
import os
import re

import joblib
import pandas as pd

from feature_extraction import extract_url_features
from text_preprocessing import text_processing

# 1. LOAD SAVED MODELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# These three files must exist in a "models" folder next to this script.
url_model = joblib.load(os.path.join(MODEL_DIR, "url_model.joblib"))
url_feature_names = joblib.load(os.path.join(MODEL_DIR, "url_feature_names.joblib"))
nlp_pipeline = joblib.load(os.path.join(MODEL_DIR, "nlp_pipeline.joblib"))


# 2. DECIDE: IS THIS A URL, OR AN EMAIL?
# A simple regex that matches things starting with http://, https://, or www.
URL_REGEX = re.compile(r'https?://[^\s<>"\')]+|www\.[^\s<>"\')]+', re.IGNORECASE)


def looks_like_url(text: str) -> bool:
    """True if the WHOLE input is just one URL, with no other text around it."""
    text = text.strip()
    if " " in text or "\n" in text:
        return False  # multiple words/lines -> this is a message, not a bare URL
    return bool(re.match(r"^(https?://|www\.)", text, re.IGNORECASE))


def extract_urls(text: str) -> list:
    """Find any URLs hiding inside a longer block of text (e.g. an email body)."""
    return URL_REGEX.findall(text)


# 3. SCORE A SINGLE URL OR A BLOCK OF EMAIL TEXT
def score_url(url: str) -> float:
    """Returns a number between 0 and 1: the model's estimated phishing probability."""
    feats = extract_url_features(url)
    row = pd.DataFrame([feats])
    row = row.reindex(columns=url_feature_names, fill_value=0)
    proba = url_model.predict_proba(row.to_numpy(dtype="float64"))[0][1]
    return float(proba)


def score_email_text(text: str) -> float:
    """Returns a number between 0 and 1: the model's estimated spam/phishing probability."""
    proba = nlp_pipeline.predict_proba([text])[0][1]
    return float(proba)

# 4. THE MAIN FUNCTION — one API and demo app will call

def predict(input_text: str) -> dict:
    """Takes raw text (a URL, or an email) and returns a dictionary explaining
    the result """
    input_text = input_text.strip()
    result = {
        "input_type": None,
        "nlp_score": None,
        "url_scores": [],
        "final_score": None,
        "verdict": None,
    }

    if looks_like_url(input_text):
        # --- Case 1: the input is just a bare URL ---
        result["input_type"] = "url"
        score = score_url(input_text)
        result["url_scores"] = [{"url": input_text, "score": score}]
        result["final_score"] = score

    else:
        # --- Case 2: treat it as email/free text ---
        result["input_type"] = "email"
        nlp_score = score_email_text(input_text)
        result["nlp_score"] = nlp_score

        embedded_urls = extract_urls(input_text)
        url_scores = [{"url": u, "score": score_url(u)} for u in embedded_urls]
        result["url_scores"] = url_scores

        if url_scores:
            worst_url_score = max(s["score"] for s in url_scores)
            result["final_score"] = 0.6 * nlp_score + 0.4 * worst_url_score
        else:
            # No links found, so we only have the text signal to go on.
            result["final_score"] = nlp_score

    result["verdict"] = "phishing" if result["final_score"] >= 0.5 else "legitimate"
    return result


# 5. A QUICK MANUAL TEST — lets you sanity-check this file on its own
if __name__ == "__main__":
    test_input = input("Paste a URL or email text to test: ")
    output = predict(test_input)
    print("\n--- Result ---")
    for key, value in output.items():
        print(f"{key}: {value}")
