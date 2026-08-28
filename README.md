# Phishing & Spam Detector

A machine learning system for detecting phishing URLs and phishing/spam emails, combining two independently trained models behind a router, a REST API, and a live demo.

Originally built as a university dissertation project, then rebuilt from the ground up with a proper ML pipeline: cleaned data, dataset-artifact investigation, a LightGBM URL classifier, a TF-IDF/Logistic Regression email classifier, and a FastAPI + Streamlit deployment layer.

**[Live demo](#) · [Full project writeup / case study](https://app.notion.com/p/Phishing-Spam-Detector-Combined-URL-Email-Classifier-3caff8f68e358025b3fece55707179dc)**

---

## Features

- **URL classifier** — LightGBM model trained on engineered lexical features (length, character composition, special-character density, use of URL shorteners, etc.)
- **Email classifier** — TF-IDF + Logistic Regression pipeline trained on cleaned, lemmatized email text
- **Router** — detects whether input is a URL or free text, extracts embedded links from email bodies, and combines both model scores into a single prediction
- **REST API** — FastAPI endpoint for programmatic use
- **Web demo** — Streamlit app for interactive testing
- **Documented data investigation** — this project deliberately surfaces and fixes several real dataset artifacts (see [Notes on Data Quality](#notes-on-data-quality) below) rather than reporting a single accuracy number at face value

## Project Structure

```
PhishingDetection/
├── data/                      # training data (not committed — see Setup)
├── models/                    # trained model files (not committed — see Setup)
├── feature_extraction.py      # shared URL feature engineering (train + inference)
├── text_preprocessing.py      # shared email text cleaning (train + inference)
├── url_model.py               # trains and saves the URL model
├── email_model.py             # trains and saves the email model
├── merge_datasets.py          # combines/rebalances multiple URL data sources
├── router.py                  # routes input to the right model(s) and combines scores
├── api.py                     # FastAPI app
├── app_streamlit.py           # Streamlit demo app
├── evaluate.py                # reports held-out test metrics for both models
├── sanity_check.py            # real-world generalization checks (not just test-set accuracy)
└── requirements.txt
```

## Setup

**1. Clone and create a virtual environment**
```bash
git clone <your-repo-url>
cd PhishingDetection
python -m venv phishenv
phishenv\Scripts\activate        # Windows
# source phishenv/bin/activate   # macOS/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get the data**

Data files aren't committed to this repo (they're large, and some are third-party datasets with their own licensing — see [Data Sources](#data-sources)). Download the following into a `data/` folder:

- [PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets/kaggleprollc/phishing-url-websites-dataset-phiusiil)
- [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) (used to supplement path diversity in the legitimate class — see writeup)
- [An email spam/phishing dataset](https://www.kaggle.com/datasets/bagavathypriya/email-spam-dataset)

Then run the data prep and merge scripts to produce the final training files:
```bash
python prepare_email_data.py
python merge_datasets.py
```

**4. Train the models**
```bash
python url_model.py
python email_model.py
```
This saves `url_model.joblib`, `url_feature_names.joblib`, and `nlp_pipeline.joblib` into `models/`.

**5. Evaluate**
```bash
python evaluate.py
```

**6. Run the demo or API**
```bash
streamlit run app_streamlit.py
# or
uvicorn api:app --reload
```

## Results

Evaluated on a held-out test split (30% of data, never seen during training):

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| URL model (LightGBM) | 81.6% | 83.2% | 79.1% | 81.1% |
| Email model (TF-IDF + LogReg) | 87.2% | 78.9% | 63.7% | 70.5% |

**Note on the email model's recall (63.7%):** this is the metric I'm most focused on improving next. For a phishing detector, a missed phishing email (false negative) is a more costly error than a false positive, and the current recall means roughly a third of phishing/spam emails in testing weren't caught. See [Future Improvements](#future-improvements).

Metrics above are on a held-out test split. See [`sanity_check.py`](sanity_check.py) for real-world generalization checks that go beyond test-set accuracy alone.

## Notes on Data Quality

Test-set accuracy alone turned out to be a misleading signal for this project. During development, this pipeline surfaced three real, non-obvious dataset artifacts:

1. **A scheme-prefix artifact** — the presence of `http(s)://` correlated with the label in the source data for reasons unrelated to actual phishing behavior, causing the model to misclassify ordinary real-world URLs.
2. **A URL-length artifact** — benign and phishing URLs had systematically different length distributions in the source dataset, in the opposite direction from real-world intuition.
3. **A path-presence artifact** — one dataset's "legitimate" class contained zero URLs with a path (only bare domains), causing any URL with a path — including completely ordinary ones — to be flagged as phishing.

Each was found by testing the model against real, everyday URLs rather than trusting the test-set accuracy number alone, then traced back to its root cause in the data and fixed at the source (feature engineering, dataset supplementation, or dataset combination) rather than patched superficially. Full details in the [project writeup](https://app.notion.com/p/Phishing-Spam-Detector-Combined-URL-Email-Classifier-3caff8f68e358025b3fece55707179dc).

## Data Sources

- [PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets/kaggleprollc/phishing-url-websites-dataset-phiusiil) — Prasad, A. & Chandra, S. (2024), *Computers & Security*
- [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
- [Email Spam dataset](https://www.kaggle.com/datasets/bagavathypriya/email-spam-dataset)

## Future Improvements

- **Improve email model recall** — currently 63.7% on the phishing class, meaning roughly a third of phishing/spam emails are missed in testing. Planned approaches: lowering the decision threshold, sourcing more phishing email examples, and revisiting class weighting
- Replace the router's weighted-average combination with a trained meta-model over both scores
- Add SHAP-based explainability to predictions
- Compare the TF-IDF baseline against a fine-tuned DistilBERT model
- Add unit tests and CI
