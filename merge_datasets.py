import os
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# 1. Load and clean phishing_site_urls.csv
new_data = pd.read_csv(os.path.join(DATA_DIR, "phishing_site_urls.csv"))

url_charset = re.compile(r"^[a-zA-Z0-9\-._~:/?#\[\]@!\$&'()*+,;=%\s]*$")
is_clean = new_data["URL"].apply(lambda u: bool(url_charset.match(str(u))))
new_data = new_data[is_clean].copy()

new_data = new_data.rename(columns={"URL": "url", "Label": "label_text"})
new_data["type"] = new_data["label_text"].map({"good": "benign", "bad": "phishing"})
new_data = new_data[["url", "type"]]

print("New source, after cleaning:")
print(new_data["type"].value_counts())
print()


# 2. Load your existing (already-prepared) dataset
existing_data = pd.read_csv(os.path.join(DATA_DIR, "phishing_url.csv"))
existing_data = existing_data.rename(columns={"URL": "url"})
existing_data["type"] = existing_data["type"].map({1: "benign", 0: "phishing"})
existing_data = existing_data[["url", "type"]]

print("Existing dataset:")
print(existing_data["type"].value_counts())
print()

# 3. Combine, deduplicate, and rebalance
combined = pd.concat([existing_data, new_data], ignore_index=True)
combined = combined.drop_duplicates(subset="url")
combined = combined.dropna(subset=["url", "type"])

print("Combined (before rebalancing):")
print(combined["type"].value_counts())
print()

benign = combined[combined["type"] == "benign"]
phishing = combined[combined["type"] == "phishing"]

target_size = min(len(benign), len(phishing))
benign_sampled = benign.sample(n=target_size, random_state=42)
phishing_sampled = phishing.sample(n=target_size, random_state=42)

final = pd.concat([benign_sampled, phishing_sampled])
final = final.sample(frac=1, random_state=42)  # shuffle

print("Final, balanced combined dataset:")
print(final["type"].value_counts())
print()


# 4. Verify the path artifact is actually fixed before saving
def has_path(url):
    parts = str(url).split("/", 1)
    return len(parts) > 1 and len(parts[1].strip("/")) > 0

final["has_path"] = final["url"].apply(has_path)
print("Path presence by class (should be reasonably similar for both now):")
print(final.groupby("type")["has_path"].mean())
final = final.drop(columns=["has_path"])


# 5. Save
OUTPUT_PATH = os.path.join(DATA_DIR, "phishing_url_combined.csv")
final.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
