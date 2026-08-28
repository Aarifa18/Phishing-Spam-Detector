"""
Feature extraction for a SINGLE URL. This mirrors the exact
same logic so predictions at inference time match what the model learned.
"""
import re
from urllib.parse import urlparse


def is_https(url: str) -> int:
    return 1 if urlparse(url).scheme == "https" else 0


def count_digits(url: str) -> int:
    return sum(ch.isnumeric() for ch in url)


def count_letters(url: str) -> int:
    return sum(ch.isalpha() for ch in url)


_SHORTENING_PATTERN = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|"
    r"db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|"
    r"q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
    r"x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"tr\.im|link\.zip\.net"
)


def uses_shortening_service(url: str) -> int:
    return 1 if _SHORTENING_PATTERN.search(url) else 0


_IP_PATTERN = re.compile(
    r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\."
    r"([01]?\d\d?|2[0-4]\d|25[0-5])\/)|"
    r"((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)"
    r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}"
)


def uses_ip_address(url: str) -> int:
    return -1 if _IP_PATTERN.search(url) else 1


SPECIAL_CHARS = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',',
                  '//', '&', '/', ';', ':', '^', '~', '|', '<', '>', '{', '}']


def normalize_url(url: str) -> str:
    return re.sub(r"^https?://", "", str(url), flags=re.IGNORECASE)


def extract_url_features(url: str) -> dict:
    url = str(url)
    https_flag = is_https(url)  # check the ORIGINAL url, before stripping
    normalized = normalize_url(url)
 
    features = {
        "url_len": len(normalized),
        "HTTPS": https_flag,
        "digits": count_digits(normalized),
        "letters": count_letters(normalized),
        "ShorteningService": uses_shortening_service(normalized),
        "IP_address": uses_ip_address(normalized),
    }
    for char in SPECIAL_CHARS:
        features[char] = normalized.count(char)
 
    url_len_safe = max(features["url_len"], 1)  # avoid divide-by-zero on an empty string
    features["digit_ratio"] = features["digits"] / url_len_safe
    features["letter_ratio"] = features["letters"] / url_len_safe
    features["special_char_density"] = (
        features["url_len"] - features["letters"] - features["digits"]
    ) / url_len_safe
 
    return features
