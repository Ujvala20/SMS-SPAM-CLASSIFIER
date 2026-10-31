import re
import pandas as pd
from data import load_data

URL_PATTERN = re.compile(r"(http|https|www\.|\.com|\.co\.uk|\.net|\.org)", re.IGNORECASE)
CURRENCY_PATTERN = re.compile(r"[£$€₹]")


def add_features(df):
    df = df.copy()
    df["length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["digit_count"] = df["text"].str.count(r"\d")
    df["has_url"] = df["text"].apply(lambda t: int(bool(URL_PATTERN.search(t))))
    df["has_currency"] = df["text"].apply(lambda t: int(bool(CURRENCY_PATTERN.search(t))))
    df["caps_ratio"] = df["text"].apply(
        lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
    )
    return df


FEATURE_COLS = ["length", "word_count", "digit_count", "has_url", "has_currency", "caps_ratio"]

if __name__ == "__main__":
    df = add_features(load_data())
    print(df.groupby("label")[FEATURE_COLS].mean().round(3))
    print()
    print("% of messages with a URL:")
    print((df.groupby("label")["has_url"].mean() * 100).round(1))
    print()
    print("% of messages with a currency symbol:")
    print((df.groupby("label")["has_currency"].mean() * 100).round(1))