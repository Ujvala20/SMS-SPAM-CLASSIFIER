import pandas as pd
import plotly.express as px

from evaluate import load_results, load_predictions
from features import add_features

SPAM_WORDS = ["free", "win", "winner", "prize", "claim", "call", "txt", "text",
              "urgent", "cash", "offer", "reply", "stop", "mobile"]


def explain_error(row):
    """Simple rule-based reasons why the model may have been wrong."""
    reasons = []
    text = row["text"].lower()
    found = [w for w in SPAM_WORDS if w in text.split()]

    if row["type"] == "False negative (spam missed)":
        if row["length"] < 80:
            reasons.append("short message, little text to learn from")
        if not row["has_url"] and not row["has_currency"]:
            reasons.append("no URL or currency symbol")
        if not found:
            reasons.append("no common spam trigger words")
        if row["digit_count"] < 5:
            reasons.append("few digits")
    else:
        if found:
            reasons.append("contains spam-like words: " + ", ".join(found))
        if row["has_url"]:
            reasons.append("contains a URL")
        if row["has_currency"]:
            reasons.append("contains a currency symbol")
        if row["digit_count"] >= 8:
            reasons.append("many digits (phone number or amount)")

    return "; ".join(reasons) if reasons else "no obvious pattern, borderline case"


def error_table(preds, model, threshold=0.5):
    df = preds[["text", "y_true", model]].copy()
    df.columns = ["text", "y_true", "spam_probability"]
    df["y_pred"] = (df["spam_probability"] >= threshold).astype(int)
    df = add_features(df)

    wrong = df[df["y_true"] != df["y_pred"]].copy()
    wrong["type"] = wrong["y_true"].map({
        1: "False negative (spam missed)",
        0: "False positive (ham flagged)"})
    wrong["reason"] = wrong.apply(explain_error, axis=1)
    wrong["spam_probability"] = wrong["spam_probability"].round(3)
    return wrong[["type", "text", "spam_probability", "reason"]].reset_index(drop=True)


def error_counts_fig(errors):
    counts = errors["type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    return px.bar(counts, x="type", y="count", color="type",
                  title="Types of errors made by the best model")


if __name__ == "__main__":
    data = load_results()
    best = data["best_model"]
    preds = load_predictions()

    errors = error_table(preds, best)
    errors.to_csv("errors.csv", index=False)

    print(f"Best model: {best}")
    print(f"Total errors: {len(errors)} out of {len(preds)} test messages\n")
    print(errors["type"].value_counts(), "\n")

    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.width", 200)
    for t in errors["type"].unique():
        print(f"--- {t} ---")
        print(errors[errors["type"] == t][["text", "spam_probability", "reason"]]
              .head(8).to_string(index=False), "\n")

    error_counts_fig(errors).show()