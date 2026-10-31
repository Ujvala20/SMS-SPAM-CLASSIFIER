import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from data import load_data
from features import add_features, FEATURE_COLS


def build_pipeline(clf, use_extra):
    transformers = [("tfidf",
                     TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
                     "text")]
    if use_extra:
        transformers.append(("num", MinMaxScaler(), FEATURE_COLS))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])


def get_models():
    return {
        "Naive Bayes (TF-IDF)": build_pipeline(MultinomialNB(), False),
        "Logistic Regression (TF-IDF)": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced"), False),
        "Naive Bayes (TF-IDF + features)": build_pipeline(MultinomialNB(), True),
        "Logistic Regression (TF-IDF + features)": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced"), True),
    }


if __name__ == "__main__":
    df = add_features(load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"] + FEATURE_COLS], df["y"],
        test_size=0.2, random_state=42, stratify=df["y"])

    results = {}
    fitted = {}
    preds_df = X_test[["text"]].copy()
    preds_df["y_true"] = y_test.values

    for name, pipe in get_models().items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        preds_df[name] = proba
        fitted[name] = pipe

    best_name = max(results, key=lambda n: results[n]["f1"])

    joblib.dump(fitted[best_name], "spam_model.joblib")
    json.dump({"best_model": best_name, "results": results},
              open("results.json", "w"), indent=2)
    preds_df.to_csv("test_predictions.csv", index=False)

    summary = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].round(3)
    print(summary.to_string())
    print(f"\nBest model: {best_name}")
    print("Saved: spam_model.joblib, results.json, test_predictions.csv")