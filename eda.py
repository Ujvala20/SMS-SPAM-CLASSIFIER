import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

from data import load_data
from features import add_features

COLORS = {"ham": "#2E86AB", "spam": "#E4572E"}


def class_balance(df):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    return px.pie(counts, names="label", values="count", hole=0.4,
                  color="label", color_discrete_map=COLORS,
                  title="Spam vs Ham distribution")


def length_distribution(df):
    return px.histogram(df, x="length", color="label", nbins=50,
                        barmode="overlay", opacity=0.7,
                        color_discrete_map=COLORS,
                        title="Message length: spam vs ham")


def digit_boxplot(df):
    return px.box(df, x="label", y="digit_count", color="label",
                  color_discrete_map=COLORS,
                  title="Digits per message: spam vs ham")


def feature_rates(df):
    rates = (df.groupby("label")[["has_url", "has_currency"]].mean() * 100).reset_index()
    rates = rates.melt(id_vars="label", var_name="feature", value_name="percent")
    return px.bar(rates, x="feature", y="percent", color="label", barmode="group",
                  color_discrete_map=COLORS,
                  title="% of messages containing a URL / currency symbol")


def top_words(df, label, n=15):
    cv = CountVectorizer(stop_words="english", min_df=2)
    X = cv.fit_transform(df[df["label"] == label]["text"])
    counts = X.sum(axis=0).A1
    return (pd.Series(counts, index=cv.get_feature_names_out())
            .sort_values(ascending=False).head(n))


def top_words_chart(df, label, n=15):
    words = top_words(df, label, n).sort_values()
    return px.bar(x=words.values, y=words.index, orientation="h",
                  title=f"Top {n} words in {label} messages",
                  labels={"x": "count", "y": "word"},
                  color_discrete_sequence=[COLORS[label]])


if __name__ == "__main__":
    df = add_features(load_data())

    print("Top spam words:\n", top_words(df, "spam", 10), "\n")
    print("Top ham words:\n", top_words(df, "ham", 10))

    class_balance(df).show()
    length_distribution(df).show()
    digit_boxplot(df).show()
    feature_rates(df).show()
    top_words_chart(df, "spam").show()
    top_words_chart(df, "ham").show()