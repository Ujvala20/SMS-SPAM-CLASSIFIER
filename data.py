import pandas as pd


def load_data(path="spam.csv"):
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df = df.drop_duplicates().reset_index(drop=True)
    df["y"] = (df["label"] == "spam").astype(int)  # spam = 1, ham = 0
    return df


if __name__ == "__main__":
    raw = pd.read_csv("spam.csv", encoding="latin-1")
    df = load_data()
    print("Rows before cleaning:", len(raw))
    print("Rows after removing duplicates:", len(df))
    print(df["label"].value_counts())
    print(df.head())