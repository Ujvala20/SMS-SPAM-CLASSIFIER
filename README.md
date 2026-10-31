# SpamLens: SMS Spam Classifier

A website where you paste a text message and get a verdict (spam or ham), the signals behind it, and the exact words that decided it. A second Streamlit dashboard holds the full data analysis and model comparison.

**Live demo:** `<FILL: link after deployment, or delete this line>`

![Home](screenshots/01_home.png)

---

## Problem statement

Spam messages waste time and are used for fraud. The goal is to detect spam SMS accurately while keeping false alarms low, so that real messages are not blocked.

## Dataset

- **SMS Spam Collection** (UCI Machine Learning Repository, via Kaggle)
- 5,572 messages originally, **5,169 after removing duplicates**
- 4,516 ham and 653 spam, so the data is **imbalanced** (about 13% spam)

## Approach

1. **Cleaning**: removed duplicates and converted labels to 0/1
2. **Feature engineering**: message length, word count, digit count, URL flag, currency flag, capital-letter ratio
3. **Text vectorisation**: TF-IDF with unigrams and bigrams, English stop words removed
4. **Models**: Multinomial Naive Bayes and Logistic Regression (balanced class weights), each trained with and without the extra features, giving 4 models
5. **Evaluation**: stratified 80/20 split, judged on precision, recall, F1, ROC-AUC and PR-AUC. Accuracy alone is misleading here because a model that always says "ham" would score about 87%.
6. **Error analysis**: reviewed the messages the model got wrong and why

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (TF-IDF) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Logistic Regression (TF-IDF) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Naive Bayes (TF-IDF + features) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Logistic Regression (TF-IDF + features) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

**Best model:** `<FILL: name>` with an F1 of `<FILL>`.

![Accuracy section](screenshots/04_accuracy.png)

### Key findings

- Spam messages are much longer and contain far more digits than ham: `<FILL: your numbers>`
- `<FILL: X>`% of spam contains a URL versus `<FILL: Y>`% of ham.
- Adding the handcrafted features changed F1 by `<FILL: +/- value>` for Naive Bayes and `<FILL: +/- value>` for Logistic Regression.

## Error analysis

On the test set the best model made `<FILL>` errors: `<FILL>` spam messages were missed and `<FILL>` real messages were flagged.

- **Missed spam** was mostly written like casual conversation, with no URL, currency symbol or typical trigger words.
- **Wrongly flagged ham** mostly contained phone numbers, money amounts, or words like "call" and "free".

## Explainability

For every message, the website removes each word in turn and measures how much the spam probability drops. Words that raise the probability are shown in red and words that lower it in green. This works with any model.

| Spam result | Ham result |
|---|---|
| ![Spam](screenshots/02_spam_result.png) | ![Ham](screenshots/03_ham_result.png) |

## Features

**Website (FastAPI)**
- Paste any message and get a verdict with a spam probability
- Adjustable sensitivity slider that changes the decision threshold
- Highlighted message showing which words decided the result
- Signals checked: link, phone number, money, pressure words, capitals
- Model accuracy section with metrics, model comparison and confusion matrix

**Dashboard (Streamlit)**
- Data analysis charts, word clouds, model comparison, ROC and PR curves, and an error analysis tab

## Tech stack

Python, pandas, scikit-learn, FastAPI, Uvicorn, Streamlit, Plotly, HTML, CSS and JavaScript

## Project structure

```
spam-classifier/
├── static/index.html      # website
├── server.py              # FastAPI backend
├── app.py                 # Streamlit dashboard
├── data.py                # loading and cleaning
├── features.py            # feature engineering
├── eda.py                 # EDA charts
├── train.py               # trains and compares models
├── evaluate.py            # evaluation charts
├── error_analysis.py      # error table and reasons
├── spam.csv               # dataset
├── spam_model.joblib      # saved best model
├── results.json           # saved metrics
├── test_predictions.csv   # saved test predictions
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (creates spam_model.joblib, results.json, test_predictions.csv)
python train.py

# 3. Start the website, then open http://127.0.0.1:8000
uvicorn server:app --reload

# Optional: the analysis dashboard
streamlit run app.py
```

## Limitations and future work

- The model depends on vocabulary, so spam that avoids typical wording can slip through.
- The dataset is small, English-only and dated, so modern phishing messages may be underrepresented.
- Possible improvements: character n-grams, transformer embeddings (e.g. BERT), and a larger multilingual dataset.

## Author

`<FILL: your name, college, year>`
