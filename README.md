# SpamLens: SMS Spam Classifier

A website where you paste a text message and get a verdict (spam or ham), the signals behind it, and the exact words that decided it. A second Streamlit dashboard holds the full data analysis and model comparison.

![Home](screenshots/01_home.png)

---

## Problem Statement

Spam messages waste time and are used for fraud. The goal is to detect spam SMS accurately while keeping false alarms low, so that real messages are not blocked.

## Dataset

- **SMS Spam Collection** (UCI Machine Learning Repository, via Kaggle)
- 5,572 messages originally, **5,169 after removing duplicates**
- 4,516 ham and 653 spam, so the data is **imbalanced** (about 13% spam).

## Approach

1. **Cleaning:** Removed duplicates and converted labels to 0/1.
2. **Feature Engineering:** Message length, word count, digit count, URL flag, currency flag, and capital-letter ratio.
3. **Text Vectorisation:** TF-IDF with unigrams and bigrams, with English stop words removed.
4. **Models:** Multinomial Naive Bayes and Logistic Regression (balanced class weights), each trained with and without the extra features, giving four models.
5. **Evaluation:** Stratified 80/20 split, evaluated using precision, recall, F1, ROC-AUC, and PR-AUC. Accuracy alone is misleading here because a model that always says "ham" would score about 87%.
6. **Error Analysis:** Reviewed the messages the models classified incorrectly.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Naive Bayes (TF-IDF) | 0.969 | 1.000 | 0.756 | 0.861 |
| Logistic Regression (TF-IDF) | 0.977 | 0.928 | 0.885 | 0.906 |
| Naive Bayes (TF-IDF + features) | 0.978 | 0.974 | 0.847 | 0.906 |
| Logistic Regression (TF-IDF + features) | 0.981 | 0.937 | 0.908 | 0.922 |

**Best model:** Logistic Regression (TF-IDF + features) with an F1 of 0.922.

![Accuracy section](screenshots/04_accuracy.png)

### Key Findings

- Spam messages are much longer and contain far more digits than ham: they average 138 characters and 15.4 digits, versus 70 characters and 0.3 digits for ham.
- 18% of spam contains a URL versus 0% of ham, and 33% of spam contains a currency symbol versus 0% of ham.
- Adding the handcrafted features raised F1 from 0.861 to 0.906 (+0.045) for Naive Bayes and from 0.906 to 0.922 (+0.016) for Logistic Regression.
- Naive Bayes had perfect precision (1.000) but low recall (0.756), so it never flagged a real message but missed about a quarter of the spam. Logistic Regression balanced the two better.

## Error Analysis

On the test set the best model made 20 errors out of 1,034 messages: 12 spam messages were missed and 8 real messages were flagged as spam.

- **Missed Spam:** Mostly written like casual conversation, with no URL, currency symbol, or typical trigger words.
- **Wrongly Flagged Ham:** Mostly contained phone numbers, money amounts, or words like "call" and "free".

*Note: These two patterns should be verified against the actual misclassified messages in `errors.csv` before final submission.*

## Explainability

For every message, the website removes each word in turn and measures how much the spam probability drops. Words that raise the probability are shown in red and words that lower it in green. This works with any model.

| Spam Result | Ham Result |
|---|---|
| ![Spam](screenshots/02_spam_result.png) | ![Ham](screenshots/03_ham_result.png) |

## Features

### Website (FastAPI)

- Paste any message and get a verdict with a spam probability.
- Adjustable sensitivity slider that changes the decision threshold.
- Highlighted message showing which words decided the result.
- Signals checked: link, phone number, money, pressure words, and capitals.
- Model accuracy section with metrics, model comparison, and confusion matrix.

### Dashboard (Streamlit)

- Data analysis charts and visualisations.
- Word clouds.
- Model comparison.
- ROC and PR curves.
- Error analysis tab.

## Tech Stack

Python, pandas, scikit-learn, FastAPI, Uvicorn, Streamlit, Plotly, HTML, CSS, and JavaScript.

## Project Structure

```text
spam-classifier/
├── static/index.html          # Website
├── server.py                  # FastAPI backend
├── app.py                     # Streamlit dashboard
├── data.py                    # Loading and cleaning
├── features.py                # Feature engineering
├── eda.py                     # EDA charts
├── train.py                   # Trains and compares models
├── evaluate.py                # Evaluation charts
├── error_analysis.py          # Error table and reasons
├── spam.csv                   # Dataset
├── spam_model.joblib          # Saved best model
├── results.json               # Saved metrics
├── test_predictions.csv       # Saved test predictions
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python train.py
```

This creates `spam_model.joblib`, `results.json`, and `test_predictions.csv`.

### 3. Start the Website

```bash
uvicorn server:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### 4. Run the Analysis Dashboard (Optional)

```bash
streamlit run app.py
```

## Limitations and Future Work

- The model depends on vocabulary, so spam that avoids typical wording can slip through.
- The dataset is small, English-only, and dated, so modern phishing messages may be underrepresented.
- Possible improvements include character n-grams, transformer embeddings (e.g., BERT), and a larger multilingual dataset.

## Author

**Ujvala Kukatla**   
Sreenidhi Institute of Science and Technology 