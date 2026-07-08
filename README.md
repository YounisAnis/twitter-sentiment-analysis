# Twitter Sentiment Analysis: Cryptocurrency Tweets (January 2021)

I built this as a complete data engineering and machine learning project using 33,000+ crypto tweets from January 2021, the month Bitcoin crossed $40,000 for the first time.

The project covers the full pipeline: loading a misformatted dataset, cleaning the noisy tweet text, exploring patterns in the data, comparing two sentiment tools, and training three machine learning models on two different classification tasks.

---

## What This Project Covers

| Section         | What It Does                                                                        |
| --------------- | ----------------------------------------------------------------------------------- |
| Part 1          | Discovers the dataset is JSONL disguised as a CSV file                              |
| Part 2          | Flattens deeply nested Twitter JSON using`pd.json_normalize()`                    |
| Part 3          | Fixes Unix millisecond timestamps that were parsed incorrectly                      |
| Part 4          | Cleans tweet text: removes mentions, URLs, special characters, and applies stemming |
| Part 5          | Explores hashtags, activity by hour, and TextBlob sentiment                         |
| Part 5 Extended | Adds VADER sentiment, word clouds, and hourly sentiment trends                      |
| Part 6          | Trains three ML models to predict follower count category from tweet text           |
| Part 6 Extended | Trains the same three models to predict tweet sentiment — a more natural task      |
| Part 7          | Discusses results, overfitting, and what accuracy really means                      |
| Part 8          | Reflects on why data engineering is most of the work                                |

---

## Models Used

I trained and compared three models on both classification tasks:

- Linear Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## Features Added Beyond Basic Analysis

- VADER vs TextBlob comparison with agreement rate
- Word clouds for positive and negative tweets (separate visuals)
- Hourly sentiment trend: stacked volume bar and positive percentage line chart
- Random Forest feature importance: top 20 TF-IDF tokens
- 5-fold stratified cross-validation with box plot
- Sentiment as classification target using VADER labels as ground truth
- Confusion matrix for the best performing sentiment model
- Classification report with precision, recall, and F1-score per class

---

## Requirements

```bash
pip install pandas numpy matplotlib plotly scikit-learn xgboost
pip install textblob vaderSentiment wordcloud nltk
```

Run this once inside Python to download the required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
```

---

## How to Run

1. Clone this repository
2. Download the dataset (see the Dataset section below)
3. Place the file inside the `data/` folder and name it `twitter_data.csv`
4. Open `twitter_sentiment_analysis.ipynb` in Jupyter Notebook or JupyterLab
5. Run all cells from top to bottom

---

## Dataset

**Size:** approximately 182 MB
**Format:** Newline-delimited JSON (JSONL), despite the `.csv` file extension

The dataset is too large to store directly in this GitHub repository. GitHub blocks files above 100 MB. Follow the steps below to get it.

### Where to Download

The dataset I used in this project is a collection of cryptocurrency tweets from January 2021. Search Kaggle for:

> Cryptocurrency Twitter Dataset January 2021

Or use this direct Kaggle link if available:

```
https://www.kaggle.com/datasets/muhammadyounasdf/cryptocurrency-tweets-january-2021
```

After downloading, place the file here:

```
data/
    twitter_data.csv
```

### Why the Dataset Is Not in This Repo

GitHub has a hard limit of 100 MB per file. This dataset is 182 MB, so it is above that limit. If I try to push it, the upload will fail. The `data/` folder is listed in `.gitignore` so Git ignores the file automatically.

---

## Project Structure

```
your-repo/
    twitter_sentiment_analysis.ipynb   # Main notebook (40 cells)
    README.md                          # This file
    .gitignore                         # Excludes the dataset from Git
    data/
        .gitkeep                       # Keeps the empty folder tracked by Git
        twitter_data.csv               # Place downloaded dataset here (ignored by Git)
```

---

## Results Summary

**Task 1: Follower Count Prediction (Low / Medium / High)**

I tested Linear SVM, Random Forest, and XGBoost on this task. The Random Forest came out on top with 48.2% test accuracy. But the training accuracy was 73.7%, so that gap shows clear overfitting. The task is also inherently difficult because the words someone uses in a tweet are not a strong predictor of how many followers they have.

**Task 2: Sentiment Classification (Positive / Neutral / Negative)**

I ran the same three models using VADER labels as the ground truth target. This is a better-framed NLP task because tweet vocabulary is directly connected to emotional tone. The notebook includes a full confusion matrix and a classification report with precision, recall, and F1-score for each class.

---

## Key Learnings

- Real-world data rarely comes in the format you expect
- I always check timestamps before doing any time-based analysis, the unit is not always what it looks like
- VADER is a better tool than TextBlob for social media text because it understands slang, capitalization, and punctuation patterns
- Accuracy alone is not enough, F1-score gives a fairer picture on imbalanced classes
- The way you define the classification task matters just as much as the model you choose
- Data engineering takes most of the time, but it is also where most of the real learning happens

---

## Author

**Younis Anis**
[LinkedIn](https://www.linkedin.com/in/younis-anis-85ba27250/) | younasanis6@gmail.com
