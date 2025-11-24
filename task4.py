# ----------------------------------------------------------
# TASK-04: Sentiment Analysis on Social Media Data
# Analyze and visualize sentiment patterns in social media posts.
# ----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import re

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
path = r"D:\Prodigy\task4\data-task4.csv"   
df = pd.read_csv(path)

print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())

# ----------------------------------------------------------
# FIX COLUMN NAMES
# ----------------------------------------------------------
df.columns = ["id", "topic", "label", "text"]

# Drop missing text rows
df = df.dropna(subset=["text"])

# ----------------------------------------------------------
# CLEAN TEXT FUNCTION
# ----------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)  # URLs
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Special chars
    text = text.lower()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ----------------------------------------------------------
# SENTIMENT ANALYSIS
# ----------------------------------------------------------
def get_sentiment(sentence):
    return TextBlob(sentence).sentiment.polarity

df["sentiment_score"] = df["clean_text"].apply(get_sentiment)

def categorize(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["sentiment_score"].apply(categorize)

print("\nSentiment Counts:\n", df["sentiment"].value_counts())

# ----------------------------------------------------------
# VISUALIZE SENTIMENT DISTRIBUTION
# ----------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=df["sentiment"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# ----------------------------------------------------------
# WORDCLOUD - ALL TEXT
# ----------------------------------------------------------
all_words = " ".join(df["clean_text"])

wc = WordCloud(width=1200, height=600, stopwords=STOPWORDS).generate(all_words)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of All Posts")
plt.show()

# ----------------------------------------------------------
# WORDCLOUD BY SENTIMENT
# ----------------------------------------------------------
for s in ["Positive", "Negative", "Neutral"]:
    words = " ".join(df[df["sentiment"] == s]["clean_text"])

    if len(words.strip()) == 0:
        continue

    wc = WordCloud(width=1200, height=600, stopwords=STOPWORDS).generate(words)

    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{s} WordCloud")
    plt.show()

print("\nAnalysis Completed Successfully!")
