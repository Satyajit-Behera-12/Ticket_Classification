import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Cleans and normalizes input text
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines subject and description and applies cleaning
    """
    df["text"] = df["Subject"] + " " + df["Description"]
    df["text"] = df["text"].apply(clean_text)
    return df
