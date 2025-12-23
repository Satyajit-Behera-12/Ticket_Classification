import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_dataframe

DATA_PATH = "../data/sample_tickets.csv"
MODEL_PATH = "../model/ticket_classifier.pkl"

def train_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Preprocess text
    df = preprocess_dataframe(df)

    X = df["text"]
    y = df["Category"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ML Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    predictions = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Save model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(pipeline, file)

if __name__ == "__main__":
    train_model()
