import pickle

MODEL_PATH = "../model/ticket_classifier.pkl"

def load_model():
    """
    Loads trained ML model from disk
    """
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

def predict_category(subject: str, description: str) -> str:
    """
    Predicts ticket category based on input text
    """
    model = load_model()
    text = subject + " " + description
    prediction = model.predict([text])
    return prediction[0]
