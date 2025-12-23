from flask import Flask, request, jsonify
import pickle

MODEL_PATH = "../model/ticket_classifier.pkl"

app = Flask(__name__)

# Load model once at startup
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    subject = data.get("subject", "")
    description = data.get("description", "")

    text = subject + " " + description
    category = model.predict([text])[0]

    return jsonify({
        "predicted_category": category
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
