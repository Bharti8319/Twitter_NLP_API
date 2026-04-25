# IMPORTS
from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)

# LOAD MODEL + TOKENIZER
# Updated to use Twitter.h5 instead of model.h5 based on directory contents
model = load_model("Twitter.h5", compile=False)

class Keras3To2Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'keras.src.legacy.preprocessing.text':
            module = 'keras_preprocessing.text'
        return super().find_class(module, name)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = Keras3To2Unpickler(f).load()

# PREPROCESS FUNCTION
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# RISK FUNCTION
def calculate_final_risk(text, pred_prob, pred_label):
    text_lower = text.lower()
    risk = 50 if pred_label == 0 else 10
    
    if pred_label == 0:
        risk += (1 - pred_prob) * 50
    else:
        risk += pred_prob * 20
    
    risk += text_lower.count("!") * 2
    caps = sum(1 for c in text if c.isupper())
    risk += min(caps, 10)
    
    return min(risk, 100)

# API ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.get_json()
    text = data["text"]
    
    clean = preprocess(text)
    
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=100)
    
    pred_prob = model.predict(padded)[0][0]
    
    pred_label = 1 if pred_prob > 0.5 else 0
    label = "normal speech" if pred_label == 1 else "hate speech"
    
    risk = calculate_final_risk(text, pred_prob, pred_label)
    
    return jsonify({
        "text": text,
        "prediction": label,
        "risk_score": round(risk, 2),
        "confidence": round(pred_prob * 100, 2)
    })


# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True)
