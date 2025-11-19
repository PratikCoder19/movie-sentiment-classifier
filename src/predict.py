import pickle
from preprocess import clean_text

# Load model + vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_sentiment(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]
    return "Positive 😊" if pred == 1 else "Negative 😡"


# Example usage:
#print(predict_sentiment("The movie was amazing with brilliant acting!"))
