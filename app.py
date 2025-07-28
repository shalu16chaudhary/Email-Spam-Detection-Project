from flask import Flask, request, jsonify, render_template
import joblib
import re
from flask_cors import CORS

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Clean the input text
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = "🚫 Spam" if prediction == 0 else "✅ Not Spam"
    return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
