# ✉️ Spam Email Detector Web App

This is a simple and powerful **Spam Email Detection** web application built using **Flask**, **Scikit-learn**, and **Machine Learning**. It uses a trained classification model to determine whether an input message is **Spam** or **Not Spam**.

🌐 **Live Demo:** https://email-spam-detection-project.onrender.com 


---

## 🚀 Features

- 🔍 Classifies any text message as **Spam** or **Not Spam**
- 📊 Uses a trained machine learning model (Multinomial Naive Bayes or similar)
- 🧠 Preprocessing with TF-IDF vectorizer
- 🌐 Hosted live using **Render**
- 🔁 CORS Enabled for cross-origin requests

---

## 🧠 Tech Stack

| Technology | Description |
|------------|-------------|
| Python     | Programming Language |
| Flask      | Backend Web Framework |
| Scikit-learn | Machine Learning Model |
| Joblib     | Model serialization |
| Flask-CORS | To allow cross-origin requests |
| Render     | Hosting platform |

---

## 📁 Project Structure

```bash
.
├── app.py               # Main Flask application
├── model.pkl            # Trained ML model
├── vectorizer.pkl       # TF-IDF Vectorizer
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
