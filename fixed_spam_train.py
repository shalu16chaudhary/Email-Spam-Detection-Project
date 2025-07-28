import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load & Prepare Data
df = pd.read_csv("spam.csv", encoding='latin1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
df['Category'] = df['Category'].map({'ham': 1, 'spam': 0}).astype(int)

# Step 2: Clean Text
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

df['Message'] = df['Message'].apply(clean_text)

# Step 3: Train-test split
X = df['Message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train all models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Linear Kernel)": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

metrics = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    f1 = f1_score(y_test, y_pred)
    metrics[name] = f1

best_model_name = max(metrics, key=metrics.get)
best_model = models[best_model_name]
print(f"✅ Best Model: {best_model_name}")

# Step 6: Save
joblib.dump(best_model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Trained model and vectorizer saved successfully.")
