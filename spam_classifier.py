# 📌 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 📌 Load Dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 📌 Encode Labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 📌 Text Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 📌 Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# 📌 Prediction
y_pred = model.predict(X_test)

# 📌 Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 📌 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 📌 Predict Custom Message
sample_msg = ["Win $1000 now! Click here to claim your prize."]
sample_vector = vectorizer.transform(sample_msg)
prediction = model.predict(sample_vector)
print("\n🔍 Message:", sample_msg[0])
print("🧠 Prediction:", "Spam" if prediction[0] == 1 else "Ham")
