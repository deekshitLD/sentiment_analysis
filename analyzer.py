import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('sentiment_data.csv')

# Clean the text data
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stop_words]
    text = ' '.join(text)
    return text

data['text'] = data['text'].apply(clean_text)

# Convert the text data into a numerical representation using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])

# Split the data into training and testing sets
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the model's accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
