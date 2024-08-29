import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
# Load custom data
data = pd.read_csv('data.csv')
# Vectorize the questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])
# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X, data['answer'])
# Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
