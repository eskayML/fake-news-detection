import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# I found a simple dataset that lacks missing values
# which makes it easier for us to train a model using the data

# load the data
df = pd.read_csv('fake_or_real_news.csv')
# columns
print(df.columns)
# We'll take out the headline and use it to train our model ,
# So if the headline of a similar sample is used ,
# our model tries to correctly classify it is real or fake news


X = df['title']
y = df['label']

# split full data into parts for training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=.2)
model = Pipeline([
    ('tokenizer', CountVectorizer()),
    ('estimator', MultinomialNB())
])

# view a sample from the dataset
# print(df['title'][:5])

# fit the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

pickle.dump(model, open('naive_bayes_1.0.pkl', 'wb'))
