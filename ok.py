import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

print('hi')

# Load datasets
data_fake = pd.read_csv('Datasets/Fake.csv')
data_true = pd.read_csv('Datasets/True.csv')

# Assign class labels
data_fake["class"] = 0
data_true['class'] = 1

# Remove last 10 entries for manual testing
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)

# Correctly set the class for manual testing data
data_fake_manual_testing.loc[:, 'class'] = 0
data_true_manual_testing.loc[:, 'class'] = 1

# Combine the datasets
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)


# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# Apply preprocessing to the text data
data['text'] = data['text'].apply(wordopt)

# Split the data into training and testing sets
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorize the text data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train the models
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

# Save the models and vectorizer
with open('models.pkl', 'wb') as file:
    pickle.dump((LR, DT, GB, RF, vectorization), file)


def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"


def manual_testing(news):
    with open('models.pkl', 'rb') as file:
        LR, DT, GB, RF, vectorization = pickle.load(file)
    
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction:{}".format(
        output_label(pred_LR[0]),
        output_label(pred_DT[0]),
        output_label(pred_GB[0]),
        output_label(pred_RF[0])
    ))


# Example usage
news = str(input("Enter news text: "))
manual_testing(news)