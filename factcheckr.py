import pandas as pd
import re
import string
import pickle
import streamlit as st

# Load models and vectorizer from pickle file
with open('models.pkl', 'rb') as file:
    LR, DT, GB, RF, vectorization = pickle.load(file)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return {
        "Logistic Regression": output_label(pred_LR[0]),
        "Decision Tree": output_label(pred_DT[0]),
        "Gradient Boosting": output_label(pred_GB[0]),
        "Random Forest": output_label(pred_RF[0])
    }

# Streamlit app
st.title("FackCheckr™")

#st.write("Enter news text for classification:")

news = st.text_area("Enter news text for classification")

if st.button("Classify"):
    if news.strip() != "":
        predictions = manual_testing(news)
        st.write("**Predictions:**")
        st.write(f"**Logistic Regression:** {predictions['Logistic Regression']}")
        st.write(f"**Decision Tree:** {predictions['Decision Tree']}")
        st.write(f"**Gradient Boosting:** {predictions['Gradient Boosting']}")
        st.write(f"**Random Forest:** {predictions['Random Forest']}")
    else:
        st.write("Please enter some text to classify.")
