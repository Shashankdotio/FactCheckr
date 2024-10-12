import pandas as pd
import re
import string
import pickle

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
    with open('models.pkl', 'rb') as file:
        LR, DT, GB, RF, vectorization = pickle.load(file)
    
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    # Get probability predictions
    prob_LR = LR.predict_proba(new_xv_test)[0]
    prob_DT = DT.predict_proba(new_xv_test)[0]
    prob_GB = GB.predict_proba(new_xv_test)[0]
    prob_RF = RF.predict_proba(new_xv_test)[0]

    # Calculate percentages
    results = {
        "Logistic Regression": {"Fake": prob_LR[0] * 100, "Real": prob_LR[1] * 100},
        "Decision Tree": {"Fake": prob_DT[0] * 100, "Real": prob_DT[1] * 100},
        "Gradient Boosting": {"Fake": prob_GB[0] * 100, "Real": prob_GB[1] * 100},
        "Random Forest": {"Fake": prob_RF[0] * 100, "Real": prob_RF[1] * 100}
    }

    return results
