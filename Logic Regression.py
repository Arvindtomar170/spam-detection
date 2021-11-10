
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

message_data = pd.read_csv("spam1.csv",encoding = "latin")
message_data.head()
message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})


message_data_copy = message_data['message'].copy()

def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)
message_data_copy = message_data_copy.apply(text_preprocess)
vectorizer = TfidfVectorizer(stop_words='english')
message_mat = vectorizer.fit_transform(message_data_copy)
message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat,message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)
Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
print(accuracy_score(spam_nospam_test,pred))