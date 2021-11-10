from flask import Flask,render_template,request,url_for
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix


data = pd.read_csv("spam2.csv", encoding = "latin-1")
data = data[['v1', 'v2']]
data = data.rename(columns = {'v1': 'label', 'v2': 'text'})

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

def review_messages(msg):
    msg = msg.lower()

    return msg

def review_messages2(msg):

    msg=msg.lower()
    nltk_pos = [tag[1] for tag in pos_tag(word_tokenize(msg))]
    msg = [tag[0] for tag in pos_tag(word_tokenize(msg))]
    wnpos = ['a' if tag[0] == 'J' else tag[0].lower() if tag[0] in ['N', 'R', 'V'] else 'n' for tag in nltk_pos]
    msg = " ".join([lemmatizer.lemmatize(word, wnpos[i]) for i, word in enumerate(msg)]) 
    msg = [word for word in msg.split() if word not in stopwords]
    return msg
data['text'] = data['text'].apply(review_messages)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.27, random_state = 1)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train) 
X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test) 
print(confusion_matrix(y_test, y_pred))
def pred(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def func():
    if request.method=='POST':
        text=request.form['Description']
        res=pred(text)
        return render_template('base.html',msg=res)
    return render_template('base.html')
if __name__=='__main__':
    app.run(debug=True)
    




