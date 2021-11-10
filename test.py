# app=Flask(__name__)
# @app.route('/',methods=['GET','POST'])
# def func():
#     if request.method=='POST':
#         text=request.form['text']
#         res=pred(text)
#         return render_template('',msg=res)
#     return render_template('')

import nltk
nltk.download('stopwords')