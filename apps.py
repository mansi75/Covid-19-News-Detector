from flask import Flask,render_template,url_for,request
import pandas as pd 
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


from sklearn.linear_model import PassiveAggressiveClassifier

import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        #message = re.sub('\[.*?\]', '', message)
        #message = re.sub("\\W", '', message) 
        #message = re.sub('https?://\S+|www\.\S+', '', message)
        #message = re.sub('<.*?>+', '', message)
        #message = re.sub(r'[^\w\s]', '', message)
        #message = re.sub('\n', '', message)
        #message = re.sub('\w*\d\w*', '', message)
        message = re.sub('<[^>]*>', '', message)
        message = re.sub(r'[^\w\s]','', message)
        message = message.lower()  
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction) 
    



if __name__ == '__main__':
    app.run(debug=True)