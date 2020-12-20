import pickle
import numpy as np  # array of array
import pandas as pd  # read & write file
import re  # regular expression provessing
import nltk  # natural language toolkit
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nlpaug.augmenter.char as nac
import requests

nltk.download('stopwords')
stops = stopwords.words('english')
stemmer = SnowballStemmer('english')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
# Step 1- Flask Instantiation
from flask import Flask, render_template, request
app = Flask('Fake Job Posting')


# Step 2- Setting up Routes
@app.route('/')
def index():
    return render_template('index.html')


def description_to_words(raw_description):
    # 1. Delete HTML
    description_text = BeautifulSoup(raw_description, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', description_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return (' ').join(stemming_words)


def cleaning(description_feature):
    # remove all the special characters
    cleaned_description_feature = re.sub(r'\W', ' ', str(description_feature))
    # remove all single characters
    cleaned_description_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', cleaned_description_feature)  # \s single character
    # remove single characters from the start
    cleaned_description_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', cleaned_description_feature)  # ^ single digits
    # remove multiple spaces with single space
    cleaned_description_feature = re.sub(r'\s+', ' ', cleaned_description_feature, flags=re.I)
    # removing prefixed 'b'
    cleaned_description_feature = re.sub(r'^b\s+', '', cleaned_description_feature)  # ^b alphanumeric
    # converting to lowercase
    cleaned_description_feature = cleaned_description_feature.lower()
    #
    cleaned_description_feature = re.sub(r'\d+', '', cleaned_description_feature)
    blob = TextBlob(cleaned_description_feature)
    blob.correct
    return str(blob)


@app.route('/', methods=["POST", "GET"])
def results():
    if request.method == 'POST':
        model = pickle.load(open('modelSVM.pkl', 'rb'))
        txt_field = request.form['postMessage']
        message = txt_field
        txt_field = description_to_words(txt_field)
        txt_field = cleaning(txt_field)
        cleanedData=pickle.load(open('cleanedtxt.pkl', 'rb'))
        cleanedData.append(txt_field)
        Vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
        cleanedData = Vectorizer.fit_transform(cleanedData).toarray()
        cleaned_txt_field = cleanedData[-1]
        y_pred = model.predict(cleaned_txt_field.reshape(1, -1))
        print(y_pred[0])
    return render_template('response.html', txt_field=message, y_pred=y_pred[0])

# Step 3- Run the application
app.run("127.0.0.1", "5000", debug=True)
