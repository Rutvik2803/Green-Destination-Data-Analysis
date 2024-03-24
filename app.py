import pandas as pd
import numpy as np
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

DataNews = pd.read_csv('fake.csv')

DataNews = DataNews.fillna('')

DataNews['content'] = DataNews['author']+' '+DataNews['title']

x = DataNews.drop(columns='label',axis=1)
y = DataNews['label']

Stemmed_Porter = PorterStemmer()

def stemmed(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [Stemmed_Porter.stem(word) for word in stemmed_content if not word in stopwords.words('English')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

DataNews['content'] = DataNews['content'].apply(stemmed)

x = DataNews['content'].values
y = DataNews['label'].values
vectorizer = TfidfVectorizer()
vectorizer.fit(x)

x = vectorizer.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2,stratify=y)
model = LogisticRegression()
model.fit(x_train,y_train)


st.title("NEWS PREDICTOR")
input_text = st.text_input("ENTER NEWS")

def predication(input_text):
    input_data = vectorizer.transform(input_text)
    predication = model.predict(input_data)
    return predication

if input_text:
    pre = predication(input_text)
    if pre == 1:
        st.write("NEWS IS REAL :-)")
    else:
        st.write("NEWS IS FAKE!!!")



