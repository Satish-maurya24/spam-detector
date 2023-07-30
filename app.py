import streamlit as st
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle as pkl
ps = PorterStemmer()
import string
model = pkl.load(open("model.pkl" , 'rb'))



st.title("Spam-detector")

input_message = st.text_area("enter the message")


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

transformed_message = transform_text(input_message)

tfdif = pkl.load(open('vectorizer.pkl' , 'rb'))

vectorized_message = tfdif.transform([transformed_message])

result = model.predict(vectorized_message)
if st.button('predict'):
  if result == 1:
      st.header("Spam")
  else:
      st.header("not spam")

