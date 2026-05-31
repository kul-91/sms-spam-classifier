import streamlit as st
import pickle
import  string
import nltk
from nltk.corpus import  stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message")


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

if st.button("Predict"):
    # preprocessing
    transformed_sms = transform_text(input_sms)

    # vectorizer
    vector_input = tfidf.transform([transformed_sms])

    # prediction
    result = model.predict(vector_input)

    # display
    if result[0] == 1:
        st.header("SMS is Spam")
    else :
        st.header("SMS is not Spam")