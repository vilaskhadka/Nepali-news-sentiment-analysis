import streamlit as st
import pandas as pd
from nepali_stemmer.stemmer import NepStemmer
from nltk.corpus import stopwords
import pickle

st.set_page_config(layout='wide')
tfidf = pickle.load(open('/Users/IT Folders/Python/Deerwalk/Projects/News sentiment analysis/vectorizer.pkl', 'rb'))
model = pickle.load(open('/Users/IT Folders/Python/Deerwalk/Projects/News sentiment analysis/model.pkl', 'rb'))

st.write(' # Nepali News Sentiment Analysis')

def preprocess(sentence):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~।/'
    data1 = ' '.join([word for word in sentence.split() if word not in punctuations])
    data2 = ''.join((word for word in data1 if not word.isdigit()))
    data3 = NepStemmer().stem(data2)
    stopword = stopwords.words('nepali')
    data4 = ' '.join([word for word in data3.split() if word not in stopword])
    return data4


input_news = st.text_input('Enter news heading')

if st.button('Predict'):
    transform_news = preprocess(input_news)
    vectorize_news = tfidf.transform([transform_news])
    sentiment_pred = model.predict(vectorize_news)[0]
    if sentiment_pred ==1:
        st.header('Positive News')
    if sentiment_pred ==0:
        st.header('Neutral Sentiment News')
    if sentiment_pred ==-1:
        st.header('PNegative News')


col1, col2 ,col3, col4, col5 =st.columns((5))
news=pd.read_excel ('/Users/IT Folders/Python/Deerwalk/Projects/News sentiment analysis/news.xlsx')
with col1:
    st.write('###### Province 1')
    df=pd.DataFrame(news[news['Province']==1])
    df=df.drop(columns=['final Headlines','Unnamed: 0'])
    st.dataframe(df, hide_index= True)
with col2:
    st.write('###### Province 2')
    df=pd.DataFrame(news[news['Province']==2])
    df=df.drop(columns=['final Headlines','Unnamed: 0'])
    st.dataframe(df, hide_index= True)
with col3:
    st.write('###### Province 3')
    df=pd.DataFrame(news[news['Province']==3])
    df=df.drop(columns=['final Headlines','Unnamed: 0'])
    st.dataframe(df, hide_index= True)
with col4:
    st.write('###### Province 4')
    df=pd.DataFrame(news[news['Province']==4])
    df=df.drop(columns=['final Headlines','Unnamed: 0'])
    st.dataframe(df, hide_index= True)
with col5:
    st.write('###### Province 5')
    df=pd.DataFrame(news[news['Province']==5])
    df=df.drop(columns=['final Headlines','Unnamed: 0'])
    st.dataframe(df, hide_index= True)