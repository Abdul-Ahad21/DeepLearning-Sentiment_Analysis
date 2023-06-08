#STEP 1
import streamlit as st 
import pandas as pd
import seaborn as sns
import transformers
from transformers import pipeline
import re
import matplotlib.pyplot as plt

#STEP 2 (UNREQUIRED)

#STEP 3
st.title("Sentiment Analysis using StreamLit")
st.write("Enter some text and I'll tell you its sentiment.")

text = st.text_input("Enter some text:")

model = pipeline("sentiment-analysis")

#STEP 4
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    return text

def analyze_sentiment(text):
    text = clean_text(text)
    result = model(text, truncation=True)
    if result[0]['label'] == 'NEGATIVE':
        return 1 - result[0]['score'] 
    else:
        return result[0]['score']

def visualize_sentiment(df, fig=None):
    sns.set_style('darkgrid')
    chart = sns.barplot(x='sentiment', y='count', data=df)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    if fig:
        st.pyplot(fig)
    else:
        st.pyplot()

if text:
    score = analyze_sentiment(text)
    st.write("Sentiment polarity score:", score)
    df = pd.DataFrame({'sentiment': ['positive', 'negative'], 'count': [max(round(score, 2), 0), max(round(1 - score, 2), 0)]})
    fig, ax = plt.subplots()
    visualize_sentiment(df, fig=fig)

    #STEP 5
    # RUNNN!