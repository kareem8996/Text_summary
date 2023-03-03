#!/usr/bin/env python
# coding: utf-8

# # deployment

#necessary libraries for deployment
import streamlit as st
import requests
from streamlit_lottie import st_lottie

#lottie function
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


#Page header layout
from PIL import Image
image = Image.open('ApplAiOnly_Logo.png')
right_column, left_column = st.columns(2)
with right_column:
    st.title("                   ")
    st.title("                   ")
    st.title("                   ")
    st.title("Text Summarizer")
with left_column:   
    st.image(image)    
    

with st.form(key="form1"):
    text=st.text_input(label="Enter the required text")
    right_column, left_column = st.columns(2)
    with right_column:
        minl=st.text_input(label="Enter the minmum length for abstractive summary")
        submit=st.form_submit_button(label="Abstractive Summary")
    with left_column:
        maxl=st.text_input(label="Enter the maximum length for abstractive summary")
        submit2=st.form_submit_button(label="Extractive Summary")
    


# 1)Abstractive Summary

def Asummarize(text,minl,maxl):
    # importing the needed libraries
    import torch
    import json 
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

    # # Setting the model
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    # # preprocessing the text
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=10,
                                        no_repeat_ngram_size=2,
                                        min_length=minl,
                                        max_length=maxl,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output


# 2)Extractive Summary


def Esummarize(text):

    # Installing the required libraries
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Creating a frequency table to keep the
    # score of each word

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq



    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (0.8 * average)):
            summary += " " + sentence
    return summary



#Output of the summary for the given type

if(submit==True):
    if(len(text.split())>=100):
        minl=int(minl)
        maxl=int(maxl)
        outputt=Asummarize(text,minl,maxl)
        st.subheader(outputt, anchor=None)
        submit=False
    else:
        st.subheader("Abstractive summary needs more words to summarize :(", anchor=None)
if(submit2==True):
    outputt2=Esummarize(text)
    st.subheader(outputt2, anchor=None)
    submit2=False


#To adjust the lottie under the summary
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
st.subheader("""             """)
#Robot lottie file
animation_header = load_lottie("https://assets3.lottiefiles.com/private_files/lf30_ssm93drs.json")
st_lottie(animation_header, speed=1, height=200, key="forth")




#footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: white;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: blue;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://linkedin.com/in/youssef-salem3" target="_blank">Youssef Salem</a>
<a style='display: block; text-align: center;' href="https://linkedin.com/in/nour-ahmeddd-" target="_blank">Nour Ahmed</a>
</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
