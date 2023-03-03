# Text summarization

Text summarization refers to the technique of shortening long pieces of text. The intention is to create a coherent and fluent summary having only the main points outlined in the document.

- In fact, the International Data Corporation (IDC) projects that the total amount of digital data circulating annually around the world would sprout from 4.4 zettabytes in 2013 to hit 180 zettabytes in 2025. That’s a lot of data!

- With such a big amount of data circulating in the digital space, there is need to develop machine learning algorithms that can automatically shorten longer texts and deliver accurate summaries that can fluently pass the intended messages.

- Furthermore, applying text summarization reduces reading time, accelerates the process of researching for information, and increases the amount of information that can fit in an area.

- Summarization can be classifed into two types Extractive and Abstractive.
Concatenation vs. Understanding. Words vs. Concepts. Reductive vs. Illuminative. Extractive vs. Abstractive.

## 1)Extractive Summarization:
Extractive summarization involves identifying important sections from text and generating them verbatim which produces a subset of sentences from the original text.

## 2)Abstractive Summarization:
Abstractive summarization uses natural language techniques to interpret and understand the important aspects of a text and generate a more “human” friendly summary.


To give an analogy, extractive summarization is like a highlighter, while abstractive summarization is like a pen.


## So how we managed to do it ?

For the Extractive summarization We used NLTK library and for the Abstractive summarization we used Transformers to use a pretrained model (t5) to create the abstractive summary.

Let's dive deep in the app:

* First the user enter the text that needs to be summarized.
* Second the user enter the minmum and maximum lengths for the abstractive summary.
* Finally choose the type of summarization you want.

## Install the required packages:

`pip install requrements.txt`

## Deployment

And here is the link to access the app: https://texttsummarizeryn.streamlit.app/ 
![WhatsApp Image 2023-03-03 at 14 50 00](https://user-images.githubusercontent.com/126875631/222775062-86aaadc7-cb97-48a7-adb7-2dc6ec5fbd7f.jpg)

## Demo

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/126875631/222781035-e8171e5a-17b6-4968-a1ce-cbed6a7e232d.gif)


