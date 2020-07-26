#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle
import spacy
import requests
import concurrent.futures
import wikipedia
import re
import itertools
from gensim.summarization.bm25 import BM25
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
import operator

st.write("""
# Question Answering using Wikipedia deployed using streamlit
For this project, I have built a question answering machine that takes the question or query,  perform document retrieval and search for relevant wikipedia pages, identify relevant passages and use BM25 to perform ranking of the passages to determine the answer or search query.
""")


# Functionalize model fittting
import pickle


def search_page(page_id):
    res = wikipedia.page(pageid=page_id)
    return res.content

def post_process(doc):
        pattern = '|'.join([
            '== References ==',
            '== Further reading ==',
            '== External links',
            '== See also ==',
            '== Sources ==',
            '== Notes ==',
            '== Further references ==',
            '== Footnotes ==',
            '=== Notes ===',
            '=== Sources ===',
            '=== Citations ===',
        ])
        p = re.compile(pattern)
        indices = [m.start() for m in p.finditer(doc)]
        min_idx = min(*indices, len(doc))
        return doc[:min_idx]

def preprocess(doc):
    passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
    return passages

#df =  user_input_features()
st.subheader('Question')
#st.write(df)
sample_question = st.selectbox("Sample Questions:", ["Who is the founder of Amazon?", "Where is the largest city in Japan?", "What language is spoken in Japan?"])

question = st.text_input('Input your question here and press the Start button.') 
#st.subheader('Result or answer to the question or query')
#filename = 'LogisticRegression'
#model = pickle.load(open(filename,'rb'))
#preprocess = pickle.load(open('preprocess_scale','rb'))
#pred = model.predict(preprocess.transform(df))
#st.write(pred)
#if question:
    #st.write(question)
if st.button('Start'):
    if question=='':
        question=sample_question
    my_bar = st.progress(0)
    my_bar.progress(25)

    # text preprocessing using spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    query = ' '.join(token.text for token in doc if token.pos_ in {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'})

    # using wikipedia api search using query
    params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json'
             }
    res = requests.get('https://en.wikipedia.org/w/api.php', params=params)
    pages = res.json()
    
    # extract relevant passages or documents
    with concurrent.futures.ThreadPoolExecutor() as executor:
        process_list = [executor.submit(search_page, page['pageid']) for page in pages['query']['search']]
        docs = [post_process(p.result()) for p in process_list]    
    
    # rank the corpus using bm25
    passages = list(itertools.chain(*map(preprocess, docs)))
    corpus = [[token.lemma_ for token in nlp(p)] for p in passages]
    bm25 = BM25(corpus)

    # extract the top 10 relevant passages
    topn=10
    tokens = [token.lemma_ for token in nlp(question)]
    scores = bm25.get_scores(tokens)
    pairs = [(s, i) for i, s in enumerate(scores)]
    pairs.sort(reverse=True)
    passages = [passages[i] for _, i in pairs[:topn]]
    # prepare the question and answering pipeline
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    qa_pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
    my_bar.progress(70)
    # obtain the answers
    answers = []
    for passage in passages:
        answer = qa_pipeline(question=question, context=passage)
        answer['text'] = passage
        answers.append(answer)
    answers.sort(key=operator.itemgetter('score'), reverse=True)
    my_bar.progress(100)
    # write the answer to the user
    st.write(answers[0]['answer'])