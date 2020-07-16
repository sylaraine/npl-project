#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:43:51 2020

@author: yansun
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords

# =============================================================================
# Encoding problem: export csv file in utf-8 format
# Add column names
# Do not parse in 'index = False' or the apply function will not work
# https://www.shanelynn.ie/python-pandas-read_csv-load-data-from-csv-files/
# =============================================================================
news_senti_data = pd.read_csv("all-data.csv",header=None,names = ['senti','headline'])

# =============================================================================
# Pre-process headlines so that each healine consists meaning words that are stored in a list
# https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/
# =============================================================================

news_senti_data['headline'] = news_senti_data['headline'].str.lower()

def tokenize_headline(row):
    headline = row['headline']
    tokens = nltk.word_tokenize(headline)
    words = [w for w in tokens if w.isalpha()]
    return words

news_senti_data['tokens'] = news_senti_data.apply(tokenize_headline,axis=1)

#print(news_senti_data['tokens'].head())

stop_words = set(stopwords.words('english'))

def remove_stopwords(row):
    words = row['tokens']
    meaningful_words = [w for w in words if w not in stop_words]
    return meaningful_words

news_senti_data['meaningful_words'] = news_senti_data.apply(remove_stopwords,axis=1)

#print(news_senti_data['meaningful_words'].head())

# Jul 15 Github
