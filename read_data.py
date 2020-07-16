#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:43:51 2020

@author: yansun
"""
from pandas import Series, DataFrame
import pandas as pd

news_senti_data = pd.read_csv("all-data.csv",header=None,names = ['senti','headline'])
print(news_senti_data.head())