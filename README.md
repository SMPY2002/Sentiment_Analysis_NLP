# Sentiment Analysis using Naive Bayes Method

## Overview

Welcome to the Sentiment Analysis project! In this project, we implement a sentiment analysis model using the Naive Bayes method. What sets this project apart is that it does not rely on any standard machine learning libraries specifically designed for Naive Bayes classification. The goal is to showcase a manual, step-by-step implementation of the algorithm, providing a deeper understanding of the underlying principles.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Word Clouds](#word-clouds)
6. [Word Frequency Analysis](#word-frequency-analysis)
7. [Naive Bayes Model](#naive-bayes-model)
8. [Testing Phase](#testing-phase)
9. [Accuracy on Unseen Tweets](#accuracy-on-unseen-tweets)
10. [Conclusion](#conclusion)

## Introduction

Sentiment Analysis is a Natural Language Processing (NLP) task that involves determining the sentiment expressed in a piece of text, such as positive, negative, or neutral. The Naive Bayes method is a probabilistic algorithm commonly used for classification tasks, making it suitable for sentiment analysis.

## Dataset

We start by loading the dataset from "twitter_data.csv" and performing initial exploratory data analysis to understand its structure.

```python
# Section: Dataset
# Code snippet for loading and exploring the dataset

Data Preprocessing
Data preprocessing is a crucial step in NLP tasks. We perform various preprocessing steps, including removing stopwords, stemming, and handling class imbalance.
# Section: Dataset
## Code snippet for loading and exploring the dataset
```python
# Importing necessary libraries
import math 
import numpy as np 
import pandas as pd 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# NLP Libraries 
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset 
df = pd.read_csv("twitter_data.csv", encoding="latin")
# Creating a copy of dataframe df , named new_df which contains only 100000 examples

new_dataframe_class_0 = df[df['target'] == 0].head(50000)
new_dataframe_class_1 = df[df['target'] == 1].head(50000)

# Concatenate the two subsets to create the final DataFrame with equal counts of both classes
new_df = pd.concat([new_dataframe_class_0, new_dataframe_class_1])

# Shuffle the rows to randomize the order
new_df = new_df.sample(frac=1).reset_index(drop=True)

# Get the default NLTK stopwords list
default_stopwords = set(stopwords.words('english'))

# Add additional stopwords that you want to keep (e.g., negation words)
custom_stopwords = set(['not', 'no', 'against', 'nor'])

# Create a set that excludes words in custom_stopwords from default_stopwords
final_stopwords = default_stopwords - custom_stopwords

print(final_stopwords)

# Creates an instance of the PorterStemmer.
port_stem = PorterStemmer()

# Define the Stemming Function and perform operations 
def stemming(content):
    # Remove non-alphabetic characters using regular expression
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    
    # Tokenize the text into words
    stemmed_content = stemmed_content.split()
    
    # Apply stemming using Porter Stemmer, and exclude stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in final_stopwords]
    
    # Join the stemmed words into a single string
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content

# Applies the stemming function to the 'text' column of the DataFrame and creates a new column 'stemmed_content' containing the preprocessed text.
new_df['stemmed_content'] = new_df['text'].apply(stemming)

