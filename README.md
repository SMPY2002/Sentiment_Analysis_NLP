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

## Necessary Libraries

```python
# Section: Dataset
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
```
## Dataset
We start by loading the dataset from "twitter_data.csv" which is present on kaggle and performing initial exploratory data analysis to understand its structure.
```python
# Load the dataset 
df = pd.read_csv("twitter_data.csv", encoding="latin")

# Creating a copy of dataframe df , named new_df which contains only 100000 examples
new_dataframe_class_0 = df[df['target'] == 0].head(50000)
new_dataframe_class_1 = df[df['target'] == 1].head(50000)

# Concatenate the two subsets to create the final DataFrame with equal counts of both classes
new_df = pd.concat([new_dataframe_class_0, new_dataframe_class_1])

# Shuffle the rows to randomize the order
new_df = new_df.sample(frac=1).reset_index(drop=True)
```
## Data Preprocessing
Data preprocessing is a crucial step in NLP tasks. We perform various preprocessing steps, including removing stopwords, stemming, and handling class imbalance.
```python
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
```
Output - 
  target	 text	                                                stemmed_content
0	0	Three cheers for fiber to the home... now we o...   	three cheer fiber home wait year
1	1	looking something new..	                                look someth new
2	1	Up having QT - feeling better this morning as ...	    qt feel better morn rememb god merci new everi...
3	0	Ugh. Riddler had his revenge on me....	                ugh riddler reveng
4	0	@lauren42 hope Yall have fun without me tomorr...	    lauren hope yall fun without tomorrow im jealo...

## Exploratory Data Analysis
Explore the dataset to gain insights into class distribution, null values, and other relevant information.
```python
# Information about dataset 
df.info()
print("\nPrinting the  first five rows of the dataset:")
df.head()

#Rename columns and reading dataset again
column_names = ['target','id','date','flag','user','text']
df= pd.read_csv('twitter_data.csv',names = column_names, encoding = 'latin')

# Printing the shape of dataset
print("Shape of the dataset is -")
df.shape

# Again checking the info of dataset 
df.info()
print("\nPrinting the  first five rows of the dataset:")
df.head()

# Checking if any null values present or not 
df.isnull().sum()

#CONVERT TARGET 4 TO 1
df.replace({'target':{4:1}}, inplace=True)

# Again checking class distribution of target column
df['target'].value_counts()

# dropping the rest of the column other than target and text 
df.drop(['id','date','flag','user'],axis=1,inplace=True)

# Again printing the dataset
df.head()
df.info()
```
Output
	target	text
0	0	@switchfoot http://twitpic.com/2y1zl - Awww, t...
1	0	is upset that he can't update his Facebook by ...
2	0	@Kenichan I dived many times for the ball. Man...
3	0	my whole body feels itchy and like its on fire
4	0	@nationwideclass no, it's not behaving at all....

Information of dataframe - 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1600000 entries, 0 to 1599999
Data columns (total 2 columns):
 *   Column  Non-Null Count    Dtype 
---  ------  --------------    ----- 
 0   target  1600000 non-null  int64 
 1   text    1600000 non-null  object
dtypes: int64(1), object(1)
memory usage: 24.4+ MB

## Word Clouds
Generate word clouds to visually represent the most frequent words in positive and negative tweets.
```python
# Separate positive and negative tweets
positive_tweets = new_df[new_df['target'] == 1]
negative_tweets = new_df[new_df['target'] == 0]

# Function to generate and plot word clouds
def generate_word_cloud(data, sentiment):
    all_text = ' '.join(data['stemmed_content'])
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    # Plot word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Tweets')
    plt.axis('off')
    plt.show()

# Generate word clouds for positive and negative tweets
generate_word_cloud(positive_tweets, 'Positive')
generate_word_cloud(negative_tweets, 'Negative')
```
![image](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/31104c09-d76e-49f4-ab00-147c5144ba06)
![image](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/754704f4-7f57-49d5-8ee6-8ebcdbb3902c)

## Word Frequency Analysis
Analyze the frequency of words in positive and negative tweets to understand their impact on sentiment.
```python
# Section: Word Frequency Analysis
# Code snippet for word frequency analysis

# Step 1: Create a list of all unique words
all_words = ' '.join(new_df['stemmed_content']).split()
unique_words = set(all_words)

# Step 2: Create an empty DataFrame to store word frequencies
word_freq_df = pd.DataFrame(columns=['unique_word', 'positive', 'negative'])

# Step 3: Count word frequencies for positive (target=1) and negative (target=0) labeled tweets
for word in unique_words:
    positive_count = sum((new_df['target'] == 1) & (new_df['stemmed_content'].str.contains(word)))
    negative_count = sum((new_df['target'] == 0) & (new_df['stemmed_content'].str.contains(word)))
    
    word_freq_df = pd.concat([word_freq_df, pd.DataFrame({'unique_word': [word], 'positive': [positive_count], 'negative': [negative_count]})])

# Optional: You can add a column for total frequency if needed
word_freq_df['total'] = word_freq_df['positive'] + word_freq_df['negative']

# Display the resulting DataFrame
print(word_freq_df.head())

# Sum of values in the 'positive' and 'negative' columns
sum_positive = word_freq_df['positive'].sum()
sum_negative = word_freq_df['negative'].sum()

# Count of total unique words
total_unique_words = len(word_freq_df)

# Append a new row with sums and count to the last row of the DataFrame
word_freq_df.loc[len(word_freq_df)] = {'unique_word': 'Total', 'positive': sum_positive, 'negative': sum_negative, 'total': total_unique_words}


# Display the results
print(f"Sum of positive values: {sum_positive}")
print(f"Sum of negative values: {sum_negative}")
print(f"Count of total unique words: {total_unique_words}")
```
Output - 
     unique_word positive negative total
0  troyeatsbrain        0        1     1
0          worri      168      156   324
0            ozd        3        6     9
0         imagen        1        0     1
0        phadden        1        0     1

Sum of positive values: 3135912
Sum of negative values: 2986661
Count of total unique words: 71283
## Naive Bayes Model

Implement the Naive Bayes model, including Laplacian smoothing and calculating conditional probabilities.

### Formulas Used:

#### Bayes' Theorem:

\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]

#### Conditional Probability:

\[ P(A|B) = \frac{P(A \cap B)}{P(B)} \]

#### Laplacian Smoothing:

\[ P(w|c) = \frac{count(w, c) + \alpha}{count(c) + \alpha \cdot V} \]

#### Log Likelihood:

\[ \text{log likelihood} = \log\left(\frac{P(w|pos)}{P(w|neg)}\right) \]

#### Log Prior:

\[ \text{log prior} = \log\left(\frac{\text{num\_positive\_tweets}}{\text{num\_negative\_tweets}}\right) \]
```python
# Section: Naive Bayes Model
# Code snippet for implementing Naive Bayes model

```
