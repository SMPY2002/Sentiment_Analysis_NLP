# Sentiment Analysis using Naive Bayes Method (Learning Purpose)

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
![Screenshot 2024-03-12 074846](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/af252421-b2d9-4392-bedf-91376aaf779f)


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


![Screenshot 2024-03-12 075018](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/eb6ebf27-e35c-4c52-8b98-fac36d4e9d5f)


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

![Screenshot 2024-03-12 075130](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/6249e18f-c1a9-4b0d-b8c1-1d71647d104b)

![Screenshot 2024-03-12 075309](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/17b28d8f-5afd-4669-84b8-0f425ad8693e)


## Naive Bayes Model

Implement the Naive Bayes model, including Laplacian smoothing and calculating conditional probabilities.

### Formulas Used:

#### Bayes' Theorem:

- P(A|B) = (P(B|A) * P(A)) / P(B)

#### Conditional Probability:

- P(A|B) = P(A ∩ B) / P(B)

#### Laplacian Smoothing:

- P(w|c) = (count(w, c) + α) / (count(c) + αV)

#### Log Likelihood:

- log likelihood = log(P(w|pos) / P(w|neg))

#### Log Prior:

- log prior = log(num_positive_tweets / num_negative_tweets)
### Final Probability Function is  -
![Screenshot 2024-03-12 075949](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/25ef0016-fd44-4705-8947-2d40546445ae)
![Screenshot 2024-03-12 080143](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/a1a6e83c-647e-433b-9558-6ac959671407)
![Screenshot 2024-03-12 080203](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/60494cbc-d6ef-4103-aabc-cd8a8f0b4266)

```python
# Section: Naive Bayes Model
# Code snippet for implementing Naive Bayes model

# Smoothing  (Laplacian smoothing)

alpha = 1

# Calculate conditional probabilities for each word
word_freq_df['P(w|pos)'] = (word_freq_df['positive'] + alpha) / (sum_positive +  total_unique_words)
word_freq_df['P(w|neg)'] = (word_freq_df['negative'] + alpha) / (sum_negative +  total_unique_words)

# Display the updated DataFrame with conditional probabilities
print(word_freq_df.head())

```
```python
# Apply Laplacian smoothing and compute lambda(log likelihood) score for each word

# Laplacian smoothing function
def laplacian_smoothing(positive_freq, negative_freq, total_unique_words):
    smoothing_factor = 1  # Laplacian smoothing factor
    
    # Calculate conditional probabilities using Laplacian smoothing
    prob_pos = (positive_freq + smoothing_factor) / (sum_positive + total_unique_words)
    prob_neg = (negative_freq + smoothing_factor) / (sum_negative + total_unique_words)
    
    # Calculate probability ratio
    ratio_w = prob_pos / prob_neg
    
    # Compute lambda score
    lambda_w = np.log(ratio_w)
    
    return lambda_w

# Apply Laplacian smoothing and compute lambda(log likelihood) score for each word
word_freq_df['lambda_score'] = word_freq_df.apply(
    lambda row: laplacian_smoothing(row['positive'], row['negative'], total_unique_words),
    axis=1
)

# Save the DataFrame to a CSV file
word_freq_df.to_csv('word_frequency.csv', index=False)

# Display the updated DataFrame with lambda scores
print(word_freq_df.head())
```
```python
# Log Prior Calculation - 
# Step 1: Count the number of positive and negative tweets
num_positive_tweets = sum(new_df['target'] == 1)
num_negative_tweets = sum(new_df['target'] == 0)

# Step 2: Calculate the ratio of positive to negative tweets
ratio_pos_neg = num_positive_tweets / num_negative_tweets

# Step 3: Calculate the log prior
log_prior = np.log(ratio_pos_neg)

# Step 4: Save the computed log prior value in a .txt file
with open('log_prior.txt', 'w') as file:
    file.write(str(log_prior))

# Display the results
print(f"Number of positive tweets: {num_positive_tweets}")
print(f"Number of negative tweets: {num_negative_tweets}")
print(f"Ratio of positive to negative tweets: {ratio_pos_neg}")
print(f"Log prior: {log_prior}")
```
Output -
![Screenshot 2024-03-12 075517](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/ab63e083-8ee5-4a88-95dd-68188e87cc58)

# Note :- As you see log prior is zero (it means that your dataset is balanced , if not balanced then it not become zero )

## Testing Phase
Test the Naive Bayes model on a user-provided tweet to predict its sentiment (positive or negative).
	1. Loading the Word_Frequency csv file which is saved during model training
 	2. Loading the log prior calculation which is saved as log_prior.txt file
Doing the same task for user tweet also, first pre-processed its entered tweet , then with the help of above loading files we calculate its sentiment.(See the code below)
Note :- It is always advisable to store the Pre-Processed tweets , its frequency count and other relevant things , so that when required then we do not run all the time taken functions (when dataset is very large, then it is more computationallt Expensive.)
```python
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the precomputed data and models
loaded_word_freq_df = pd.read_csv('word_frequency.csv')
# loaded_log_prior = log_prior
with open('log_prior.txt', 'r') as file:
        loaded_log_prior = float(file.read())

# Define the Stemming Function
def stemming(content):
    # Remove non-alphabetic characters using regular expression
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert to lowercase
    stemmed_content = stemmed_content.lower()
    
    # Tokenize the text into words
    stemmed_content = stemmed_content.split()
    
    # Apply stemming using Porter Stemmer, and exclude stopwords
    final_stopwords = set(stopwords.words('english')) - set(['not', 'no', 'against', 'nor'])
    port_stem = PorterStemmer()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in final_stopwords]
    
    # Join the stemmed words into a single string
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content

# Define the log likelihood calculation function
def calculate_log_likelihood(stemmed_input):
    # Split the input into words
    words = stemmed_input.split()

    # Initialize likelihoods
    log_likelihood_pos = 0
    log_likelihood_neg = 0

    # Iterate through words and calculate log likelihood
    for word in words:
        if word in loaded_word_freq_df['unique_word'].values:
            # Get the lambda score for the word
            lambda_w = loaded_word_freq_df.loc[loaded_word_freq_df['unique_word'] == word, 'lambda_score'].values[0]

            # Update log likelihoods
            log_likelihood_pos += lambda_w if lambda_w > 0 else 0
            log_likelihood_neg += -lambda_w if lambda_w < 0 else 0

    return log_likelihood_pos, log_likelihood_neg

# Define the prediction function
def predict_sentiment(log_likelihood_pos, log_likelihood_neg):
    # Calculate the final log likelihoods
    log_likelihood_pos += loaded_log_prior
    log_likelihood_neg += loaded_log_prior

    # Determine the predicted sentiment class
    predicted_class = 1 if log_likelihood_pos > log_likelihood_neg else 0
    return predicted_class

# Input a new tweet
new_tweet = input("Enter the tweet: ")

# Preprocess the input tweet
preprocessed_tweet = stemming(new_tweet)

# Calculate log likelihood
log_likelihood_pos, log_likelihood_neg = calculate_log_likelihood(preprocessed_tweet)

# Make predictions
predicted_sentiment = predict_sentiment(log_likelihood_pos, log_likelihood_neg)

print("Your Entered Tweet : -",new_tweet)

# Display the result
if predicted_sentiment == 1:
    print("Positive sentiment!")
else:
    print("Negative sentiment!")

```
Output - 
![Screenshot 2024-03-12 075824](https://github.com/SMPY2002/Sentiment_Analysis_NLP/assets/118500436/ab99c8a6-8ccb-4f6b-87e4-812eadaa7078)

## Accuracy on Unseen Tweets
Evaluate the accuracy of the model on unseen tweets from the remaining dataset.
```python
# Create a DataFrame with the remaining tweets for testing
remaining_tweets = df[~df.index.isin(new_df.index)].head(5000)

# Create a new column for predicted labels in the remaining_tweets DataFrame
remaining_tweets['predicted_target'] = remaining_tweets['text'].apply(lambda tweet: predict_sentiment(*calculate_log_likelihood(preprocess_input_tweet(tweet), word_freq_df), log_prior=log_prior))

# Calculate accuracy
accuracy = sum(remaining_tweets['target'] == remaining_tweets['predicted_target']) / len(remaining_tweets)

# Display the accuracy 
print(f"Accuracy on remaining tweets: {accuracy * 100:.2f}%")
```
# Output - Accuracy on unseen tweets is : 80.78%

## Conclusion
Congratulations! You've successfully implemented a Sentiment Analysis model using the Naive Bayes method. Feel free to explore further and customize the code to enhance the model's performance. Thank you for reading, and happy coding!                                                                                                          
