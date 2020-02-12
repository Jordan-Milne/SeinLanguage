import pandas as pd
import nltk
from nltk import tokenize
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd

#  Read in Scripts
df = pd.read_csv('data/scripts.csv')

# List all unique episodes in the DataFrame
epi = list(df['SEID'].unique())

# Using the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Function that takes a character and an episode then toeknizes by sentence.
# Each sentence said by that character gets a sentiment analysis and the mean of all the sentences is returned
def sent(char, episode):
    mask1 = df['SEID'] == episode
    mask2 = df['Character']== char
    analyze = df[mask1 & mask2]['Dialogue'].tolist()
    sent = []
    for i in analyze:
        toke = tokenize.sent_tokenize(i)
        for x in toke:
            sent.append(sid.polarity_scores(x))
    return np.mean([i['compound'] for i in sent]).round(4)


# Function That returns the amount of lines a character has in an episode
def lines(char, episode):
    mask1 = df['SEID'] == episode
    mask2 = df['Character']== char
    return len(df[mask1 & mask2]['Character'])

def guest(char, episode):
    mask1 = df['SEID'] == episode
    mask2 = df['Character']== char
    x = len(df[mask1 & mask2]['Character'])
    return False if x ==0 else True

# Creating and filling out the DataFrame with sentient scores and amount of lines
sei = pd.DataFrame(epi, columns=['id'])

sei['jer_sent'] = [sent('JERRY', i) for i in epi]
sei['ela_sent'] = [sent('ELAINE', i) for i in epi]
sei['kra_sent'] = [sent('KRAMER', i) for i in epi]
sei['geo_sent'] = [sent('GEORGE', i) for i in epi]

sei['jer_lines'] = [lines('JERRY', i) for i in epi]
sei['ela_lines'] = [lines('ELAINE', i) for i in epi]
sei['kra_lines'] = [lines('KRAMER', i) for i in epi]
sei['geo_lines'] = [lines('GEORGE', i) for i in epi]


sei['frank'] = [guest('FRANK', i)  for i in epi]
sei['newman'] = [guest('NEWMAN', i)  for i in epi]
sei['peterman'] = [guest('PETERMAN', i)  for i in epi]
sei['puddy'] = [guest('PUDDY', i)  for i in epi]

sei = pd.read_csv('data/sei.csv')
sei


sei.to_csv('data/sei2.csv', index = False)
