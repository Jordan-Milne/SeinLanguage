import pandas as pd
import nltk
from nltk import tokenize
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd


df = pd.read_csv('scripts.csv')
epi = list(df['SEID'].unique())
df

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




nltk.download('vader_lexicon')
nltk.download('punkt')
sid = SentimentIntensityAnalyzer()


epi = list(df['SEID'].unique())
sei = pd.DataFrame(epi, columns=['id'])


sei['jer_sent'] = [sent('JERRY', i) for i in epi]
sei['ela_sent'] = [sent('ELAINE', i) for i in epi]
sei['kra_sent'] = [sent('KRAMER', i) for i in epi]
sei['geo_sent'] = [sent('GEORGE', i) for i in epi]



def lines(char, episode):
    mask1 = df['SEID'] == episode
    mask2 = df['Character']== char
    return len(df[mask1 & mask2]['Character'])

sei['jer_lines'] = [lines('JERRY', i) for i in epi]
sei['ela_lines'] = [lines('ELAINE', i) for i in epi]
sei['kra_lines'] = [lines('KRAMER', i) for i in epi]
sei['geo_lines'] = [lines('GEORGE', i) for i in epi]

sei.to_csv('sei.csv', index = False)
