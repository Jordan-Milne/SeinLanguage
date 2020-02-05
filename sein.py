import pandas as pd

df = pd.read_csv('sei.csv')
df.drop([78], inplace = True)

loc = pd.read_csv('location.csv')
loc.drop([78], inplace = True)
rate = pd.read_csv('ratings.csv')

df['location'] = loc['locations']
df['rating'] = rate['rating']
