import pandas as pd

df = pd.read_csv('data/sei2.csv')
df.drop([78], inplace = True)
df.reset_index(drop=True, inplace=True)
df


loc = pd.read_csv('data/loc_good.csv')
loc.drop([78], inplace = True)
loc.reset_index(drop=True, inplace=True)
loc

rate = pd.read_csv('data/rat.csv')
rate

df['location'] = loc['locations']
df['rating'] = rate['rating']
df


info = pd.read_csv('data/episode_info.csv')
info.drop([78], inplace = True)
info.drop([0], inplace = True)
info.reset_index(drop=True, inplace=True)
info

df = df.merge(info, left_on='id', right_on='SEID')
df.drop(axis=1, columns=['Season','EpisodeNo','SEID'], inplace = True)


df


df.to_csv('data/final3.csv', index=False)
