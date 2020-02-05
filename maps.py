
from gazpacho import get, Soup
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd

# Storing the link for every episode map guide in 'epis'
url = 'https://mapsaboutnothing.com/episodes/'
html = get(url)
soup = Soup(html)
table = soup.find('div', {'class' : 'entry-content'})
links = table.find('a')
links = links[:-9]
links[1].attrs['href']
epis = [i.attrs['href'] for i in links]

# Function that takes episode link as input and returns all the places in the episode
def get_places(episode):

    url = episode
    html = get(url)
    soup = Soup(html)
    table = soup.find('div', {'class' : 'entry-content'})
    links = table.find('a')
    place = [i.text for i in links][:-6]
    places = []
    for i in place:
        if 'map' not in i:
            places.append(i)
    return places

real = [get_places(i) for i in epis]

locations = pd.DataFrame({ 'locations': real})
locations.to_csv('locations.csv', index= False)


flat_list = []
for sublist in real:
    for item in sublist:
        flat_list.append(item)

word_count = Counter(flat_list)

top_locs = [i for i in word_count if word_count[i] >3]

# save as JSON
