from gazpacho import get, Soup
from bs4 import BeautifulSoup

url = 'https://www.imdb.com/list/ls050927293/?sort=release_date,asc&st_dt=&mode=detail&page=1'
html = get(url)
soup = Soup(html)
div = soup.find('div', {'class' : 'ipl-rating-star small'}, strict=True)
ratings1 = [i.find('span', {'class' : 'ipl-rating-star__rating'}).text for i in div]

url = 'https://www.imdb.com/list/ls050927293/?sort=release_date,asc&st_dt=&mode=detail&page=2'
html = get(url)
soup = Soup(html)
div = soup.find('div', {'class' : 'ipl-rating-star small'}, strict=True)
ratings2 = [i.find('span', {'class' : 'ipl-rating-star__rating'}).text for i in div]
ratings = ratings1 + ratings2
len(ratings)
import pandas as pd
rating = pd.DataFrame({'rating' : ratings})
rating.to_csv('ratings.csv', index = False)
