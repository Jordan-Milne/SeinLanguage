from gazpacho import get, Soup
import re
import pandas as pd
from itertools import groupby

url = 'https://seinfeldscripts.com/seinfeld-scripts.html'
html = get(url)
soup = Soup(html)
table = soup.find('table')[1]
refs = table.find('a')
links = [i.attrs['href'] for i in refs]
links = [i.replace(" ", "") for i in links]
links


def scrape_script(episode):
    url = 'https://seinfeldscripts.com/' + str(episode)
    html = get(url)
    soup = Soup(html)
    table = soup.find('div', {'id' : 'content'})
    script = table.find('p')
    scrip = [i.remove_tags() for i in script]
    lines = same_line(scrip)
    scri = [i.replace('\n','') for i in lines]
    spaces = [re.sub(' +', ' ', i) for i in scri]
    lines = same_line(spaces)
    bracks = [re.sub('\[.*?\]', '', i) for i in lines]
    return bracks

scripts = [scrape_script(i) for i in links]
pd.DataFrame(scripts, columns = ['script'])
