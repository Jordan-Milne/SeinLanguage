import json
import glob
import os
import re

# Loading all the JSON files from seperate folders into one file
result = []
files = os.listdir('seinfeld/')


for i in sorted(glob.glob('seinfeld/*.txt')):
    f = open(i)
    result.append(f)

x = {}

def real_script(episode):

    def same_line(link):
        list2 = []
        for i in link:
            if re.match("^[a-zA-Z]*\:", i):
                list2.append(i)
            else:
                list2[-1] += i
        return list2


    scrip = result[125].readlines()
    lines = same_line(scrip)
    scri = [i.replace('\n','') for i in lines]
    spirc = re.sub("[\(\[].*?[\)\]]", "", scri)
    spaces = [re.sub(' +', ' ', i) for i in spirc]
    lines = same_line(spaces)
    # bracks = [re.sub('\[.*?\]', '', i) for i in lines]
    return lines



def same_line(link):
    list2 = []
    for i in link:
        if re.match("^[a-zA-Z]*\:", i):
            list2.append(i)
        else:
            list2[-1] += f' {i}'
    return list2


scrip = result[0].readlines()
percent = [ x for x in scrip if "%" not in x ]
scri = [i.replace('\n','') for i in percent]
lines = same_line(scri)
bracks = [re.sub('\[.*?\]', '', i) for i in lines]
brac = [re.sub('\(.*?\)', '', i) for i in bracks]
spaces = [re.sub(' +', ' ', i) for i in brac]
spaces












real_script(result[1])

fh.readlines()
