from bs4 import BeautifulSoup
from urllib import request
import requests
from urllib.error import HTTPError
import pandas as pd
import sqlite3
import json

conn = sqlite3.connect('draft.db')
c = conn.cursor()
def getCollegeURL(name, college):
    name = name.split()
    first = name[0].lower()
    last = name[1].lower()
    query = first + '%20' + last + '%20' + college + '%20sports%20reference'
    response = requests.get("http://www.google.com/search",
              params={'q': query, 'limit':1})
    #print(response.text)
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all('a'):
        print(link.get('href'))
    #statsrow = soup.findAll('a', {"href":first})
    #urlList = []
    return statsrow
"""
    for i in json.loads(response.json())['d']['results']:
        print(i['Title'])
"""
    #html = request.urlopen(url=url)
    #return results
"""
        soup = BeautifulSoup(html, "html.parser")
        statsrow = soup.findAll('h3',{"class": "r"})
        urlList = []
        return statsrow
    except HTTPError:
        return 'boi'
"""


print(getCollegeURL('Anthony Davis', 'University of Kentucky'))
