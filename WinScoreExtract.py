from urllib import request
from bs4 import BeautifulSoup

players = ['Carmelo Anthony', 'Lebron James', 'James Harden', 'Chris Bosh' ]
index = 5
for play in players:
    name = play.split()
    first = name[0].lower()
    last = name[1].lower()
    if(len(last) <= 5):
        index = len(last)
    else: index = 5
    url = 'http://www.basketball-reference.com/players/'+ last[0] + '/'+ last[0:index] + first[0:1+1] + '01.html'
    html = request.urlopen(url=url)

    soup = BeautifulSoup(html, "html.parser")

    mydivs = soup.findAll("div", {"class": "p3"})

    text = None
    for i in mydivs:
        text = (i.text)

    ip = text.splitlines()
    ip = ip[7:len(ip) - 1]
    print(ip)