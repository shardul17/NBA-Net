from urllib import request
from urllib.error import HTTPError
from bs4 import BeautifulSoup

#http://www.sports-reference.com/cbb/players/karl-anthony-towns-1.html

players = ['Carmelo Anthony', 'James Harden', 'Chris Bosh', 'Karl-Anthony Towns', 'Andrew Wiggins', 'Kevin Durant','Kevin Garnett']
for play in players:
    name = play.split()
    first = name[0].lower()
    last = name[1].lower()
    url = 'http://www.sports-reference.com/cbb/players/'+ first + '-' + last +'-' + '1.html'
    str = ''
    try:
        html = request.urlopen(url=url)
        soup = BeautifulSoup(html, "html.parser")
        statsrow = soup.find('tr',{"class": "thead"})
        text = []
        for i in statsrow:
            text.append(i.text)
        measurements = soup.find('div',{'class': 'nothumb'})
        print(name)
        for stats in measurements:
            lol = stats.find('p')
            if(type(lol) == int or lol == None):
                pass
            else:
                str = lol.text
                split = str.split('()')
                print(split)

    except HTTPError:
        print("Didnt go to college")
