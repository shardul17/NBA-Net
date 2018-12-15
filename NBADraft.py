from bs4 import BeautifulSoup
from urllib import request
from urllib.error import HTTPError
import pandas as pd
import sqlite3

conn = sqlite3.connect('draft.db')
c = conn.cursor()
def getCollegeURL(first, last,draftyear):
    playerID = 1
    url = ''
    while True:
        try:
            url = 'http://www.sports-reference.com/cbb/players/' + first + '-' + last + '-' + str(playerID) + '.html'
            html = request.urlopen(url)
            playerID += 1
            #'row_summable sortable stats_table now_sortable'
            soup = BeautifulSoup(html,"html.parser")
            year = soup.find('div',attrs={'class':"table_outer_container"}).find('div',attrs={'class':'overthrow table_container'}).find('table',attrs={'id':'players_per_game'}).tbody.find_all('tr')[-1].th.find_all('a')[0].text.split('-')[1]
            if(draftyear[len(draftyear)-2: len(draftyear) ] == year):
                return url
                break
            else:
                pass
        except HTTPError:
            break

def getCollegeStats(url):
    url  = 'http://www.sports-reference.com/cbb/players/'+ first + '-' + last +'-' + '1.html'
    try:
        html = request.urlopen(url=url)
        soup = BeautifulSoup(html, "html.parser")
        statsrow = soup.find('tr',{"class": "thead"})
        text = []
        for i in statsrow:
            text.append(i.text)
        measurements = soup.find('div',{'class': 'nothumb'})

        for stats in measurements:
            lol = stats.find('p')
            if(type(lol) == int or lol == None):
                pass
            else:
                string = lol.text
                index = string.find('(')
                english = string[0: index]
                metric = string[index:len(string) - 1]
                index = english.find(',')
                if(index != -1):
                    height = english[0: index]
                    weight = english[index + 2: len(english) - 1]
                    text.append(height)
                    text.append(weight)
                else:
                    height = english[0: len(english) - 1]
                    text.append(height)

                return text
    except HTTPError:
        return []


year = '2014'

url = 'http://www.basketball-reference.com/draft/NBA_' + year + '.html'

html = request.urlopen(url)

soup = BeautifulSoup(html, "html.parser")

column_headers = [th.text for th in soup.findAll('tr', limit=2)[1].findAll('th')]
#print(column_headers)

data_rows = soup.findAll('tr')[2:]
#print(data_rows)

player_data = []  # create an empty list to hold all the data

for i in range(len(data_rows)):
    player_row = []

    for td in data_rows[i].findAll('td'):
        player_row.append(td.text)

    if(len(player_row) != 0):
        name = player_row[2]# their name
        ws = player_row[17]# their nba win score

        url = getCollegeURL(name, player_row[3],2010)
        college_data = getCollegeStats(name)
        if(len(college_data)!= 0):


            #college data: G	MP	FG	FGA	FG%	2P	2PA	2P%	3P	3PA	3P%	FT	FTA	FT%	TRB	AST	STL	BLK	TOV	PF	PTS HT WT
            #player_row: Pick	Team	Player	College	Yrs	G	MP	PTS	TRB	AST	FG%	3P%	FT%	MP	PTS	TRB	AST	WS	WS/48	BPM	VORP
            c.execute('CREATE TABLE IF NOT EXISTS draft(name TEXT, career TEXT, college TEXT, G REAL, mp REAL, fgm REAL, fga REAL, fgper REAL, twopm REAL, twopa REAL, twoper REAL, threem REAL, threea REAL, threeper REAL, ftm REAL, fta REAL, ftper REAL, reb REAL, ast REAL, stl REAL, blk REAL, tov REAL, pf REAL, pts REAL, ht TEXT, wt TEXT)')
#
#            conn.commit()
#            c.close()
#            conn.close()



        #    print(college_data)
             #print(name)
             #print(college_data[0])
            c.execute("INSERT INTO draft(name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, tov, pf, pts, ht, wt)  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",(name, college_data[0], college_data[1], college_data[3], college_data[4], college_data[5], college_data[6], college_data[7], college_data[8], college_data[9], college_data[10], college_data[11], college_data[12], college_data[13], college_data[14], college_data[15], college_data[16], college_data[17], college_data[18], college_data[19], college_data[20], college_data[21], college_data[22], college_data[23], college_data[24], college_data[25]))

            conn.commit()

             #      c.execute("INSERT INTO draft VALUES(name, 'Andrew', 'career', 'Kansas', 35, 32.8, 5.4, 12.1, .448, 4.2, 8.5, .493, 1.2, 3.6, .341, 5.0, 6.5, .775, 5.9, 1.5, 1.2, 1.0, 2.3, 2.7, 17.1, 6-8, 200)")

            # probably append we need to append college stats to the WS of the player_row, and just do stuff on dat
        else:
            pass

        player_data.append(player_row)
    else:
        pass



getCollegeStats()
