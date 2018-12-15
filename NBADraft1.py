from bs4 import BeautifulSoup
from urllib import request
from urllib.error import HTTPError
import pandas as pd
import sqlite3

conn = sqlite3.connect('draftNEW.db')
c = conn.cursor()
#def create database(player_row, college_stats):

def urlChecker(first, last,draftyear):
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

def getCollegeStats(collegeurl):
    url = collegeurl
    string = ''
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
        yrsPro = player_row[4]
        ws = ws/yrsPro

        name = name.split()
        first = name[0].lower()
        last = name[1].lower()

        collegeurl = urlChecker(first,last,year)
        #collegestats: Season	School	Conf	G	MP	FG	FGA	FG%	2P	2PA	2P%	3P	3PA	3P%	FT	FTA	FT%	ORB	DRB	TRB	AST	STL	BLK	TOV	PF	PTS HEIGHT WEIGHT
        #player_row: Pk	Tm	Player	College	Yrs	G	MP	PTS	TRB	AST	FG%	3P%	FT%	MP	PTS	TRB	AST	WS	WS/48	BPM	VORP


                    #college data: G	MP	FG	FGA	FG%	2P	2PA	2P%	3P	3PA	3P%	FT	FTA	FT%	TRB	AST	STL	BLK	TOV	PF	PTS HT WT
                    #player_row: Pick	Team	Player	College	Yrs	G	MP	PTS	TRB	AST	FG%	3P%	FT%	MP	PTS	TRB	AST	WS	WS/48	BPM	VORP


                     #      c.execute("INSERT INTO draft VALUES(name, 'Andrew', 'career', 'Kansas', 35, 32.8, 5.4, 12.1, .448, 4.2, 8.5, .493, 1.2, 3.6, .341, 5.0, 6.5, .775, 5.9, 1.5, 1.2, 1.0, 2.3, 2.7, 17.1, 6-8, 200)")

                    # probably append we need to append college stats to the WS of the player_row, and just do stuff on dat

        if(collegeurl!= None):
            college_stats = getCollegeStats(collegeurl)
            college_stats.insert(0,first + ' ' + last)
            college_stats.append([ws])
            print(college_stats)
            print(player_row[17]/player_row[4])
            print(college_stats[0])
            c.execute('CREATE TABLE IF NOT EXISTS draftNEW(name TEXT, career TEXT, college TEXT, G REAL, mp REAL, fgm REAL, fga REAL, fgper REAL, twopm REAL, twopa REAL, twoper REAL, threem REAL, threea REAL, threeper REAL, ftm REAL, fta REAL, ftper REAL, reb REAL, ast REAL, stl REAL, blk REAL, tov REAL, pf REAL, pts REAL, ht TEXT, wt TEXT, ws REAL)')
            c.execute("INSERT INTO draftNEW(name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, tov, pf, pts, ht, wt, ws)  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",(college_stats[0], college_stats[1], college_stats[2], college_stats[4], college_stats[5], college_stats[6], college_stats[7], college_stats[8], college_stats[9], college_stats[10], college_stats[11], college_stats[12], college_stats[13], college_stats[14], college_stats[15], college_stats[16], college_stats[17], college_stats[18], college_stats[19], college_stats[20], college_stats[21], college_stats[22], college_stats[23], college_stats[24], college_stats[25], college_stats[26], (player_row[17]/player_row[4]))



        #player_data.append(player_row)
    else:
        pass

#
#            conn.commit()
#            c.close()
#            conn.close()



        #    print(college_data)
             #print(name)
             #print(college_data[0])
