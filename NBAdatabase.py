import sqlite3

conn = sqlite3.connect('draft.db')
c = conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS draft(last TEXT, first TEXT, career TEXT, college TEXT, G REAL, mp REAL, fgm REAL, fga REAL, fgper REAL, twopm REAL, twopa REAL, twoper REAL, threem REAL, threea REAL, threeper REAL, ftm REAL, fta REAL, ftper REAL, reb REAL, ast REAL, stl REAL, blk REAL, tov REAL, pf REAL, pts REAL, ht REAL, wt REAL)')

def data_entry():
    c.execute("INSERT INTO draft VALUES('Wiggins', 'Andrew', 'career', 'Kansas', 35, 32.8, 5.4, 12.1, .448, 4.2, 8.5, .493, 1.2, 3.6, .341, 5.0, 6.5, .775, 5.9, 1.5, 1.2, 1.0, 2.3, 2.7, 17.1, 6-8, 200)")
    conn.commit()
    c.close()
    conn.close()


create_table()
data_entry()
