import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

class DraftGetter:
    def __init__(self, players, features):
        self.rows = None
        self.features = None
        if(players > 358):
            raise Exception('Rows is greater than DataBase')
        else:
            self.rows = players

        if(features > 21):
            raise Exception('Dont have that many features')
        else:
            self.features = features

        self.data = None
        conn = sqlite3.connect('newdraft.db')
        self.c = conn.cursor()

    def getDraftTable(self):
        data = self.read_from_db()
        self.data = [self.getCorrectArray(data[x]) for x in range(0, self.rows)]
        return self.data, [ data[x][len(data[x]) -1 ] for x in range(0, self.rows)]

    def getCorrectArray(self,x):
        t = []
        for i in range(3, 3 + self.features):
            if(i == 22):
                t.append(float(self.getHeight(x[i])))
            elif(i == 23):
                t.append(float(self.getWeight(x[i])))
            else:
                if(self.isfloat(x[i])):
                    t.append(float(x[i]))
                else:
                    t.append(0)
        return t

    def read_from_db(self):
        self.c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, pts, ht, wt, ws FROM draftYear")
        data = self.c.fetchall()
        return data

    def getWeight(self,string):
        weight = string.split('l')
        if(len(weight) >= 0):
            return float(weight[0])
        else:
            return 0
    def getHeight(self,string):
        height = string.split('-')
        if(len(height) == 2):
            return float(height[0])*12 + float(height[1])
        else:
            return 0.0
    def isfloat(self,value):
      try:
        float(value)
        return True
      except:
        return False

#outputvals = '1: G, 2:mp, 3:fgm, 4:fga, 5:fgper, 6:twopm, 7:twopa, 8:twoper, 9:threem, 10:threea, 11:threeper, 12:ftm, 13:fta, 14:ftper, 15:reb, 16:ast, 17:stl, 18:blk, 19:pts, 20:ht, 21:wt, 22:ws'
def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False

if __name__ == '__main__':
    #features = int(input('How many features do you wanna examine?: '))
    features = 21
    draft = DraftGetter(players=358, features=features)
    vals, ws = draft.getDraftTable()

    vals_standard = StandardScaler().fit_transform(vals)

    cor_mat1 = np.corrcoef(np.array(vals_standard).T)

    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    print(len(eig_vecs), len(eig_vecs[0]))
    pairs = [[eig_vals[x],eig_vecs[:,x]] for x in range(0, len(eig_vals))]
    pairs.sort(key=lambda i: i[0], reverse=True)

    MAX_EIGENVALS = 2

    matrix = None
    for i in range(0, MAX_EIGENVALS):
        if(i ==0):
            matrix = pairs[0][1].reshape(len(pairs[0][1]) , 1)
        else:
            matrix = np.hstack( (matrix, pairs[i][1].reshape(len(pairs[i][1]) , 1)) )
    Y = vals_standard.dot(matrix)
    print(Y)
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier( hidden_layer_sizes= (10,10,10) ,solver='sgd',learning_rate_init=0.01,max_iter=1000)
    mlp.fit(Y, StandardScaler().fit_transform(ws))
