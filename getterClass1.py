import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


class DraftGetter:
    def __init__(self, players, features):
        self.rows = None
        self.features = None
        if(players > 317):
            raise Exception('Rows is greater than DataBase')
        else:
            self.rows = players

        if(features > 21):
            raise Exception('Dont have that many features')
        else:
            self.features = features

        self.data = None
        conn = sqlite3.connect('draft.db')
        self.c = conn.cursor()

    def getDraftTable(self):
        data = self.read_from_db()
        self.data = [self.getCorrectArray(data[x]) for x in range(0, self.rows)]
        return self.data, [x[len(x) - 1] for x in data]

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
                    t.append(x[i])
        return t

    def read_from_db(self):
        self.c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, pts, ht, wt, ws FROM draft")
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

class Neural_Network(object):
    def __init__(self):
        #define hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, \
                                  self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, \
                                  self.outputLayerSize)
    def forward(self, X):
        #propogate inputs thru network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, slef.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

if __name__ == '__main__':
    #features = int(input('How many features do you wanna examine?: '))
    features = 21
    draft = DraftGetter(players=100, features=features)
    vals, ws = draft.getDraftTable()

    vals_standard = StandardScaler().fit_transform(vals)

    cor_mat1 = np.corrcoef(np.array(vals_standard).T)

    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    pairs = [[eig_vals[x],eig_vecs[:,x]] for x in range(0, len(eig_vals))]
    pairs.sort(key=lambda i: i[0], reverse=True)

    plt.plot(eig_vals)
    plt.title('Eigenval Plot')
    plt.xlabel('Vals')
    plt.ylabel('EigenVal')
    plt.grid()
    plt.show()

    MAX_EIGENVALS = 10

    matrix = None
    for i in range(0, MAX_EIGENVALS):
        if(i ==0):
            matrix = pairs[0][1].reshape(len(pairs[0][1]) , 1)
        else:
            matrix = np.hstack( (matrix, pairs[i][1].reshape(len(pairs[i][1]) , 1)) )

    X = vals_standard.dot(matrix)



conn = sqlite3.connect('draft.db')
c = conn.cursor()
def read_from_db():
    c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, tov, pf, pts, ht, wt, ws FROM draft")
    data = c.fetchall()
    return data


data = read_from_db()
target = []
test = []

for row in data:
    wt = row[25][0:2]
    weight = int(wt)
    ht = row[24].split('-')
    ht = int(ht[0])*12 + int(ht[1])
    height = int(ht)
    fromdb=[row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[23],row[24],wt]
    test.append(fromdb)
    ws = [row[26]]
    target.append(ws)
Y = np.array(target)


print(X)
print("")
print("")
print("")
print("")
print("")
print(Y)
