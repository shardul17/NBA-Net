import numpy as np
from scipy import optimize
import sqlite3
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
        conn = sqlite3.connect('draftYEAR.db')
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

class DraftGetterTest:
    def __init__(self, players, features):
        self.rows = None
        self.features = None
        if(players > 45):
            raise Exception('Rows is greater than DataBase')
        else:
            self.rows = players

        if(features > 21):
            raise Exception('Dont have that many features')
        else:
            self.features = features

        self.data = None
        conn = sqlite3.connect('DraftNEW.db')
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
        self.c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, pts, ht, wt, ws FROM DraftNEW")
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







# Whole Class with additions:
#New complete class, with changes:
class Neural_Network(object):
    def __init__(self, Lambda=0):
        #Define Hyperparameters
        self.inputLayerSize = 15
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        #Regularization Parameter:
        self.Lambda = Lambda

    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1

        return dJdW1, dJdW2

    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)

        return cost, grad

    def train(self, trainX, trainY, testX, testY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        #Make empty list to store training costs:
        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res







if __name__ == '__main__':
    #features = int(input('How many features do you wanna examine?: '))
    """
            TRAIN DATA STUFFFF

    """
    features = 21
    draft = DraftGetter(players=358, features=features)
    vals, ws = draft.getDraftTable()

    vals_standard = StandardScaler().fit_transform(vals)

    cor_mat1 = np.corrcoef(np.array(vals_standard).T)

    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    #print(len(eig_vecs), len(eig_vecs[0]))
    pairs = [[eig_vals[x],eig_vecs[:,x]] for x in range(0, len(eig_vals))]
    pairs.sort(key=lambda i: i[0], reverse=True)

    MAX_EIGENVALS = 15

    matrix = None
    for i in range(0, MAX_EIGENVALS):
        if(i ==0):
            matrix = pairs[0][1].reshape(len(pairs[0][1]) , 1)
        else:
            matrix = np.hstack( (matrix, pairs[i][1].reshape(len(pairs[i][1]) , 1)) )

    trainX = vals_standard.dot(matrix)

    train1 = trainX[0: (len(trainX))/2]
    train2 = trainX[(len(trainX))/2: 357]




    conn = sqlite3.connect('draftYEAR.db')
    c = conn.cursor()
    c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, tov, pf, pts, ht, wt, ws FROM draftYEAR")
    data = c.fetchall()

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
    trainY = Y/np.amax(Y, axis=0)

    trainY1 = trainY[0: (len(trainY))/2]
    trainY2 = trainY[(len(trainY))/2: 357]

    print(trainY1)
    print("")
    print(trainY2)


    """
                TEST DATA STUFFFF


    """



    features = 21
    draft = DraftGetterTest(players=45, features=features)
    vals, ws = draft.getDraftTable()

    vals_standard = StandardScaler().fit_transform(vals)

    cor_mat1 = np.corrcoef(np.array(vals_standard).T)

    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    pairs = [[eig_vals[x],eig_vecs[:,x]] for x in range(0, len(eig_vals))]
    pairs.sort(key=lambda i: i[0], reverse=True)


    MAX_EIGENVALS = 15

    matrix = None
    for i in range(0, MAX_EIGENVALS):
        if(i ==0):
            matrix = pairs[0][1].reshape(len(pairs[0][1]) , 1)
        else:
            matrix = np.hstack( (matrix, pairs[i][1].reshape(len(pairs[i][1]) , 1)) )
    X = vals_standard.dot(matrix)
    print(X.shape)




    conn = sqlite3.connect('DraftNEW.db')
    c = conn.cursor()
    c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, tov, pf, pts, ht, wt, ws FROM DraftNEW")
    data = c.fetchall()

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

    tY = np.array(target)
    testY = tY/np.amax(tY, axis=0)



    """


                    NEURAL STUFFFFF


    """

    """
    #Training Data:
    trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
    trainY = np.array(([75], [82], [93], [70]), dtype=float)

    #Testing Data:
    testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
    testY = np.array(([70], [89], [85], [75]), dtype=float)


    #Normalize:
    trainX = trainX/np.amax(trainX, axis=0)
    trainY = trainY/100 #Max test score is 100

    #Normalize by max of training data:
    testX = testX/np.amax(trainX, axis=0)
    testY = testY/100 #Max test score is 100

    """
    NN = Neural_Network()
    T = trainer(NN)
    T.train(train1,trainY1,train2,trainY2)
    print(NN.forward(X))
    print(train1)
    i=0
    d =[]
    while i<45:
        name = data[i][0]
        yHat = NN.forward(X)
        ws = yHat[i]

        print(name + str(ws))
    #    d.append(name, ws)
        i = i + 1

    print(testY)
