from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy import optimize
import sqlite3
from sklearn.preprocessing import StandardScaler
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

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
        if(players > 35):
            raise Exception('Rows is greater than DataBase')
        else:
            self.rows = players

        if(features > 21):
            raise Exception('Dont have that many features')
        else:
            self.features = features

        self.data = None
        conn = sqlite3.connect('draftTEST.db')
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
        self.c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, pts, ht, wt, ws FROM draftTEST")
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
        self.inputLayerSize = 2
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

    MAX_EIGENVALS = 2

    matrix = None
    for i in range(0, MAX_EIGENVALS):
        if(i ==0):
            matrix = pairs[0][1].reshape(len(pairs[0][1]) , 1)
        else:
            matrix = np.hstack( (matrix, pairs[i][1].reshape(len(pairs[i][1]) , 1)) )

    trainX = vals_standard.dot(matrix)

    train1 = trainX[0: (len(trainX))/2]
    train2 = trainX[(len(trainX))/2: 357]

    print(train1)
    print("")

    print(train2)





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


    """
                TEST DATA STUFFFF


    """



    features = 21
    draftTest = DraftGetterTest(players=35, features=features)
    valsTest, ws = draftTest.getDraftTable()

    valsTest_standard = StandardScaler().fit_transform(valsTest)

    cor_matTest = np.corrcoef(np.array(valsTest_standard).T)

    eig_valsTest, eig_vecsTest = np.linalg.eig(cor_matTest)
    #print(len(eig_vecs), len(eig_vecs[0]))
    pairsTest = [[eig_valsTest[x],eig_vecsTest[:,x]] for x in range(0, len(eig_valsTest))]
    pairsTest.sort(key=lambda i: i[0], reverse=True)

    MAX_EIGENVALS = 2

    matrixTest = None
    for i in range(0, MAX_EIGENVALS):
        if(i ==0):
            matrixTest = pairsTest[0][1].reshape(len(pairsTest[0][1]) , 1)
        else:
            matrixTest = np.hstack( (matrixTest, pairsTest[i][1].reshape(len(pairsTest[i][1]) , 1)) )
    testX = vals_standard.dot(matrixTest)

    testXHALF = testX[0: (len(testX))/2]

    #print(testXHALF.shape)

    connTEST = sqlite3.connect('draftTEST.db')
    cTEST = connTEST.cursor()
    cTEST.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, tov, pf, pts, ht, wt, ws FROM draftTEST")
    dataTEST = cTEST.fetchall()

    targetTEST = []
    testTEST = []

    for row in dataTEST:
        wt = row[25][0:2]
        weight = int(wt)
        ht = row[24].split('-')
        ht = int(ht[0])*12 + int(ht[1])
        height = int(ht)
        fromdb=[row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[23],row[24],wt]
        testTEST.append(fromdb)
        ws = [row[26]]
        targetTEST.append(ws)

    tY = np.array(targetTEST)
    testY = tY/np.amax(tY, axis=0)

    """"
    X = [[0., 0.], [1., 1.]]
    y = [[0, 1], [1, 1]]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

    clf.fit(X, y)

    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

    print(clf.predict([[1., 2.]]))
    print(clf.predict([[0., 0.]]))

    iris = datasets.load_iris()
    feature_columns = skflow.infer_real_valued_columns_from_input(trainX)
    classifier = skflow.LinearRegressor(feature_columns=feature_columns)
    classifier.fit(trainX, trainY, steps=200, batch_size=64)
    predictions = list(classifier.predict(testX, as_iterable=True))
    score = metrics.accuracy_score(trainY, predictions)

    print(testY.shape)
    print("")
    print(len(predictions))
    """
