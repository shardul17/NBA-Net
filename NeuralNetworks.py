import numpy as np
from scipy import optimize
import sqlite3
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from getterClass3.py import DraftGetterTest


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
        conn = sqlite3.connect('draftYEAR.db')
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
        self.c.execute("SELECT name, career, college, G, mp, fgm, fga, fgper, twopm, twopa, twoper, threem, threea, threeper, ftm, fta, ftper, reb, ast, stl, blk, pts, ht, wt, ws FROM draftYEAR")
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
    draft = DraftGetter(players=358, features=features)
    vals, ws = draft.getDraftTable()

    vals_standard = StandardScaler().fit_transform(vals)
    print(vals_standard.shape)
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
    Y = StandardScaler().fit_transform(ws)


class Neural(object):

    def __init__(self):
        self.inputLayerSize = 15
        self.outputLayerSize = 13
        self.hiddenLayerSize = 3
        self.W1 = np.random.rand(self.inputLayerSize,self.hiddenLayerSize)

        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)


    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = Neural.sigmoid(self.z2)
        self.z3 = np.dot(self.z2, self.W2)
        self.yHat = Neural.sigmoid(self.z3)

        return self.yHat

    def getVars(self):
        return self.yHat

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def sigmoidPrime(x):
        return np.exp(-1*x)/(1+np.exp(-1*x))**2

    def costFunctionPrime(self,x,y):

        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), Neural.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * Neural.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def costFunction(self,x,y):
        self.yHat = self.forward(x)
        j = 0.5* sum((y - self.yHat)**2)
        return j

    def f(x):
        return x**2

    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def computeNumericalGradient(self, N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2 * e)

            # Return the value we changed to zero:
            perturb[p] = 0

        # Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)

        return cost, grad

    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 100, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


if __name__ == '__main__':

    X = X
    y = Y

    X = X / np.amax(X, axis=0)

    y = y / 100

    X1 = np.array(([3,5], [5,1], [10,2]), dtype=float)
    y1 = np.array(([75], [82], [93]), dtype=float)
    X1 = X/np.amax(X, axis=0)
    y1 = y/100 #Max test score is 100

    NN = Neural()
    T = trainer(NN)
    T.train(X1,y1)
    plt.plot(T.J)

    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
