from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy import optimize
import sqlite3
from sklearn.preprocessing import StandardScaler
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics



iris = datasets.load_iris()
feature_columns = skflow.infer_real_valued_columns_from_input(iris.data)
classifier = skflow.LinearClassifier(n_classes=3, feature_columns=feature_columns)
classifier.fit(iris.data, iris.target, steps=200, batch_size=64)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)

print(iris.target.shape)
print("")
print(len(iris_predictions))
