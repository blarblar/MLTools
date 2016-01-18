import pandas as pd
import random
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import ensemble
from sklearn import preprocessing

def mockData():
    rand = []
    classPartOne = []
    classPartTwo = []
    c = []
    
    for i in range(0, 1000):
        rand.append(random.randint(0, 10))
        classPartOne.append(random.randint(-10, 10))
        classPartTwo.append(random.randint(0, 100))
        c.append(0 if classPartOne[-1] < 0 and classPartTwo[-1] % 2 == 0 else 1)
    df = pd.DataFrame({ "Random" : rand, "Col1" : classPartOne, "Col2" : classPartTwo, "Dep" : c})
    return df

def doLearn(dataframe, column="Dep"):
    print(doSVM(dataframe, column))
    print(doRandomForest(dataframe, column))
    print(doSVMNormal(dataframe, column))
    
def doRandomForest(dataframe, column):
    acc = 0
    bestNum = 0
    for i in range(1, 3):
        clf = ensemble.RandomForestClassifier(n_estimators=10**i)
        clf.fit(dataframe.drop(column, 1), dataframe[column])
        score = cross_validation.cross_val_score(clf, dataframe.drop(column, 1), dataframe[column], cv=10).mean()
        if(score > acc):
            acc = score
            bestNum = i
    return "Random forest accuracy: " + str(acc) + " Num Trees: " + str(10**bestNum)
    
def doSVM(dataframe, column):
    acc = 0
    bestC = 0
    bestGamma = 0
    for c in range(-5, 15, 2):
        for gamma in range(-5, 3, 2):
            clf = SVC(gamma=2**gamma, C=2**c, kernel='rbf')
            clf.fit(dataframe.drop(column, 1), dataframe[column])
            score = cross_validation.cross_val_score(clf, dataframe.drop(column, 1), dataframe[column], cv=10).mean()
            if(score > acc):
                acc = score
                bestC = c
                bestGamma= gamma
    return "SVM accuracy: " + str(acc) + " C: " + str(2**bestC) + " Gamma: " + str(2**bestGamma)

def doSVMNormal(dataframe, column):
    acc = 0
    bestC = 0
    bestGamma = 0
    preprocessing.normalize(dataframe.drop(column, 1), copy=False)
    for c in range(-5, 15, 2):
        for gamma in range(-5, 3, 2):
            clf = SVC(gamma=2**gamma, C=2**c, kernel='rbf')
            clf.fit(dataframe.drop(column, 1), dataframe[column])
            score = cross_validation.cross_val_score(clf, dataframe.drop(column, 1), dataframe[column], cv=10).mean()
            if(score > acc):
                acc = score
                bestC = c
                bestGamma= gamma
    return "SVM accuracy: " + str(acc) + " C: " + str(2**bestC) + " Gamma: " + str(2**bestGamma)
