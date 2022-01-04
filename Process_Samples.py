import json
import os
import math
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
from datetime import datetime
from requests import api
from apiclient.discovery import build
import numpy as np
from tqdm import tqdm
import requests
import json
import re
import time
import requests
import ast
import codecs


j = open("sample_videos.json", "r",encoding="utf-8")
j = j.read()
j = codecs.decode(j, "unicode-escape")
w = json.loads(j)
w = [x["_source"] for x in w]
# print(w[0])
seen = dict()






a = []
for elem in w:
    if elem['id'] in seen:
        continue
    else:
        seen[elem['id']] = True
        a.append(elem)
# print(len(w))
# print(len(a))
unique_games = set([x["game_name"] for x in a])
# print(len(unique_vals))

# This list was made with the help of Kibana
top_15 = ["N/A",
"Minecraft",
"Fortnite",
"Grand Theft Auto V",
"Roblox",
"Among Us",
"Happy Wheels",
"Five Nights at Freddys",
"PUBG MOBILE",
"Garrys Mod",
"Tom Clancys Rainbow Six Siege",
"Call of Duty: Black Ops III",
"Subnautica",
"Call of Duty: Advanced Warfare",
"Hello Neighbor"]

def one_hot_games(game, unique_games):
    # automatically adds an "other" category to mark for the non-top 15 in our case
    a = [0]*(len(unique_games) + 1)
    try:
        a[unique_games.index(game)] = 1
    except Exception as e:
        # if it is not in any, simply add a 1 to the "other" category
        a[-1] = 1
    return a
            

# add .append to array
# game_one_hots = [[one_hot_games(x["game_name"], top_15),x["game_name"] ] for x in a]
# print(game_one_hots)
l_div_d = [math.log(x["l/d"]) if x["l/d"] >0 else 0.1  for x in a]
max_l_div_d = l_div_d[np.argmax(l_div_d)]
print("Max l_div_d is ","https://youtube.com/watch?v=" +str(a[np.argmax(l_div_d)]["id"]))
print("Belongs to ", a[np.argmax(l_div_d)]["id"])
v_div_s = [math.log(x["v/s"]) if x["l/d"] > 0 else 0.1 for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
# print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))

v_div_s = [math.log(x["v/s"]) for x in a]
max_v_div_s = v_div_s[np.argmax(v_div_s)]
# print("Max v_dix_s is ", max_v_div_s)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(v_div_s)]["id"]))


subscribers = [math.log(max(0.1, x["views"]/x["v/s"])) for x in a]
max_subs = subscribers[np.argmax(subscribers)]
print("Max subscribers is ", max_subs)
print("Belongs to ", "https://youtube.com/watch?v=" + str(a[np.argmax(subscribers)]["id"]))



# np.argmax(])
inputs = [([math.log(max(0.1, x["l/d"]))/max_l_div_d,math.log(max(0,x["v/s"]))/max_v_div_s,math.log(max(0.1,x["views"]/x["v/s"]))/max_subs ] +one_hot_games(x["game_name"], top_15)) for x in a]
print(inputs[0])
inputs = np.array(inputs,dtype="float")
# print(inputs[0].shape)

outputs = [[min(4,max(0, math.log( x["views"], 10)-3))] for x in a]  
outputs = np.array(outputs,dtype="float")
# print(outputs.shape)




# load some more libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(linewidth=180)
class FeedforwardNeuralNetworkSGD:
    
    def __init__(self, layers, alpha = 0.1, batchSize = 32):
        self.W = []
        
        self.layers = layers
        
        self.alpha = alpha
        # Change in Activation function!
        # C = 2.5
        self.c = 2.5
        derivatice_elu_single = lambda x: (self.c * np.exp(x)) if(x <=0) else 1
        self.vectorized_ELU_Derivatice= np.vectorize(derivatice_elu_single)
        elu_single = lambda x: (self.c * np.exp(x) - 1) if(x <=0) else x
        self.vectorized_ELU = np.vectorize(elu_single)

        
        
        self.batchSize = batchSize
        
        
        for i in np.arange(0, len(layers) - 2):
            self.W.append(np.random.randn(layers[i] + 1, layers[i + 1] + 1)/100.0)
            
        self.W.append(np.random.randn(layers[-2] + 1, layers[-1])/100.0)
        
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoidDerivative(self, z):
        return z * (1 - z)
    
    def getNextBatch(self, X, y, batchSize):
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])
    
    def fit(self, X, y, epochs = 10000, update = 1000):
        X = np.hstack((X, np.ones([X.shape[0],1])))

        for epoch in tqdm(np.arange(0,epochs)):
            
            p = np.arange(0,X.shape[0])
            np.random.shuffle(p)
            X = X[p]
            y = y[p]

            for (x, target) in self.getNextBatch(X, y, self.batchSize):
                A = [np.atleast_2d(x)]
                B = [np.atleast_2d(x)]
                
                for layer in np.arange(0, len(self.W)):
                    
                    net = A[layer].dot(self.W[layer])
                    out = self.vectorized_ELU(net)
                    
                    A.append(out)
                    B.append(net)
                    
                error = A[-1] - target
                D = [error * self.vectorized_ELU_Derivatice(B[-1])]
                
                for layer in np.arange(len(A) - 2, 0, -1):
                    delta = D[-1].dot(self.W[layer].T)
                    delta = delta * self.vectorized_ELU_Derivatice(B[layer])
                    D.append(delta)
                    
                D = D[::-1]
                
                for layer in np.arange(0, len(self.W)):
                    self.W[layer] -= self.alpha * A[layer].T.dot(D[layer])
                    
                
    def predict(self, X, addOnes = True):
        p = np.atleast_2d(X)
        
        if addOnes:
            p = np.hstack((p, np.ones([X.shape[0],1])))
        
        for layer in np.arange(0, len(self.W)):
            p = self.vectorized_ELU(np.dot(p, self.W[layer]))
            
        return p
    
    def computeLoss(self, X, y):
        y = np.atleast_2d(y)
        
        predictions = self.predict(X, addOnes = False)
        loss = np.sum((predictions - y)**2) / 2.0
        
        return loss




(trainX, testX, trainY, testY) = train_test_split(inputs, outputs, test_size = 0.50, random_state = 1)
np.random.seed(7468378)

# trainX = trainX.astype('float64')/255.0
# testX = testX.astype('float64')/255.0

# trainX = trainX.reshape([60000, 28*28])
# testX = testX.reshape([10000, 28*28])

trainY = to_categorical(trainY, 5)
testY = to_categorical(testY, 5)

print(inputs.shape)
print(outputs.shape)
model = FeedforwardNeuralNetworkSGD([inputs.shape[1], 32, 16, testY.shape[1]], 0.005, 16)
model.fit(trainX, trainY, 100, 10)

print("Training set accuracy")
predictedY = model.predict(trainX)
predictedY = predictedY.argmax(axis=1)

trainY = trainY.argmax(axis=1)
print(classification_report(trainY, predictedY))

print("Test set accuracy")
predictedY = model.predict(testX)
predictedY = predictedY.argmax(axis=1)

testY = testY.argmax(axis=1)
print(classification_report(testY, predictedY))