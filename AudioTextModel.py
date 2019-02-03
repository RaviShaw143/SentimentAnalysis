# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical


       
#splits the data into training and testing data and then reduce the features using PCA      
def splitDataAfterPCA(input, output):
    train_data, test_data, train_lbl, test_lbl = train_test_split(input, output, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_data)

    # Apply transform to both the training set and the test set.    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    pca = PCA(.95) 
    # Fit on training set only.
    pca.fit(train_data)
    
    # Apply transform to both the training set and the test set.
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    return train_data, test_data, train_lbl, test_lbl 
        

def getDataForLSTM(X,Y, testX, testY):
    
    #dividing the Text features training data into further train and validation set
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.20, random_state=0)
    
    #expands the dimension of data as LSTM expects a 3 dimension input
    trainX = np.expand_dims(trainX, 2)
    valX = np.expand_dims(valX, 2)
    trainY = to_categorical(trainY)
    valY = to_categorical(valY)
    
    #testing set to evaluate the model for Audio Features only
    testX = np.expand_dims(testX, 2)
    testY = to_categorical(testY)
    
    return trainX, trainY, valX, valY, testX, testY
    


    
def LSTModelAudText(trainX, trainY, valX, valY, testX, testY):
    model = Sequential()
    model.add(LSTM(int(len(trainX[0])), dropout=0.2, return_sequences=True, recurrent_dropout=0.2, input_shape = (int(len(trainX[0])),1)))
    model.add(LSTM(390, dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(120, dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))     
    
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])    
        
    model.fit(trainX, trainY,
            batch_size=10,
            epochs=5,
            verbose=2,
            validation_data=(valX, valY))
    
    score, acc = model.evaluate(testX, testY,
                                batch_size=50,
                                verbose=2)
    
    print('Audio text SCORES')
    print('Test score:', score)
    print('Test accuracy:', acc)
    


featureFilesDirectory = os.path.join(os.getcwd(),"RNN")
teamAudioText = os.path.join(featureFilesDirectory,"audioTextSWFeat.csv")
audioTextFeat = pd.read_csv(teamAudioText)
inputAudioTextFeat = audioTextFeat.loc[:, audioTextFeat.columns != 'sentiment']
print (inputAudioTextFeat.shape)
output = audioTextFeat["sentiment"][:].values 


#splits the data of joined audioAndTextFeatures, AudioFeatures, and Text Features in to training and testing data
train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext = splitDataAfterPCA(inputAudioTextFeat, output)

#dividing the features training data into further train and validation set
trainX_AT, trainY_AT, valX_AT,  valY_AT, testX_AT, testY_AT  = getDataForLSTM(train_X_Audtext, train_Y_Audtext, test_X_Audtext, test_Y_Audtext)

print("LSTM audio Text shapes:")

print (trainX_AT.shape)

#reportRNNResults()
print("Audio and Text SW Model  t results:")
#LSTModelTextSW(trainX_TextSW,  trainY_TextSW, valX_TextSW, valY_TextSW, testX_TextSW, testY_TextSW)
LSTModelAudText(trainX_AT, trainY_AT, valX_AT,  valY_AT, testX_AT, testY_AT)
print (trainX_AT.shape)       