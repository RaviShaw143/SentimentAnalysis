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
    print(train_data.shape)
    print(test_data.shape)
    print(train_lbl.shape)
    print(test_lbl.shape)
    
    return train_data, test_data, train_lbl, test_lbl 
    
#trains a given model with training data provided and then returns the score of the model
def trainModelAndGetScore(model,train_data, test_data, train_lbl, test_lbl):
    model.fit(train_data,train_lbl)
    result = model.score(test_data,test_lbl)
    pred = model.predict(test_data) 
    acc = accuracy_score(test_lbl, pred)
    f1 = f1_score(test_lbl, pred, average = "micro")
    prec = precision_score(test_lbl, pred,average = "micro")
    rec = recall_score(test_lbl, pred,average = "micro")
    result = {'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result

#trains a given model with training data provided by finding the best hyperparameters for the model and  then returns the precision and recall score of the model
def findBestEstimator(model, tuned_parameters, X_train, X_test, y_train, y_test):
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, tuned_parameters, cv=10,
                        scoring='%s_micro' % score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
       # print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    

#Gives SVC models (SVC, LinearSVC and NuSVC) results 
def getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001],
                        'C': [0.001,0.01,0.1,1, 10, 100, 1000]},
                        {'kernel': ['linear'],'gamma': [0.01, 0.001, 0.0001], 'C': [0.001,0.01,0.1,1, 10, 100, 1000]}]
    #findBestEstimator(SVC(),tuned_parameters,train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com)
    
    tuned_parameters_lin = [{'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['ovr']},
                        {'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']},
                        {'dual': [True],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']}]
    
    tuned_parameters_nuSVC = [{'nu':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'gamma': [0.01, 0.001, 0.0001]}]
    
    findBestEstimator(SVC(),tuned_parameters,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(LinearSVC(),tuned_parameters_lin,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(NuSVC(),tuned_parameters_nuSVC,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
 
#gives RandomForestClassifier models (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier) results
def getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):    
    tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': [None],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
    tune_param_DClf = [{'max_depth':[3],'min_samples_leaf':[1]}]
    
    findBestEstimator(RandomForestClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(ExtraTreesClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(DecisionTreeClassifier(),tune_param_DClf,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)

#gives Logistic Regression model results
def getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):        
    tune_params_LR2 = [{'penalty':['l2'], 'solver':['newton-cg','lbfgs','sag'], 'max_iter':[5000],'multi_class':['multinomial','ovr']}]
    tune_params_LR1 = [{'penalty':['l1'], 'solver':['liblinear','saga'],'max_iter':[5000],'multi_class':['ovr']}]
    findBestEstimator(LogisticRegression(),tune_params_LR1,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com )
    findBestEstimator(LogisticRegression(),tune_params_LR2,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com )

#gets the results of all models used at once 
def getSupervisedModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):
    getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)



featureFilesDirectory = './TEAMDataset/CsvFiles'
teamAudioFeat = os.path.join(featureFilesDirectory,"TEAMAudio_Features.csv")
teamTextFeat = os.path.join(featureFilesDirectory,"TEAMTextFeatures.csv")
teamTextSWFeat = os.path.join(featureFilesDirectory,"TEAMSWTextFeatures.csv")
teamAudioText = os.path.join(featureFilesDirectory,"audioTextFeat1.csv")
teamAudioTextSW =os.path.join(featureFilesDirectory,"audioTextSWFeat.csv")


audioFeat = pd.read_csv(teamAudioFeat)
inputAudioFeat = audioFeat.loc[:, audioFeat.columns != 'sentiment']
print (inputAudioFeat.shape)

textFeat = pd.read_csv(teamTextFeat)
inputTextFeat = textFeat.loc[:, textFeat.columns != 'sentiment']
print (inputTextFeat.shape)


textSWFeat = pd.read_csv(teamTextSWFeat)
inputTextSWFeat = textSWFeat.loc[:, textSWFeat.columns != 'sentiment']
print (inputTextSWFeat.shape)

audioTextFeat = pd.read_csv(teamAudioText)
inputAudioTextFeat = audioTextFeat.loc[:, audioTextFeat.columns != 'sentiment']
print (inputAudioTextFeat.shape)

audioTextSWFeat = pd.read_csv(teamAudioTextSW)
inputAudioTextSWFeat = audioTextSWFeat.loc[:, audioTextSWFeat.columns != 'sentiment']
print (inputAudioTextSWFeat.shape)


output = audioFeat["sentiment"][:].values 


#splits the data of joined audioAndTextFeatures, AudioFeatures, and Text Features in to training and testing data
train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud = splitDataAfterPCA(inputAudioFeat, output)
train_X_text, test_X_text, train_Y_text, test_Y_text = splitDataAfterPCA(inputTextFeat, output)
train_X_textSW, test_X_textSW, train_Y_textSW, test_Y_textSW = splitDataAfterPCA(inputTextSWFeat, output)
train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext = splitDataAfterPCA(inputAudioTextFeat, output)
train_X_AudtextSW, test_X_AudtextSW, train_Y_AudtextSW, test_Y_AudtextSW = splitDataAfterPCA(inputAudioTextSWFeat, output)

def reportSupervisedModelsResults(feature):
    
    if feature == "audio":
        getSupervisedModelResults(train_X_Aud, test_X_Aud, train_Y_Aud, test_Y_Aud)
    if feature == "text":
        getSupervisedModelResults(train_X_text, test_X_text, train_Y_text, test_Y_text)
    if feature == "textSW":
        getSupervisedModelResults(train_X_textSW, test_X_textSW, train_Y_textSW, test_Y_textSW)
    if feature == "audioText":
        getSupervisedModelResults(train_X_Audtext, test_X_Audtext, train_Y_Audtext, test_Y_Audtext)
    if feature == "audioTextSW":
        getSupervisedModelResults(train_X_AudtextSW, test_X_AudtextSW, train_Y_AudtextSW, test_Y_AudtextSW)

#use report results method to get the results on any supervised trained model
#print ("Audio & Text Results::")    
#reportResults("audioText")

#print ("Audio & Text Results ends")

#print ("Audio & Text Stop Words Results")

#reportResults("audioTextSW")

#print ("Audio & Text Stop Words Results ends")

