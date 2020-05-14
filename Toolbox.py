# -*- coding: utf-8 -*-
"""
DATS_6501: Capstone
Spring 2020

"""
import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
import time
import researchpy as rp
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_recall_fscore_support

def cramersV_matrix(catFeat, X):
    '''
    Function that takes in the list of categorical features and returns 
        the Cramer's V matrix which is similar to chi^2 except it adjusts for
        sample size
    '''
    assocMatrix = []
    for col1 in catFeat:
        for col2 in catFeat:
            tab, res = rp.crosstab(X[col1], X[col2], test = 'chi-square')
            assocMatrix.append(res.iloc[2,1])
    assocMatrix = np.array(assocMatrix).reshape((len(catFeat),-1))
    
    return assocMatrix

def corr_matrix(numFeat, X):
    '''
    Function that takes in the list of numerical features and returns the
        correlation matrix
    '''
    corr_matrix=[]
    for col1 in numFeat:
        for col2 in numFeat:
            x = X[col1].values
            y = X[col2].values
            corr_matrix.append(round(np.corrcoef(x,y)[0][1],3))
    corr_matrix = np.array(corr_matrix).reshape(len(numFeat),-1)
    
    return corr_matrix
 
def dataClean(prefix, year):
    '''
    Function to import, merge, and manipulate ACS data from different years
    prefix arg: start of filename common to all the csv files from that year
    '''
    #Specify features of interest to import
    pCols = ['SERIALNO', 'ST', 'AGEP', 'CIT', 'MAR', 'SCHL', 'SEX', \
             'ESR', 'RAC1P']
    hCols = ['SERIALNO', 'REGION', 'NP', 'VEH', 'HINCP', 'HHL', \
             'HUPAC', 'LNGI', 'MV', 'FS']
    
    #2007 year has a different code for disabilities than 2012 and 2018
    if prefix == 'ss07':
        pCols.append('DS')
    else:
        pCols.append('DIS')
    
    dfHa = pd.read_csv(os.path.join('.','data',(prefix + 'husa.csv')), usecols=hCols)
    dfHb = pd.read_csv(os.path.join('.','data',(prefix + 'husb.csv')), usecols=hCols)
    dfPa = pd.read_csv(os.path.join('.','data',(prefix + 'pusa.csv')), usecols=pCols)
    dfPb = pd.read_csv(os.path.join('.','data',(prefix + 'pusb.csv')), usecols=pCols)

    dfH = pd.concat([dfHa, dfHb])
    dfP = pd.concat([dfPa, dfPb])

    df = pd.merge(dfP, dfH, on = 'SERIALNO', how = 'inner')
    
    #Consider only Maryland
    df = df[df.ST == 24]
      
    #Remove group quarters records from consideration
    if prefix == 'psam_':
        df = df[~df.SERIALNO.str.match(str(year) + 'GQ')]
    else:
        df = df[~df.HINCP.isna()]

    #Remove children under the age of 16 from consideration
    df = df[df.AGEP >= 16]
    
    #Randomly select one adult from each household
    idx = []
    for serialno in df.SERIALNO.unique():
        house = df[df.SERIALNO == serialno]
        m = np.random.choice(len(house))
        idx.append(house.iloc[m,:].name)
    df = df.loc[idx, :]
    
    #Convert float64 columns to int64
    colst = []
    for i in df.columns:
        colst.append(df[i].dtype=='float64')
    colst = df.columns[colst]
    df[colst] = df[colst].astype(np.int64)

    #Binarize the FS feature: 2007 was benefit amount else change 2 (no FS) to 0
    #Change 2 to 0 in LNGI binary variable so values are 0 or 1
    #Recode educ level: 0 - no school, 1 - some school, 2 - HS diploma or equiv,
    #                   3 - some college, 4 - college degree
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        if df.LNGI.loc[i] == 1:
            df.loc[i, 'LNGI'] = 0
        else:
            df.loc[i, 'LNGI'] = 1
    if prefix == 'ss07':
        for i in range(len(df)):
            if df.FS.loc[i] >0:
                df.loc[i, 'FS'] = 1
            if df.DS.loc[i] == 2:
                df.loc[i, 'DS'] = 0
            if df.SCHL.loc[i] == 1:
                df.loc[i, 'SCHL'] = 0
            elif df.SCHL.loc[i] > 1 and df.SCHL.loc[i] < 9:
                df.loc[i, 'SCHL'] = 1
            elif df.SCHL.loc[i] == 9:
                df.loc[i, 'SCHL'] = 2
            elif df.SCHL.loc[i] > 9 and df.SCHL.loc[i] < 12:
                df.loc[i, 'SCHL'] = 3
            else:
                df.loc[i, 'SCHL'] = 4
    else:
        for i in range(len(df)):
            if df.FS.loc[i] == 2:
                df.loc[i, 'FS'] = 0
            if df.DIS.loc[i] == 2:
                df.loc[i, 'DIS'] = 0
            if df.SCHL.loc[i] == 1:
                df.loc[i, 'SCHL'] = 0
            elif df.SCHL.loc[i] > 1 and df.SCHL.loc[i] <= 15:
                df.loc[i, 'SCHL'] = 1
            elif df.SCHL.loc[i] == 16 or df.SCHL.loc == 17:
                df.loc[i, 'SCHL'] = 2
            elif df.SCHL.loc[i] > 17 and df.SCHL.loc[i] < 20:
                df.loc[i, 'SCHL'] = 3
            else:
                df.loc[i, 'SCHL'] = 4
 
    #Cast nominal columns to categorical data type
    catCols = ['CIT','MAR','SCHL','SEX','ESR','RAC1P','HHL','MV','HUPAC']
    rac1p_cat = CategoricalDtype(categories=np.arange(1,10), ordered=True)
    for feat in catCols:                  
        if feat == 'RAC1P':
            df[feat] = df[feat].astype(rac1p_cat)
        else:
            df[feat] = df[feat].astype('category')
    if 5 in df['ESR'].cat.categories:
        df['ESR'] = df['ESR'].cat.remove_categories([5])
        df = df.dropna()
        df = df.reset_index(drop=True)
        
    #Limit and Rearrange columns for future simplicity    
    if prefix == 'ss07':
        catCols.append('DS')
    else:
        catCols.append('DIS')
    catCols.append('LNGI')
    numCols = ['NP', 'AGEP', 'VEH', 'HINCP']
    cols = catCols + numCols + ['FS']
    df = df.loc[:, cols]
   
    return df

def convertCodeToDescrip(df):
    '''
    This functions facilitates graph interpretation by converting numeric code
    responses to their related descriptors which are in the codeMap.csv file
    '''
    codeMap = pd.read_csv(os.path.join('.','data','codeMap.csv'), header = None)
    codeDict = {}    
    for row in range(len(codeMap)):
        outlst = []
        i = 1
        while i < codeMap.shape[1]:
            if ~np.isnan(codeMap.iloc[row, i]):
                inlst = []
                inlst.append(int(codeMap.iloc[row, i]))
                inlst.append(codeMap.iloc[row, i+1])
                outlst.append(inlst)
            i += 2
        codeDict[codeMap.iloc[row, 0]] = outlst
    
    for key in codeDict.keys():
        tmplst = []
        for code, descrip in codeDict[key]:
            tmplst.append(descrip)
        df[key].cat.categories = tmplst
    
    return df

def additionalClean(df):
    '''
    Function to remove additional data following EDA
    '''
    for feat in df.columns[:9]:
        cutoff = int(0.005*len(df))
        tmpSeries = df[feat].value_counts()
        tmplist = tmpSeries[tmpSeries < cutoff].index.tolist()
        df[feat].cat.remove_categories(tmplist, inplace=True)
        df.dropna(inplace=True)
    
    df = df.drop(columns = 'LNGI')
    
    return df
    
    
    
    
def getPipeline(df):
    '''
    Function that takes df and does a GridSearch to select model and hyperparameters
        that achieved the highest accuracy
    returns a sorted by accuracy score list of score, best performing parameters,
        and pipeline for each estimator
    '''
    #Take a 1000 datapoint stratified random sample to perform model selection and 
    #hyperparameter tuning
    minority = (df.FS == 1).sum()/len(df)
    dfMin = df[df.FS == 1]
    dfMaj = df[df.FS == 0]
    m = np.int(np.ceil(1000*minority))
    dfLiteMin = dfMin.sample(n = m, axis = 0, random_state=0)
    dfLiteMaj = dfMaj.sample(n = (1000-m), axis = 0, random_state = 0)
    dfLite = pd.concat((dfLiteMaj, dfLiteMin))
    
    
    X = dfLite.iloc[:, :-1]
    y = dfLite.iloc[:, -1].values
    
    #Preprocess: numerical columns-standard scaler, categorical columns-one hot enc
    numCols = ['NP', 'AGEP', 'VEH', 'HINCP']
    catCols = ['CIT','MAR','SCHL','SEX','ESR','RAC1P','HHL','HUPAC']
    codeList = []
    for feat in catCols:
        tmp = df[feat].cat.categories.to_list()
        codeList.append(tmp)
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numCols),
            ('cat', OneHotEncoder(categories=codeList), catCols)], remainder='passthrough')
    
    #Set up the pipeline and GridSearch parameters
    clfs = {'lr': LogisticRegression(penalty='elasticnet', solver='saga', 
                                     class_weight = 'balanced', random_state=0, 
                                     max_iter=500),
            'rf': RandomForestClassifier(class_weight = 'balanced', random_state=0),
            'svc': SVC(class_weight = 'balanced', random_state=0)}
    
    pipe_clfs = {}
    for name, clf in clfs.items():
        pipe_clfs[name] = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
    
    param_grids = {}
    
    grid = [{'clf__C': [0.1, 1, 10, 100],
             'clf__l1_ratio': [0, 0.25, 0.5, 0.75, 1]}]
    param_grids['lr'] = grid
    
    grid = [{'clf__n_estimators': [2, 10, 30],
             'clf__min_samples_split': [2, 10, 30],
             'clf__min_samples_leaf': [1, 10, 30]}]
    param_grids['rf'] = grid
    
    grid = [{'clf__C': [0.1, 1, 10, 100],
             'clf__gamma': [0.01, 0.1, 1, 10, 100],
             'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
    param_grids['svc'] = grid
    
    #Create the GridSearch and save the resulting scores and params 
    print('------------Commencing GridSearch------------')
    param_estimators = []
        
    
    for name in pipe_clfs.keys():
        #GridSearchCV
        start=time.time()
        gs = GridSearchCV(estimator=pipe_clfs[name],
                          param_grid=param_grids[name],
                          scoring='recall',
                          n_jobs=-1,
                          cv=StratifiedKFold(n_splits=10,
                                             shuffle=True,
                                             random_state=0))
        #Fit the pipeline
        gs = gs.fit(X, y)
        
        # Update best_score_param_estimators
        param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
        end = time.time()
        print('------------Completed {} Grid in time: {}------------'.format(name, end-start))
        print('------------Score for estimator {}: {} ------------'.format(name, gs.best_score_))
    #Print the scores and best parameters for each estimator
    for x in param_estimators:
        print([x[0], x[1], type(x[2].named_steps['clf'])], end='\n\n')
    
    param_estimators.sort(key = lambda x : x[0], reverse = True)

    return param_estimators

def samplingTest(df, pipe_clf):
    '''
    Function to test SMOTENC and Random Oversampling methods and return the 
    results
    '''
    #Split data into train and test sets
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        stratify = y, random_state = 0)
    
    #Define smotenc and random oversampling methods
    cats = np.arange(0,9,1)
    smotenc = SMOTENC(random_state=0, categorical_features=cats)
    randos = RandomOverSampler(random_state=0)
    sampMethod = {'smotenc': smotenc, 'randos': randos}
    results = []
    for label, overSamp in sampMethod.items():
        X_train_res, y_train_res = overSamp.fit_resample(X_train, y_train)
        print('------------Oversampling complete------------')
        
        #Fit training data to the pipeline
        pipe_clf.fit(X_train_res, y_train_res)
        y_pred = pipe_clf.predict(X_test)
        
        #Get the recall score (rounding to three decimal places)
        score = round(recall_score(y_test, y_pred), 3)
        print('------------Score for {}: {}------------'.format(label, score))
        results.append([label, score, overSamp])
    results.sort(key = lambda x : x[1], reverse = True)
    
    return results

def finalForm(df, year, pipe_clf, overSamp):
    '''
    Function that takes in the pipeline and oversampling method and produces
    the final model with metrics and feature importances
    '''
    print('------------Beginning Final Model Generation------------')
    #Split data into train and test sets
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                        stratify = y, random_state = 0)    
    #Define and execute random oversampling
    X_train_res, y_train_res = overSamp.fit_resample(X_train, y_train)
    print('------------Oversampling complete------------')
    
    #Fit the pipeline
    pipe_clf.fit(X_train_res, y_train_res)
    
    #Get the score (rounding to two decimal places)
    score = round(pipe_clf.score(X_test, y_test), 3)
    print('------------Accuracy Score: {}------------'.format(score))
    
    #Create a visual of the confusion matrix and associated scores from test set
    y_pred = pipe_clf.predict(X_test)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    scoresXlabel = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}".format(
                    accuracy,precision,recall)
#    figName = ('{}'.format(year) + 'confusion.png')
    plt.figure()
    sns.heatmap(confusionMatrix, fmt='d', annot=True, cmap='Purples')
    plt.title('Confusion Matrix of Test Set Predictions (data year: {})'.format(year))
    plt.xlabel('Predicted Label' + scoresXlabel, fontsize = 'medium')
    plt.ylabel('True Label')
    plt.tight_layout()
#    plt.savefig(figName)
    plt.close()
    
    #Evaluate feature importance on both train and testing sets using permutation
    print('------------Beginning Permutation Feature Importance Test Set------------')
    featImport = permutation_importance(pipe_clf, X_test, y_test, n_repeats=10,
                                random_state=0, n_jobs=-1)
    sortIndx = featImport.importances_mean.argsort()[::-1]
    
    featDftest = pd.DataFrame(data = featImport.importances[sortIndx].T, 
                              columns=X_test.columns[sortIndx])
#    figName = ('{}'.format(year) + 'featureImp.png')
    plt.figure()
    sns.boxplot(data=featDftest, orient = 'h')
    plt.title('Feature Importances from Test Set (data year: {})'.format(year))
    plt.ylabel('Features')
    plt.xlabel('Permutation Importances')
#    plt.savefig(figName)
    plt.close()

