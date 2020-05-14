# -*- coding: utf-8 -*-
"""
DATS_6501: Capstone
Spring 2020

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Toolbox as tls

#Import data
df2018 = tls.dataClean('psam_', 2018)
df2018 = tls.additionalClean(df2018)

X = df2018.iloc[:,:-1]
catCols = X.columns[:10]
numCols = X.columns[10:]
            
cramersV = tls.cramersV_matrix(catCols, X)
plt.figure()
sns.heatmap(cramersV, annot=True, fmt = '.2f', cmap = 'BuGn_r', \
            xticklabels=catCols, yticklabels=catCols)
plt.title('Cramer\'s V Association Matrix')
plt.xticks(rotation = 45)
plt.tight_layout()

#plt.savefig('2018CramersHeatmap.png')
plt.close()
corrMatrix = tls.corr_matrix(numCols, X)   
plt.figure()
sns.heatmap(corrMatrix, annot=True, fmt = '.2f', cmap = 'BuGn_r', \
            xticklabels=numCols, yticklabels=numCols)
plt.title('Correlation Matrix')
plt.xticks(rotation = 45)
plt.tight_layout()
#plt.savefig('2018CorrelationHeatmap.png')
plt.close()

#based on evaluation of violin plots of categorical vs numerical variables, the
#   column MV was dropped due to strong relationship with Age

df2018 = df2018.drop(columns='MV')
#Produce the final sampling method and estimator
estimators = tls.getPipeline(df2018)
pipe = estimators[0][2]
samplers = tls.samplingTest(df2018, pipe)
sampling = samplers[0][2]


#Import and clean data from additional years
df2012 = tls.dataClean('ss12', 2012)
df2012 = tls.additionalClean(df2012)
df2012 = df2012.drop(columns='MV')

df2007 = tls.dataClean('ss07', 2007)
df2007 = tls.additionalClean(df2007)
df2007 = df2007.drop(columns='MV')
#Fit the final model and generate visuals of the performance metrics 
#   and feature importances
tls.finalForm(df2018, 2018, pipe, sampling)
tls.finalForm(df2012, 2012, pipe, sampling)
tls.finalForm(df2007, 2007, pipe, sampling)



