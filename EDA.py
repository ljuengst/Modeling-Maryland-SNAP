# -*- coding: utf-8 -*-
"""
DATS_6501: Capstone
Spring 2020

"""
import os
import Toolbox as tls
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import and manipulate data for ease of use
df2018 = tls.dataClean('psam_', 2018)
df2012 = tls.dataClean('ss12', 2012)
df2007 = tls.dataClean('ss07', 2007)

#Observe the data types rates of no response following data import and selection
print('\n2018 dataframe info:\n')
print(df2018.info())
print('\ntotal non-responses:')
print(df2018.isnull().sum())
print('\n2012 dataframe info:\n')
print(df2012.info())
print('\ntotal non-responses:')
print(df2012.isnull().sum())
print('\n2007 dataframe info:\n')
print(df2007.info())
print('\ntotal non-responses:')
print(df2007.isnull().sum())

#Change the categorical variable codes to their associated descriptions
df2018 = tls.convertCodeToDescrip(df2018)
df2012 = tls.convertCodeToDescrip(df2012)
df2007 = tls.convertCodeToDescrip(df2007) 
df2007.columns = df2018.columns

#Create selected dataframes for later graphing
#df2018fs = df2018[df2018.FS == 1]
#df2018nofs = df2018[df2018.FS == 0]
#df2012fs = df2012[df2012.FS == 1]
#df2012nofs = df2012[df2012.FS == 0]
#df2007fs = df2007[df2007.FS == 1]
#df2007nofs = df2007[df2007.FS == 0]

df2018['YEAR'] = [2018]*len(df2018)
df2012['YEAR'] = [2012]*len(df2012)
df2007['YEAR'] = [2007]*len(df2007)


#Graph the bar charts for each of the categorical data to get it's distribution
for df, year in [[df2018, '2018'], [df2012, '2012'], [df2007, '2007']]:
    for feat in df.columns[:11]:
#        picFile = feat + '_bar' + year + '.png'
        sns.catplot(x=feat,kind='count',data=df,orient="h")
        plt.xticks(rotation='vertical')
        plt.title('Distribution of {} Counts'.format(feat))
        plt.xlabel('{}'.format(feat))
        plt.ylabel('count')
        plt.tight_layout()
#        plt.savefig(os.path.join('.','images', year, picFile))
        plt.close()

#Create a pairplot to show the relationship between numerical variables
for df, year in [[df2018, '2018'], [df2012, '2012'], [df2007, '2007']]:
    picFile = 'pairplot' + year + '.png'
    pair = sns.pairplot(df.iloc[:,11:-2])
    plt.tight_layout()
#    plt.savefig(os.path.join('.','images', year, picFile))
    plt.close()

#Do some additional data cleaning to remove extremely low values
df2018 = tls.additionalClean(df2018)
df2012 = tls.additionalClean(df2012)
df2007 = tls.additionalClean(df2007)
dfall = pd.concat([df2018, df2012, df2007])
lst = dfall.columns[dfall.dtypes == 'object'].tolist()
for feat in lst:
    dfall[feat] = dfall[feat].astype('category')

#For each categorical feature and year, create pie charts to visually review the
#   proportions of feature categories in the SNAP 0 and 1 groups
for df, year in [[df2018, '2018'], [df2012, '2012'], [df2007, '2007']]:
    for feat in df.columns[:10]:
#        picFile = feat + '_pie' + year + '.png'
        dfyes = df[df['FS']==1].groupby(feat).count().reset_index().iloc[:, :2]
        dftot = df.groupby(feat).count().reset_index().iloc[:, :2]
        dfyes.columns = [feat, 'count']
        dftot.columns = [feat, 'count']
        if dfyes[feat].dtype == 'int64':
            labels = dfyes[feat].unique().tolist()
        else:
            labels=dfyes[feat].cat.categories.tolist()
        fig, axs = plt.subplots(1, 2, figsize=(15,6))
        axs[0].set_title('Receives SNAP')   
        axs[0].pie(dfyes.iloc[:,1], autopct='%1.1f%%', startangle=90, pctdistance=1.1)
        axs[1].set_title('Total Group')
        axs[1].pie(dftot.iloc[:,1], autopct='%1.1f%%', startangle=90, pctdistance=1.1)
        fig.suptitle('{} variable from year {}'.format(feat, year), fontsize=16)
        axs[1].legend(
                loc='upper right',
                labels=labels,
                bbox_to_anchor=(0.0, 1)
                )
#        plt.savefig(os.path.join('.','images', year, picFile))
        plt.close()

# Create a line graph and bar chart of the FS variable to visualize the imbalance
#picFile = 'SNAP.png'
dfGroup = dfall.groupby(['YEAR', 'FS']).count().iloc[:,0].unstack()
dfGroup['Ratio'] = dfGroup[1]/(dfGroup[1] + dfGroup[0])
dfGroup = dfGroup.reset_index()
dfGroup['YEAR'] = dfGroup['YEAR'].astype(str)
noben = dfGroup[0].tolist()
ben = dfGroup[1].tolist()
labels = dfGroup['YEAR'].tolist()
x = np.arange(0,len(labels))
barwidth = 0.4
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].set_title('Proportion of Individuals\n Receiving SNAP Benefits by year')
axs[0].plot(dfGroup.loc[:,'YEAR'], dfGroup.loc[:,'Ratio'], 'xb-')
axs[0].set_ylim((0, 0.1))
axs[0].set_xlabel('Year', fontsize=14)
axs[0].set_ylabel('Ratio', fontsize=14)
axs[1].set_title('Counts of SNAP Recipients \n to non-Recipients')
axs[1].bar(x-barwidth/2, ben, label = 'SNAP', width = barwidth)
axs[1].bar(x+barwidth/2, noben, label = 'no SNAP', width = barwidth)
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].set_xlabel('Year', fontsize=14)
axs[1].set_ylabel('count', fontsize=14)
axs[1].legend(loc='best')
#plt.savefig(os.path.join('.','images', 'allYears',picFile))
plt.close()

#For each feature create a stackplot to visualize changes in characteristic
#   proportions over time in the SNAP 0 and 1 groups
for feat in df2018.columns[:10]:
#    picFile = feat + '_stack' + '.png'
    dfa1 = dfall[(dfall['FS']==1) & (dfall['YEAR']==2018)].groupby(feat).count()\
          .reset_index().iloc[:, :2]
    dfa1.columns = [feat, '2018']
    dfa1['2018'] = dfa1['2018']/dfa1['2018'].sum() * 100
    dfa2 = dfall[(dfall['FS']==1) & (dfall['YEAR']==2012)].groupby(feat).count()\
          .reset_index().iloc[:, :2]
    dfa2.columns = [feat, '2012']
    dfa2['2012'] = dfa2['2012']/dfa2['2012'].sum() * 100
    dfa3 = dfall[(dfall['FS']==1) & (dfall['YEAR']==2007)].groupby(feat).count()\
          .reset_index().iloc[:, :2]
    dfa3.columns = [feat, '2007']
    dfa3['2007'] = dfa3['2007']/dfa3['2007'].sum() * 100
    dfa = pd.merge(dfa1, dfa2, how='outer', on=feat)
    dfa = pd.merge(dfa, dfa3, how ='outer', on=feat)
    dfa = dfa.T
 
    dftot1 = dfall[dfall['YEAR']==2018].groupby(feat).count()\
          .reset_index().iloc[:, :2]
    dftot1.columns = [feat, '2018']
    dftot1['2018'] = dftot1['2018']/dftot1['2018'].sum() * 100
    dftot2 = dfall[dfall['YEAR']==2012].groupby(feat).count()\
          .reset_index().iloc[:, :2]
    dftot2.columns = [feat, '2012']
    dftot2['2012'] = dftot2['2012']/dftot2['2012'].sum() * 100
    dftot3 = dfall[dfall['YEAR']==2007].groupby(feat).count()\
          .reset_index().iloc[:, :2]
    dftot3.columns = [feat, '2007']
    dftot3['2007'] = dftot3['2007']/dftot3['2007'].sum() * 100
    dftot = pd.merge(dftot1, dftot2, how='inner', on=feat)
    dftot = pd.merge(dftot, dftot3, how ='inner', on=feat)
    dftot = dftot.T
 
    y=[]
    for i in range(len(dfa.columns)):
        y.append(dfa.iloc[:0:-1,i].tolist())
    x =  dfa.index.tolist()[:0:-1]
    
    ytot=[]
    for i in range(len(dftot.columns)):
        ytot.append(dftot.iloc[:0:-1,i].tolist())
    labels = dftot.iloc[0,:].tolist()
    pal = sns.color_palette('hls')

    fig, axs = plt.subplots(1, 2, figsize=(15,8), sharey=True)
    axs[0].stackplot(x, y, labels=labels, colors=pal)
    axs[0].set_title('SNAP Participants')
    axs[0].set_xlabel('Year', fontsize = 14)
    axs[0].set_ylabel('Percentage', fontsize=14)
    axs[1].stackplot(x, ytot, labels=labels, colors=pal)
    axs[1].set_title('Total Respondents')
    axs[1].set_xlabel('Year', fontsize=14)
    axs[1].legend(bbox_to_anchor=(1.01,0.5), loc='center left')
    plt.suptitle('{} Feature Proportions by Year'.format(feat), 
                 x=0.45, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.ylim(0, 100)
#    plt.savefig(os.path.join('.','images', 'allYears', picFile))
    plt.close()
        
#The time variable doesn't appear to have a considerable impact on the visuals
#   so 2018 data was used for the next steps
#Create a boxplot to visualize relationships between categories and numerical
#   features
for cats in df2018.columns[:10]:
#        picFile = cats + '_box2018.png'
        fig, ax = plt.subplots(figsize = (8,8))
        sns.boxplot(x=cats,y='HINCP', hue = 'FS', data=df2018, ax=ax)
        ax.legend(bbox_to_anchor=(1.01,0.5), loc='center left',
                  title = 'SNAP')
        ax.set_xlabel('{}'.format(cats))
        ax.set_ylabel('HINCP')
        plt.xticks(rotation=45)
        plt.title('HINCP by {}'.format(cats), y=1)
        plt.subplots_adjust(top = 0.75)
        plt.tight_layout()
#        plt.savefig(os.path.join('.','images', '2018', picFile))
        plt.close()

for cats in df2018.columns[:10]:
    for nums in df2018.columns[10:-3]:
#        picFile = cats + 'vs' + nums + '_violin2018.png'
        fig, ax = plt.subplots(figsize = (8,8))
        sns.violinplot(x=cats,y=nums, hue = 'FS', data=df2018, ax=ax)
        ax.legend(bbox_to_anchor=(1.01,0.5), loc='center left',
                  title = 'SNAP')
        ax.set_xlabel('{}'.format(cats))
        ax.set_ylabel('{}'.format(nums))
        plt.xticks(rotation=45)
        plt.title('{} by {}'.format(nums, cats), y=1)
        plt.subplots_adjust(top = 0.75)
        plt.tight_layout()
#        plt.savefig(os.path.join('.','images', '2018', picFile))
        plt.close()


