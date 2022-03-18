# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:55:49 2022

@author: Carlo
"""
from __future__ import division
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
Import file "kobe_data.csv"
'''

directory = "C:\\Users\\Carlo\\Documents\\Machine Learning\\"
filename = "kobe_data.csv"
raw_data = pd.read_csv(directory+filename)

'''
Exploration and cleanup
'''

# Drop rows in which "shot_made_flag" has null

clean_data = raw_data.dropna()
test_data = raw_data[raw_data.isnull()]
#nonull_data = raw_data[raw_data['shot_made_flag'].isnull()==0]


#Combine features 'minutes_remaining' and 'seconds_remaining'

clean_data.loc[:, 'time_remaining'] = (clean_data.loc[:, 'minutes_remaining']*60 + 
          clean_data.loc[:, 'seconds_remaining'])

'''
Data Visualization
'''
do_plot = False
if do_plot:
    plt.figure(figsize=(10,10))
    
    # returns elements which satisfy condition
    colors = np.where(clean_data['shot_made_flag']==1, "Dodgerblue", "Crimson")
    #colors.shape
    plt.subplot(121)
    plt.title('loc_x vs loc_y')
    plt.xlabel('loc_x')
    plt.ylabel('loc_y')
    plt.scatter(clean_data.loc_x, clean_data.loc_y, color = colors, s = 4, alpha = 0.5)
    
    plt.subplot(122)
    plt.title('lon vs lat')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.scatter(clean_data.lon, clean_data.lat, color = colors, s = 4, alpha = 0.5)

# Visualize the shot zone range
groups = clean_data.groupby('shot_zone_range')
# frame as in size of each group
for column, frame in groups:
    print ('column: {0}, frame: {1}'.format(column, len(frame)))


# Function to group and plot any area category
import matplotlib.cm as cm

def grouped_plot(feature):
    groups = clean_data.groupby(feature)
    colors = cm.Set1(np.linspace(0, 1, len(groups)))
    for g, c in zip(groups, colors):
        plt.scatter(g[1].loc_x, g[1].loc_y, color = c, s = 3, alpha = 0.5)
        
if do_plot:
    plt.figure(figsize=(5,10))
    
    plt.figure(figsize=(15,10))
    plt.subplot(131)
    plt.title('shot_zone_area')
    plt.xlabel('loc_x')
    plt.ylabel('loc_y')
    grouped_plot('shot_zone_area')
    
    plt.subplot(132)
    plt.title('shot_zone_basic')
    plt.xlabel('loc_x')
    plt.ylabel('loc_y')
    grouped_plot('shot_zone_basic')
    
    plt.subplot(133)
    plt.title('shot_zone_range')
    plt.xlabel('loc_x')
    plt.ylabel('loc_y')
    grouped_plot('shot_zone_range')

'''
preprocessing
'''
clean_data.index = range(len(clean_data))
clean_data.loc[:,'month'] = clean_data['game_date'].str[5:7].astype(int) # get month 
clean_data.loc[:,'year'] = clean_data['game_date'].str[:4].astype(int) # get year
clean_data = clean_data.drop('game_date', 1)
clean_data = clean_data.drop('shot_id', 1)
clean_data = clean_data.drop('game_id', 1)
clean_data = clean_data.drop('game_event_id', 1)
clean_data = clean_data.drop('team_name', 1)
clean_data = clean_data.drop('season', 1)
clean_data.loc[:,'time_remaining'] = pd.Series(clean_data.minutes_remaining*60 + clean_data.seconds_remaining)
clean_data = clean_data.drop('seconds_remaining', 1)
clean_data = clean_data.drop('minutes_remaining', 1)
home_series = [1 if 'vs.' in x else 0 for x in clean_data.matchup]
clean_data.loc[:,'home'] = home_series
clean_data = clean_data.drop('matchup', 1)

# check correlation between loc_x and lon and loc_y and lat
A = clean_data[['loc_x', 'lon', 'loc_y', 'lat']].corr()
pal = ['Crimson', 'Dodgerblue']
sns.pairplot(clean_data, vars=['loc_x', 'loc_y', 'lat', 'lon', 'shot_distance'],
             hue='shot_made_flag', size=3, palette = pal)

clean_data = clean_data.drop('lon', 1)
clean_data = clean_data.drop('lat', 1)

'''
splitting to training and testing set
'''
from sklearn.model_selection import train_test_split
y = np.array(clean_data['shot_made_flag']).astype(int) #array
x = clean_data.drop('shot_made_flag', 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

'''
Standardize loc x and loc y to prevent unnecessary divisions by 0 
Redistribute the data around a mean of zero in one std
'''

from sklearn import preprocessing

A = x_train[['loc_x', 'loc_y']].copy()
scaler = preprocessing.StandardScaler().fit(A)
A_st = scaler.transform(A)
print (A_st.mean(axis = 0), A_st.std(axis=0))
x_train.loc[:,'angle'] = A_st[:,0]/A_st[:,1]

B = x_test[['loc_x', 'loc_y']]
B_st = scaler.transform(B)
x_test.loc[:,'angle'] = B_st[:,0]/B_st[:,1]
x_test = x_test.drop('loc_x', 1)
x_test = x_test.drop('loc_y', 1)
x_train = x_train.drop('loc_x', 1)
x_train = x_train.drop('loc_y', 1)

'''
We use Pandas' "get_dummies()" to convert categorical fields and use Scikit's classifiers
'''
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# we should not look into test set during pre-processing
# Get columns that are in training but not in test
not_in_test = np.setdiff1d(x_train.columns, x_test.columns)
not_in_train = np.setdiff1d(x_test.columns, x_train.columns)

for c in not_in_test: #add columns to test, setting to zero
    x_test[c] = 0

# make test be of same schema as training
x_test = x_test[x_train.columns]

'''
Decision tree
'''

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics

r = np.arange(1, 10000, 500)
a = []
b = []

print (x_test.shape, x_train.shape, y_test.shape, y_train.shape)

if do_plot:
    
    for i in r:
        clf = tree.DecisionTreeClassifier(min_samples_leaf = i,
                                          class_weight = 'balanced')
        clf = clf.fit(x_train, y_train)
        a.append(clf.score(x_test, y_test))
        b.append(clf.score(x_train, y_train))
        # this is done solely for visualizing the overfitting effect
        
    plt.figure(figsize = (8, 5))
    plt.plot(r, a)
    plt.plot(r, b)
    
    plt.axvspan(0, 500, color = 'r', alpha = 0.1, lw = 0)
    plt.axvspan(500, 1550, color = 'g', alpha = 0.1, lw = 0)
    
    plt.xlim(0, 10000)
    plt.ylim(0.5, 1)
    plt.xlabel('Min samples in leaf (inverse complexity)')
    plt.ylabel('Accuracy')
    # for low # of samples, the accuracy on training is very high, while it is 
    # low for test data, showing overfitting
    # sweet spot is around 500-1500 samples in each leaf

'''
Decision Tree
k-fold cross-validation
'''
r = np.arange(250, 5000, 250)
scores = []
for i in r:
    clf = tree.DecisionTreeClassifier(min_samples_leaf = i, class_weight = 'balanced')
    m = cross_val_score(clf, x_train, y_train, cv = 10, scoring = 'f1')
    scores.append((i, m.mean(), m.var()))

# minimum number of 1750 samples maximizes accuracy and f1

clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=1750,
                                  class_weight='balanced')
clf = clf.fit(x_train, y_train)

from sklearn.externals.six import StringIO
with open("kobe3.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f,
                             feature_names=x_train.columns,
                             class_names=True, filled=True,
                             rounded=True, special_characters=True)

xviztr = DataFrame()
xviztst = DataFrame()
xviztr['time_remaining'] = x_train['time_remaining']
xviztr['shot_distance'] = x_train['shot_distance']
xviztst['time_remaining'] = x_test['time_remaining']
xviztst['shot_distance'] = x_test['shot_distance']

t_min, t_max = -5, 714
d_min, d_max = -0.5, 74
tt, dd = np.meshgrid(np.arange(t_min, t_max, 1),
                     np.arange(d_min, d_max, 1))
colors = np.where(y_test==1, "black", "red")

def classgraph(n, i):
    sp = plt.subplot(3, 2, i)
    sp.set_title(n)
    plt.xlabel('time_remaining')
    plt.ylabel('distance')
    clfviz = tree.DecisionTreeClassifier(min_samples_leaf=n, class_weight='balanced')
    clfviz.fit(xviztr,y_train)
    z=clfviz.predict(list(zip(tt.ravel(), dd.ravel())))
    z = z.reshape(tt.shape)
    cs = plt.contourf(tt, dd, z, cmap = plt.cm.Paired)
    plt.scatter(xviztst['time_remaining'], xviztst['shot_distance'], c=colors, 
                s=2, alpha=1, lw=0)

leaf = [250, 500, 750, 1000, 1500, 1750]
plt.figure(figsize=(10,15))
for k in leaf:
    classgraph(k, leaf.index(k)+1)
    
'''
Model Evaluation
'''
y_predict = clf.predict(x_test)
metrics.accuracy_score(y_test, y_predict, normalize=True)

# Does this accuracy score beat the baseline majority classifier?

no = np.where(y_test == 0)[0].size
yes = np.where(y_test == 1)[0].size

print (no/(no+yes))

'''
Confusion matrix
'''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

if 1:
    plt.figure(figsize=(4,4))
    plt.title('Confusion matrix')
    ax = sns.heatmap(cm, cmap =plt.cm.Blues, annot=True, fmt='d', square=True, linewidths=0.5)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Real class')

'''
Examine precision, recall and f1 score
'''
metrx = {'precision':metrics.precision_score(y_test, y_predict),
         'recall': metrics.recall_score(y_test, y_predict),
         'f1': metrics.f1_score(y_test, y_predict)}
print (metrx)