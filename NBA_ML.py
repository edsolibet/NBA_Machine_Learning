# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:43:31 2022

@author: Carlo

This code explores the NBA players' stats from 2020-2021 basketball season, and uses a
KMeans machine learning algorithm to group them in clusters to show which players are 
most similar
"""

import pandas as pd # dataframes
import numpy as np
import matplotlib.pyplot as plt # plots
import seaborn as sns # correlation and heatmaps

'''
Load NBA Player Stats Data (csv)
'''
directory = "C:\\Users\\Carlo\\Documents\\Machine Learning\\"
filename = "NBA_PlayerRankings_20_21.csv" # csv data of top players from basketballmonster
nba = pd.read_csv(directory + filename)


''' 
Data stats
'''

# Get the number of top players and features
# nba.shape

# Average of each numeric column/feature in the dataset
# nba.mean()

# For a specific feature,
# nba.loc[:, "fg%"].mean()

# Pairwise plots
# sns.pairplot(nba[["fg%, "r/g", "b/g"]])

# Get visualtization of correlation by creating a heatmap
# correlation = nba[["fg%", "r/g", "b/g"]].corr()
# sns.heatmap(correlation, annot = True)
'''
Use kMeans to create 5 clusters of players which are most similar
'''
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters = 5, random_state = 1)
good_columns = nba._get_numeric_data().dropna(axis = 1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
n_iter = kmeans_model.n_iter_
inertia = kmeans_model.inertia_
cluster_centers = kmeans_model.cluster_centers_


''' Plot each player according to their corresponding cluster using the
Principal Component Analysis (PCA)
'''
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x = plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

'''
Predict player label from stats using the kmeans model
'''

def label_predict(player_name):
    player = good_columns.loc[nba["Name"]==player_name, :]
    player_list = player.values.tolist()
    player_cluster_label = kmeans_model.predict(player_list)
    return player_cluster_label

'''
Use Linear Regression to predict points per game (p/g) from minutes per game (m/g) due to their
high correlation. 
Split the data into training and test data sets at 80:20 ratio
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(nba[['m/g']], nba[['p/g']], test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression() # Create model
lr.fit(x_train, y_train) # Train the model
predictions = lr.predict(x_test) # Generate predictions

print(predictions)
print(y_test)

'''
Calculate confidence scores and mean squared error
'''

lr_confidence = lr.score(x_test, y_test)
print ("lr confidence (R^2): ", lr_confidence)

from sklearn.metrics import mean_squared_error
print ("Mean Squared Error (MSE): ", mean_squared_error(y_test, predictions))