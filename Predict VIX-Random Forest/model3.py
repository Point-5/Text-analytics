#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:52:02 2020

@author: lyd
"""

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt 

head = []
X = []
Y = []
with open('raw_data', 'r') as f:
    for line in f:
        lis = line.strip(' \n').split('\t')
        if len(head)==0:
                head = lis
        else:
            con = []
            for i in range(1,6):
                x = lis[i]
                if x=='':
                    con.append(0.0)
                else:
                    if '%' in x:
                        x = float(x.replace('%',''))/100.
                    else:
                        x = float(x)
                    con.append((x))
            X.append(con)
            Y.append(int(lis[0]))

names = head[1:len(head)]
X = np.array(X)
Y = np.array(Y)
print names
print X
print Y

rf = RandomForestClassifier(n_estimators=20, max_depth=4)
scores = []
# cross-validation to evaluate each model
for i in range(X.shape[1]):
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",cv=ShuffleSplit(len(X), 0.3))
    scores.append((float(format(np.mean(score), '.3f')), names[i]))

# with all feature
score = cross_val_score(rf, X, Y, scoring="r2", cv=ShuffleSplit(len(X), 0.3))
scores.append((float(format(np.mean(score), '.3f')), 'all features'))

# drop ‘Sentiment‘ feature
X = np.delete(X, 0, axis=1)
score = cross_val_score(rf, X, Y, scoring="r2", cv=ShuffleSplit(len(X),0.3))
scores.append((float(format(np.mean(score), '.3f')), 'drop Sentiment feature'))


print(sorted(scores, reverse=True))

ylabel=map(itemgetter(0), scores)
label=map(itemgetter(1), scores)
plt.bar(range(len(label)), ylabel,tick_label=label)
plt.tick_params(labelsize=6)

i=0
for a, b in zip(range(len(label)), ylabel):
    if i<=len(label):
        plt.text(a, b, ylabel[i], ha='center', va='bottom', fontsize=6)
        i=i+1
plt.show

    
