# imports
import sys
import os
sys.path.append('../..')
from vardata import *

import pickle
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE 

#helper functions
def clean_column(column):
    column = column.strip(' ')
    column = column.replace('/', '_')
    column = column.replace(' ', '_')
    column = column.lower()
    return column

print(bcolors.OKBLUE + "Loading pickle" + bcolors.ENDC)
with open(os.path.join(DATASET_DIR, DATASET_NAME, DATASET_NAME + '_base.pkl'), 'rb') as f:
    df = pickle.load(f)

# print(df['Label'].value_counts())

#plot number of recorded attacks
# make a plot number of labels
# sns.set(rc={'figure.figsize':(12, 6)})
# plt.xlabel('Attack Type')
# sns.set_theme()
# ax = sns.countplot(x='Label', data=df)
# ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
# plt.show()

#check for null values
#df.isna().sum().to_numpy()
print(bcolors.FAIL + "Number of null values before cleaning" + bcolors.ENDC)
print(df.isna().sum().sum())

#drop null values
df = df.dropna()
print(bcolors.FAIL + "Number of null values after cleaning" + bcolors.ENDC)
print(df.isna().sum().sum())

#print(df.info())

print(bcolors.OKBLUE + "Mapping attack types" + bcolors.ENDC)
# categorize attack type map
if(CLASSIFIER_TYPE == 'multi'):
    attack_group = {
        'Benign': 0,
        'DDOS attack-HOIC': 1,
        'DDOS attack-LOIC-UDP': 1,
        'DoS attacks-GoldenEye': 2,
        'DoS attacks-Hulk': 2,
        'DoS attacks-Slowloris': 2,
        'DoS attacks-SlowHTTPTest': 2,
        'FTP-BruteForce': 3,
        'SSH-Bruteforce': 3,
        'Brute Force -Web': 3,
        'Brute Force -XSS': 3,
        'SQL Injection': 4,
        'Infilteration': 4,
    }

if(CLASSIFIER_TYPE == 'binary'):
    attack_group = {
        'Benign': 0,
        'DDOS attack-HOIC': 1,
        'DDOS attack-LOIC-UDP': 1,
        'DoS attacks-GoldenEye': 1,
        'DoS attacks-Hulk': 1,
        'DoS attacks-Slowloris': 1,
        'DoS attacks-SlowHTTPTest': 1,
        'FTP-BruteForce': 1,
        'SSH-Bruteforce': 1,
        'Brute Force -Web': 1,
        'Brute Force -XSS': 1,
        'SQL Injection': 1,
        'Infilteration': 1,
    }

#Benign = 1
#DDoS = 2
#DoS = 3
#Bruteforce = 4
#Infilteration = 5

# Create grouped label column
df['Label'] = df['Label'].map(lambda x: attack_group[x])
# print(df['Label'].value_counts())

#check for infinite values
df = df.replace([np.inf, -np.inf], np.nan)
print(bcolors.FAIL + "__After inf cleaning - number of null values before cleaning" + bcolors.ENDC)
print(df.isna().sum().sum())

df = df.dropna()
print(bcolors.FAIL + "__After inf cleaning - number of null values after cleaning" + bcolors.ENDC)
print(df.isna().sum().sum())

print(bcolors.OKBLUE + "Re-sampling data" + bcolors.ENDC)
benign = df[df['Label'] == 0][:200000]
if(CLASSIFIER_TYPE == 'multi'):
    ddos = df[df['Label'] == 1][:200000]
    dos = df[df['Label'] == 2][:200000]
    bruteforce = df[df['Label'] == 3][:200000]
    infilteration = df[df['Label'] == 4][:200000]

    merge = [
        benign, ddos, dos, bruteforce, infilteration
    ]

if(CLASSIFIER_TYPE == 'binary'):
    attack = df[df['Label'] == 1][:200000]

    merge = [
        benign, attack
    ]
 
df = pd.concat(merge)
del merge
# print(df['Label'].value_counts())

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(df.drop('Label', axis=1), df['Label'])

#print(X_res.info())

sampled_df = X_res.insert(78, 'Label', y_res)
df_final = X_res.copy()
#print(df_final.info())

print(bcolors.OKBLUE + "Saving as pickle" + bcolors.ENDC)
name = '_optimized_' + CLASSIFIER_TYPE + '.pkl' 
with open(os.path.join(DATASET_DIR, DATASET_NAME, DATASET_NAME + name), 'wb') as f:
    pickle.dump(df_final, f)

print(bcolors.OKGREEN + "Done" + bcolors.ENDC)