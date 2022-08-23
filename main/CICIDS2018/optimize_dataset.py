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

print(bcolors.OKBLUE + "Loading pickle df" + bcolors.ENDC)
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

# print(df.info())

print(bcolors.OKBLUE + "Mapping attack types" + bcolors.ENDC)
# categorize attack type map
if(CLASSIFIER_TYPE == 'multi'):
    if(GROUP_TYPE == 1):
        attack_group = {
            'Benign': 0,
            'DDOS attack-HOIC': 1,
            'DDOS attack-LOIC-UDP': 1,
            'DoS attacks-GoldenEye': 2,
            'DoS attacks-Slowloris': 2,
            'DoS attacks-SlowHTTPTest': 2,
            'FTP-BruteForce': 3,
            'SSH-Bruteforce': 3,
            'Brute Force -Web': 3,
            'Brute Force -XSS': 3,
            'SQL Injection': 4,
            'Infilteration': 4,
        }
    else:
        attack_group = {
            'Benign': 1,
            'FTP-BruteForce': 2,
            'SSH-Bruteforce': 3,
            'DDOS attack-HOIC': 4,
            'Infilteration': 5,
            'DoS attacks-GoldenEye' : 6,
            'DoS attacks-Slowloris': 7,
            'DDOS attack-LOIC-UDP': 8,
            'Brute Force -Web': 9,
            'Brute Force -XSS': 10,
            'SQL Injection': 11
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

# #Benign = 1
# #DDoS = 2
# #DoS = 3
# #Bruteforce = 4
# #Infilteration = 5

# Create grouped label column
df['Label'] = df['Label'].map(lambda x: attack_group[x])
print(df['Label'].value_counts())

#check for infinite values
df = df.replace([np.inf, -np.inf], np.nan)
print(bcolors.FAIL + "__After inf cleaning - number of null values before cleaning" + bcolors.ENDC)
print(df.isna().sum().sum())

df = df.dropna()
print(bcolors.FAIL + "__After inf cleaning - number of null values after cleaning" + bcolors.ENDC)
print(df.isna().sum().sum())

print(bcolors.OKBLUE + "Re-sampling data" + bcolors.ENDC)

benign = df[df['Label'] == 0][:100000]

if(CLASSIFIER_TYPE == 'multi'):

    if(GROUP_TYPE == 1):
        ddos = df[df['Label'] == 1][:100000]
        dos = df[df['Label'] == 2]
        bruteforce = df[df['Label'] == 3][:100000]
        infilteration = df[df['Label'] == 4]
        merge = [
        benign, ddos, dos, bruteforce, infilteration
                ]
    else:
        benign = df[df['Label'] == 1][:100000]
        hoic = df[df['Label'] == 2][:100000]
        bot = df[df['Label'] == 3][:100000]
        ftp_brute = df[df['Label'] == 4][:100000]
        ssh_brute = df[df['Label'] == 5][:100000]
        dos_eye = df[df['Label'] == 6]
        dos_slow = df[df['Label'] == 7]
        ddos = df[df['Label'] == 8]
        brute_web = df[df['Label'] == 9]
        brute_xss = df[df['Label'] == 10]
        sql = df[df['Label'] == 11]
        merge = [
            benign, hoic, bot, ftp_brute, ssh_brute, dos_eye, dos_slow, ddos, brute_web, brute_xss, sql
        ]


if(CLASSIFIER_TYPE == 'binary'):
    attacks = df[df['Label'] == 1][:150000]

    merge = [
        benign, attacks
    ]
 
df = pd.concat(merge)
del merge
print(df['Label'].value_counts())

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(df.drop('Label', axis=1), df['Label'])

print(X_res.info())

sampled_df = X_res.insert(77, 'Label', y_res)
df_final = X_res.copy()
#print(df_final.info())

print(bcolors.OKBLUE + "Saving optimized dataset" + bcolors.ENDC)
# name = '_optimized_' + CLASSIFIER_TYPE + '.pkl' 
# with open(os.path.join(DATASET_DIR, DATASET_NAME, DATASET_NAME + name), 'wb') as f:
#     pickle.dump(df_final, f)
name = '_g_'+ str(GROUP_TYPE) + '_optimized_' + CLASSIFIER_TYPE + '.csv' 
df_final.to_csv(os.path.join(DATASET_DIR, DATASET_NAME, DATASET_NAME + name))

print(bcolors.OKGREEN + "Done" + bcolors.ENDC)