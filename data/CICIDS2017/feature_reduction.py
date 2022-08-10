import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

data_dir  = os.path.join(os.path.abspath(".."), "CICIDS2017")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(bcolors.WARNING + "Loading pickle" + bcolors.ENDC)
X_train = pd.read_pickle(os.path.join(data_dir, 'processed', 'train_features.pkl'))
X_val = pd.read_pickle(os.path.join(data_dir, 'processed', 'val_features.pkl'))
X_test = pd.read_pickle(os.path.join(data_dir, 'processed', 'test_features.pkl'))

y_train = pd.read_pickle(os.path.join(data_dir, 'processed', 'train_labels.pkl'))
y_val = pd.read_pickle(os.path.join(data_dir, 'processed', 'val_labels.pkl'))
y_test = pd.read_pickle(os.path.join(data_dir, 'processed', 'test_labels.pkl'))

print(bcolors.WARNING + "Calculating PCA" + bcolors.ENDC)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X_train)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, y_train], axis=1)

pca = PCA()

pca.fit(X_train)

pca = PCA(0.99)

X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
X_val_pca = pd.DataFrame(pca.transform(X_val))
X_test_pca = pd.DataFrame(pca.transform(X_test))

print(bcolors.WARNING + "Saving pickle" + bcolors.ENDC)
X_train_pca.to_pickle(os.path.join(data_dir, 'processed', 'train_features_pca25.pkl'))
X_val_pca.to_pickle(os.path.join(data_dir, 'processed', 'val_features_pca25.pkl'))
X_test_pca.to_pickle(os.path.join(data_dir, 'processed', 'test_features_pca25.pkl'))