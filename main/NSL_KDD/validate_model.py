import pickle
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical, normalize
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard
from timeit import default_timer as timer
from tensorflow import keras

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


ROOT_DIR = "D:/School/diplomska_ml/"
DATASET_DIR = ROOT_DIR + 'datasets/'
MODEL_DIR = ROOT_DIR + 'saved_models/'
DATA_DIR = ROOT_DIR + 'data/'
DATASET_NAME = 'NSL_KDD'
PICKLE_DIR = DATASET_DIR + DATASET_NAME + '/'
PICKLE_DIR = DATASET_DIR + DATASET_NAME + '/'

type = 'binary'

X_test_name = type + '_X_test.pkl'
y_test_name = type + '_y_test.pkl'

print(bcolors.OKBLUE + "Loading test data" + bcolors.ENDC)
with open(DATA_DIR + DATASET_NAME + '/' + X_test_name, 'rb') as f:
    X_test = pickle.load(f)

with open(DATA_DIR + DATASET_NAME + '/' + y_test_name, 'rb') as f:
    y_test = pickle.load(f)

print(y_test.shape)
print(bcolors.WARNING + "Load model" + bcolors.ENDC)
model = keras.models.load_model(MODEL_DIR + '/' + DATASET_NAME + '_' + type)

print(bcolors.WARNING + "Predict result" + bcolors.ENDC)
# Measure model accuracy
predictions = model.predict(
    x=X_test,
    batch_size=32,
    verbose=1
)
rounded_predictions = np.argmax(predictions,axis=1)

print(bcolors.WARNING + "Plot confusion matrix" + bcolors.ENDC)
# create the confusion matrix

if(type=='multi'): attack_label = ['Normal', 'Dos', 'Probe', 'U2R', 'R2L']
if(type=='binary'): attack_label = ['Normal', 'Attack']

con_mat = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=rounded_predictions).numpy()

con_mat_norm = np.around(con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=4)

con_mat_df = pd.DataFrame(con_mat_norm, index=attack_label, columns=attack_label)

if(type=='multi'): color = 'Blues'
if(type=='binary'): color = 'Purples'

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.get_cmap(color))
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()  