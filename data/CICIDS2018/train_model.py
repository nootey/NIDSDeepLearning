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

# DATASET_DIR = os.path.join(os.path.abspath(".."), "datasets/CICIDS2018")
# MODEL_DIR = os.path.join(os.path.abspath(".."), "model/saved_models")
# DATASET_NAME = 'cleaned_ids2018_sampled.csv'

ROOT_DIR = "D:/School/diplomska_ml/"
DATASET_DIR = ROOT_DIR + 'datasets/'
MODEL_DIR = ROOT_DIR + 'saved_models/'
DATA_DIR = ROOT_DIR + 'data/'
DATASET_NAME = 'CICIDS2018'
PICKLE_DIR = DATASET_DIR + DATASET_NAME + '/'
PICKLE_DIR = DATASET_DIR + DATASET_NAME + '/'
type = 'multi'
name = '_optimized_' + type + '.pkl' 

# optimized, but limited dataset
# print(bcolors.OKBLUE + "Reading file" + bcolors.ENDC)
# dataset is already cleaned of nan and infinite values and is scaled to 20%
#df = pd.read_csv(os.path.join(DATASET_DIR, DATASET_NAME))

print(bcolors.OKBLUE + "Loading pickle" + bcolors.ENDC)
with open(PICKLE_DIR + DATASET_NAME + name, 'rb') as f:
    df = pickle.load(f)

#helper functions 
def model_config(inputDim=-1, out_shape=(-1,)):
    model = Sequential()
    if inputDim > 0 and out_shape[1] > 0:
        model.add(Dense(79, activation='relu', input_shape=(inputDim,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(out_shape[1], activation='softmax')) 
        
        if out_shape[1] > 2:
            print(bcolors.OKBLUE + 'Categorical Cross-Entropy Loss Function' + bcolors.ENDC)
            model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        else:
            print(bcolors.OKBLUE + 'Binary Cross-Entropy Loss Function' + bcolors.ENDC)
            model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model

optimizer='adam'
epochs=25
batch_size=10

print(bcolors.OKBLUE + "Prep features and labels" + bcolors.ENDC)
features = df
labels = features.pop('Label')

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)
dummy_labels = to_categorical(labels)
normalized_features = normalize(features.values)

inputDim = len(normalized_features[0])

print(bcolors.OKBLUE + "Split train/test data" + bcolors.ENDC)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
for train_index, test_index in sss.split(X=np.zeros(normalized_features.shape[0]), y=dummy_labels):
    X_train, X_test = normalized_features[train_index], normalized_features[test_index]
    y_train, y_test = dummy_labels[train_index], dummy_labels[test_index]

X_test_name = type + '_X_test.pkl'
y_test_name = type + '_y_test.pkl'
print(bcolors.OKBLUE + "Saving test data" + bcolors.ENDC)
with open(DATA_DIR + DATASET_NAME + '/' + X_test_name, 'wb') as f:
    pickle.dump(X_test, f)

with open(DATA_DIR + DATASET_NAME + '/' + y_test_name, 'wb') as f:
    pickle.dump(y_test, f)

print(bcolors.OKBLUE + "Fit model" + bcolors.ENDC)
model = model_config(inputDim, y_train.shape)

model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))

print(bcolors.OKBLUE + "Save model" + bcolors.ENDC)
model.save(MODEL_DIR + '/' + DATASET_NAME + '_' + type)
