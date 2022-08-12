import sys
import os
sys.path.append('../..')
from vardata import *

import pickle
import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical, normalize
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

# optimized, but limited dataset
# print(bcolors.OKBLUE + "Reading file" + bcolors.ENDC)
# dataset is already cleaned of nan and infinite values and is scaled to 20%
#df = pd.read_csv(os.path.join(DATASET_DIR, DATASET_NAME))

print(bcolors.OKBLUE + "Loading pickle" + bcolors.ENDC)
with open(OPTIMIZED_DATASET_PATH, 'rb') as f:
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
            model.compile(optimizer=OPTIMIZER,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        else:
            print(bcolors.OKBLUE + 'Binary Cross-Entropy Loss Function' + bcolors.ENDC)
            model.compile(optimizer=OPTIMIZER,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model

print(bcolors.OKBLUE + "Prep features and labels" + bcolors.ENDC)
features = df
labels = features.pop('Label')

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)
dummy_labels = to_categorical(labels)
normalized_features = normalize(features.values)

y = labels.copy()
X = normalized_features.copy()

inputDim = len(normalized_features[0])

print(bcolors.OKBLUE + "Split train/test data" + bcolors.ENDC)
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
# for train_index, test_index in sss.split(X=np.zeros(normalized_features.shape[0]), y=dummy_labels):
#     X_train, X_test = normalized_features[train_index], normalized_features[test_index]
#     y_train, y_test = dummy_labels[train_index], dummy_labels[test_index]


# X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.1, random_state=1)

predictions = []

model = model_config(inputDim, dummy_labels.shape)

skf = StratifiedKFold(n_splits = NUM_FOLDS)
# skf.get_n_splits(normalized_features, dummy_labels)
skf.get_n_splits(normalized_features, labels)

kFold_index = 0

for train_index, test_index in skf.split(X, y):
    print('Train: ', train_index, 'Validation: ', test_index)
    X1_train, X1_test = X[train_index], X[test_index]
    y1_train, y1_test = y[train_index], y[test_index]

    # SKF SPLIT ACCEPTS 1D ARRAY, BUT MODEL FIT NEEDS 2D ARRAY FOR Y
    y_dum = to_categorical(y1_train)
    y_dum_test = to_categorical(y1_test)

    
    # train model
    #model.fit(x=X1_train, y=y_dum, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, validation_data=(X1_test, y_dum_test))
    
    history = model.fit(
        X1_train,
        y_dum,
        verbose=VERBOSE,
        validation_data=(X1_test, y_dum_test),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
    )

    print(bcolors.OKBLUE + "Save model" + bcolors.ENDC)
    model.save(os.path.join(MODEL_DIR, DATASET_NAME + '_' + CLASSIFIER_TYPE + '_' + str(kFold_index)))
    np.save(os.path.join(MODEL_HISTORY_DIR, DATASET_NAME + '_' + CLASSIFIER_TYPE +  '_' + str(kFold_index) + ".npy"), history.history)
    
    print(bcolors.OKBLUE + "Saving test data" + bcolors.ENDC)
    X_test_val_name = CLASSIFIER_TYPE + str(kFold_index) + '_X_test.pkl'
    y_test_val_name = CLASSIFIER_TYPE + str(kFold_index) + '_y_test.pkl'
    with open(os.path.join(DATA_DIR, 'test', X_test_val_name), 'wb') as f:
        pickle.dump(X1_test, f)

    with open(os.path.join(DATA_DIR, 'test', y_test_val_name), 'wb') as f:
        pickle.dump(y_dum_test, f)

    kFold_index += 1
