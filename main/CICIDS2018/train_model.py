import sys
import os
sys.path.append('../..')
from vardata import *

import pickle
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical, normalize
from tensorflow import keras

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

inputDim = len(normalized_features[0])

print(bcolors.OKBLUE + "Split train/test data" + bcolors.ENDC)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
for train_index, test_index in sss.split(X=np.zeros(normalized_features.shape[0]), y=dummy_labels):
    X_train, X_test = normalized_features[train_index], normalized_features[test_index]
    y_train, y_test = dummy_labels[train_index], dummy_labels[test_index]


# print(bcolors.OKBLUE + "Saving test data" + bcolors.ENDC)
# with open(os.path.join(DATA_DIR, 'test', X_test_name), 'wb') as f:
#     pickle.dump(X_test, f)

# with open(os.path.join(DATA_DIR, 'test', y_test_name), 'wb') as f:
#     pickle.dump(y_test, f)

print('REEEE', y_train.shape)

print(bcolors.OKBLUE + "Fit model" + bcolors.ENDC)
model = model_config(inputDim, y_train.shape)

model.fit(x=X_train, y=y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, validation_data=(X_test, y_test))

print(bcolors.OKBLUE + "Save model" + bcolors.ENDC)
model.save(os.path.join(MODEL_DIR, DATASET_NAME, '_', CLASSIFIER_TYPE))
