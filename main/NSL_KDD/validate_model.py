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
import tensorflow as tf
from tensorflow import keras


print(bcolors.OKBLUE + "Loading test data" + bcolors.ENDC)
with open(os.path.join(DATA_DIR_NSL, 'test', X_test_name), 'rb') as f:
    X_test = pickle.load(f)

with open(os.path.join(DATA_DIR_NSL, 'test', y_test_name), 'rb') as f:
    y_test = pickle.load(f)

print(y_test.shape)
print(bcolors.WARNING + "Load model" + bcolors.ENDC)
model = keras.models.load_model(os.path.join(MODEL_DIR, DATASET_NAME_NSL, '_', CLASSIFIER_TYPE))

print(bcolors.WARNING + "Predict result" + bcolors.ENDC)
# Measure model accuracy
predictions = model.predict(
    x=X_test,
    batch_size=BATCH_SIZE_NSL,
    verbose=1
)
rounded_predictions = np.argmax(predictions,axis=1)

print(bcolors.WARNING + "Plot confusion matrix" + bcolors.ENDC)
# create the confusion matrix

if(CLASSIFIER_TYPE=='multi'): attack_label = ['Normal', 'Dos', 'Probe', 'U2R', 'R2L']
if(CLASSIFIER_TYPE=='binary'): attack_label = ['Normal', 'Attack']

con_mat = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=rounded_predictions).numpy()

con_mat_norm = np.around(con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=4)

con_mat_df = pd.DataFrame(con_mat_norm, index=attack_label, columns=attack_label)

if(CLASSIFIER_TYPE=='multi'): color = 'Blues'
if(CLASSIFIER_TYPE=='binary'): color = 'Purples'

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.get_cmap(color))
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()  