import ssl
import sys
import os
sys.path.append('../..')
from vardata import *

import pickle
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score

def plot_history(history):

    plt.plot(history.item()['accuracy'], color="blue")
    plt.plot(history.item()['val_accuracy'], color="purple")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(RESULT_DIR, 'history', GROUP_TYPE, CLASSIFIER_TYPE, 'accuracy_e_' + str(NUM_EPOCHS) + '_b_' + str(BATCH_SIZE)  + '.png'))
    plt.clf()

    plt.plot(history.item()['loss'], color="green")
    plt.plot(history.item()['val_loss'], color="red")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(RESULT_DIR, 'history', GROUP_TYPE, CLASSIFIER_TYPE, 'loss_e_' + str(NUM_EPOCHS) + '_b_' + str(BATCH_SIZE)  + '.png'))

history = np.load(
    os.path.join(DATA_DIR, 'history', CLASSIFIER_TYPE + '_e_' + str(NUM_EPOCHS) + '_b_' + str(BATCH_SIZE) + '.npy'),
    allow_pickle=True,
    )

plot_history(history)

print(bcolors.OKBLUE + "Loading test data" + bcolors.ENDC)
with open(os.path.join(DATA_DIR, 'test', X_test_name), 'rb') as f:
    X_test = pickle.load(f)

with open(os.path.join(DATA_DIR, 'test', y_test_name), 'rb') as f:
    y_test = pickle.load(f)

print(bcolors.OKBLUE + "Load model" + bcolors.ENDC)
model = keras.models.load_model(os.path.join(MODEL_DIR, GROUP_TYPE, DATASET_NAME, CLASSIFIER_TYPE + '_e_' + str(NUM_EPOCHS) + '_b_' + str(BATCH_SIZE)))

print(bcolors.OKBLUE + "Predict result" + bcolors.ENDC)
# Measure model accuracy
predictions = model.predict(
    x=X_test,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE
)
rounded_predictions = np.argmax(predictions,axis=1)

print(bcolors.OKBLUE + "Save confusion matrix" + bcolors.ENDC)
# create the confusion matrix

if(CLASSIFIER_TYPE=='multi'): 
    if(GROUP_TYPE == 'grouped'):
        attack_label = ['Normal', 'DDOS', 'Dos', 'BruteForce', 'Infilteration']
    else:
        attack_label = ['Normal', 'FTP-BruteForce', 'SSH-BruteForce', 'DDOS-HOIC', 'Infilteration', 'DoS-GoldenEye', 'DoS-Slowloris', 'DDOS-LOIC-UDP', 'BruteForce-Web', 'BruteForce-XSS', 'SQL-Injection']
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
plt.savefig(os.path.join(RESULT_DIR, 'prediction', GROUP_TYPE, CLASSIFIER_TYPE, 'e_' + str(NUM_EPOCHS) + '_b_' + str(BATCH_SIZE)  + '.png'))

