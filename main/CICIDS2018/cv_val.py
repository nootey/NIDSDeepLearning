import sys
import os
sys.path.append('../..')
from vardata import *

import pickle
import numpy as np 
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

evals = list()

for i in range(NUM_FOLDS):
    model = keras.models.load_model(os.path.join(MODEL_DIR, DATASET_NAME + '_' + CLASSIFIER_TYPE + '_' + str(GROUP_TYPE) + '_' + str(i)))
    
    X_test_val_name = CLASSIFIER_TYPE + str(GROUP_TYPE) + '_' + str(i) + '_X_test.pkl'
    y_test_val_name = CLASSIFIER_TYPE + str(GROUP_TYPE) + '_' + str(i) + '_y_test.pkl'
    with open(os.path.join(DATA_DIR, 'test', X_test_val_name), 'rb') as f:
        X_test = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'test', y_test_val_name), 'rb') as f:
        y_test = pickle.load(f)
    
    history = np.load(
        os.path.join(MODEL_HISTORY_DIR, DATASET_NAME + '_' + CLASSIFIER_TYPE + '_' + str(GROUP_TYPE) + '_' + str(i) + ".npy"),
        allow_pickle=True,
    )

    print(bcolors.OKBLUE + "Evaluating model: " + str(i) + bcolors.ENDC)
    epochs = len(history.item()["accuracy"])
    single_eval = model.evaluate(X_test, y_test, verbose=VERBOSE)
    single_eval.append(epochs)
    evals.append(single_eval)

    df = pd.DataFrame(evals)
    df.rename(columns={0: "Loss", 1: "Accuracy", 2: "Epochs to learn"}, inplace=True)

    print(bcolors.OKBLUE + "Predicting results based of model: " + str(i) + bcolors.ENDC)

    # Measure model accuracy
    predictions = model.predict(
        x=X_test,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE
    )
    rounded_predictions = np.argmax(predictions,axis=1)

    print(bcolors.OKBLUE + "Save confusion matrix" + bcolors.ENDC)

    if(CLASSIFIER_TYPE=='multi'): 
        if(GROUP_TYPE == 1):
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
    
    sub_folder = 'epochs' + str(NUM_EPOCHS) + '_batchsize' + str(BATCH_SIZE) + '_' + CLASSIFIER_TYPE + str(i) + '_grouping' + '_' + str(GROUP_TYPE)
    # os.mkdir(sub_folder)
    plt.savefig(os.path.join(RESULT_DIR, sub_folder  + '.png'))
    # plt.show() 
    SHEET_NAME = DATASET_NAME + '_' + str(i)
    with pd.ExcelWriter(os.path.join(RESULT_DIR, 'results.xlsx'), mode="a", if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=DATASET_NAME + str(i), startrow=0, startcol=0)
        df.describe().loc[["min", "max", "mean", "std"]].to_excel(writer, sheet_name=SHEET_NAME, startrow=11, startcol=0)
    print("------\n")

print(df)