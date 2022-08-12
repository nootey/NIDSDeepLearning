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
    model = keras.models.load_model(os.path.join(MODEL_DIR, DATASET_NAME + '_' + CLASSIFIER_TYPE + '_' + str(i)))
    
    X_test_val_name = CLASSIFIER_TYPE + str(i) + '_X_test.pkl'
    y_test_val_name = CLASSIFIER_TYPE + str(i) + '_y_test.pkl'
    with open(os.path.join(DATA_DIR, 'test', X_test_val_name), 'rb') as f:
        X_test = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'test', y_test_val_name), 'rb') as f:
        y_test = pickle.load(f)
    
    history = np.load(
        os.path.join(MODEL_HISTORY_DIR, DATASET_NAME + '_' + CLASSIFIER_TYPE +  '_' + str(i) + ".npy"),
        allow_pickle=True,
    )

    print('Evaluating')
    epochs = len(history.item()["accuracy"])-100
    kys = model.evaluate(X_test, y_test, verbose=VERBOSE)
    kys.append(epochs)
    evals.append(kys)

    df = pd.DataFrame(evals)
    df.rename(columns={0: "Loss", 1: "Accuracy", 2: "Epochs to learn"}, inplace=True)

    print(bcolors.OKBLUE + "Predict result" + bcolors.ENDC)

    # Measure model accuracy
    predictions = model.predict(
        x=X_test,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE
    )
    rounded_predictions = np.argmax(predictions,axis=1)

    print(bcolors.OKBLUE + "Plot confusion matrix" + bcolors.ENDC)
    # create the confusion matrix

    # attack_label = ['Benign', 'DDOS attack-HOIC', 'Bot', 'FTP-BruteForce', 'SSH-Bruteforce',
    # 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DDOS attack-LOIC-UDP', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection']
    if(CLASSIFIER_TYPE=='multi'): attack_label = ['Normal', 'DDOS', 'Dos', 'BruteForce', 'Infilteration']
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

    # print("--- Feature and iteration : ", model_name + str(iteration), "---")
    # # print(df, "\n")
    # # print(df.describe().loc[["min","max","mean","std"]])
    # pd.set_option("display.precision", 4)

    # with pd.ExcelWriter(BASE_FOLDER + "rezultati.xlsx", mode="a") as writer:
    #     df.to_excel(writer, sheet_name=model_name + str(iteration), startrow=0, startcol=0)
    #     df.describe().loc[["min", "max", "mean", "std"]].to_excel(writer, sheet_name=model_name + str(iteration), startrow=11, startcol=0)
    # print("------\n")

print(df)