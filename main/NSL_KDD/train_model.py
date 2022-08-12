import sys
import os
sys.path.append('../..')
from vardata import *

# Import modules
import numpy as np 
import pandas as pd 
import tensorflow as tf
import pickle

# Import processing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Import plotting
import seaborn as sns
import matplotlib.pyplot as plt

# helper functions
def model_config(input_shape,output_shape):
    model = Sequential()
    model.add(Dense(42, input_dim=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape,activation='softmax'))

    if CLASSIFIER_TYPE == 'multi':
        print(bcolors.OKBLUE + 'Categorical Cross-Entropy Loss Function' + bcolors.ENDC)
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    if CLASSIFIER_TYPE == 'binary':
        print(bcolors.OKBLUE + 'Binary Cross-Entropy Loss Function' + bcolors.ENDC)
        model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model

# model stopper due to over/under fitting
def model_early_stop():
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=VERBOSE,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    return early_stopper

def normalize_features(df, name, mean=None, standard_deviation=None):
    
    # define mean
    if mean is None:
        mean = df[name].mean()
    
    # define standarad deviation
    if standard_deviation is None:
        standard_deviation = df[name].std()

    # calculate zscore
    df[name] = (df[name] - mean) / standard_deviation
    
def encode_labels(df, name):
    # use built in pandas function for converting to dummiy
    dummies = pd.get_dummies(df[name])
    # extract values from entities and encode them 
    for value in dummies.columns:
        label = f"{name}-{value}"
        df[label] = dummies[value]
    df.drop(name, axis=1, inplace=True)

# classify attacks as possible attack types
def classifiy_attacks (attack):
    if(CLASSIFIER_TYPE == 'multi'):
        if attack in dos: att_type = 1
        elif attack in probe: att_type = 2
        elif attack in user_to_root: att_type = 3
        elif attack in root_to_local: att_type = 4
        else: att_type = 0 
        
    if(CLASSIFIER_TYPE == 'binary'):
        if attack in dos: att_type = 1
        elif attack in probe: att_type = 1
        elif attack in user_to_root: att_type = 1
        elif attack in root_to_local: att_type = 1
        else: att_type = 0 
    
    return att_type

# fetch the training file
print(bcolors.WARNING + "Reading file" + bcolors.ENDC)
df_train = pd.read_csv(os.path.join(DATASET_DIR, DATASET_NAME_NSL, 'KDDTrain+.txt'))
df_test = pd.read_csv(os.path.join(DATASET_DIR, DATASET_NAME_NSL, 'KDDTest+.txt'))

# define columns
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'attack', 'difficulty']

df_train.columns = columns
df_test.columns = columns

# drop columns that aren't needed
df_train = df_train.drop('difficulty', axis=1)
df_test = df_test.drop('difficulty', axis=1)

# check for null values
# print(df_train.isnull().values.any())
# print(df_test.isnull().values.any())

#check for data types
# print(df_train.info())

base_df = pd.concat([df_train, df_test])

print(bcolors.WARNING + "Mapping attack types" + bcolors.ENDC)
# define types of possible attacks
dos = ["apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm"]
probe = ["ipsweep","mscan","nmap","portsweep","saint","satan"]
user_to_root = ["buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm"]
root_to_local = ["ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop"]

# map attack types to df, based off attack label
attack_classifier = base_df.attack.apply(classifiy_attacks)
base_df['label'] = attack_classifier

# print(base_df.head)

# define arrays to split numeric/non-numeric data to pass to df
encode_non_numeric = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
encode_numeric = []  

# drop data about attacks
base_df.drop(columns=['attack'],inplace=True)

# get numeric items
for item in base_df.columns:
    if item not in encode_non_numeric and item != "label": encode_numeric.append(item) 

print(bcolors.WARNING + "Normalizing data" + bcolors.ENDC)
# encode numeric df items as zscores
for column in encode_numeric:
    normalize_features(base_df,column) 

# encode non numeric df items as dummie variables
for column in encode_non_numeric:
    encode_labels(base_df,column)

# drop possible rows that are "NA" -> none in this dataset, but better safe than sorry
base_df.dropna(inplace=True,axis=1)


#print(base_df.groupby('label')['label'].count())

print(bcolors.WARNING + "Prep X and y" + bcolors.ENDC)
# Convert to Numpy array
# values
X_columns = base_df.columns.drop('label')
X = base_df[X_columns].values
# labels
y = pd.get_dummies(base_df['label']).values

print(bcolors.WARNING + "Train test split" + bcolors.ENDC)
# Create a train/test split
# stratify makes sure that data is correctly proportionalized
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y) 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(bcolors.OKBLUE + "Saving test data" + bcolors.ENDC)
with open(os.path.join(DATA_DIR_NSL, 'test', X_test_name), 'wb') as f:
    pickle.dump(X_test, f)

with open(os.path.join(DATA_DIR_NSL, 'test', y_test_name), 'wb') as f:
    pickle.dump(y_test, f)

print(bcolors.WARNING + "Fit model" + bcolors.ENDC)
model = model_config(X.shape[1], y.shape[1])

# train model
history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_test, y_test),
    shuffle=True,
    verbose=VERBOSE,
    batch_size=BATCH_SIZE_NSL, 
    epochs=NUM_EPOCHS_NSL,
    callbacks=[model_early_stop()],
)

print(bcolors.WARNING + "Save model" + bcolors.ENDC)
model.save(os.path.join(MODEL_DIR, DATASET_NAME_NSL, '_', CLASSIFIER_TYPE))

