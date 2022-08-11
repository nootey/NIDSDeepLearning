# imports
import pandas as pd
import numpy as np
import sys
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.utils.np_utils import to_categorical
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from tensorflow.keras.utils import get_file, plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

# define colors (for readable prints)
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

# define paths
ROOT_DIR = "D:/School/diplomska_ml/"
DATASET_DIR = ROOT_DIR + 'datasets/'
MODEL_DIR = ROOT_DIR + 'saved_models/'
DATASET_NAME = 'NSL_KDD'
PICKLE_DIR = DATASET_DIR + DATASET_NAME + '/'

type = 'multi'

# read csv
print(bcolors.OKBLUE + "Reading files" + bcolors.ENDC)
df_train = pd.read_csv(DATASET_DIR + DATASET_NAME + '/KDDTrain+.txt')
df_test = pd.read_csv(DATASET_DIR + DATASET_NAME + '/KDDTest+.txt')

# helper functions
def model_config():
    if(type == 'multi'): loss = "categorical_crossentropy"
    if(type == 'binary'): loss = "binary_crossentropy"

    model = Sequential()
    model.add(Convolution1D(64, kernel_size=122, padding="same",activation="relu", input_shape=(122, 1)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False))) 
    model.add(Reshape((128, 1), input_shape = (128, )))
    
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False))) 
    
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss=loss,optimizer='adam',metrics=['accuracy'])
    return model

def model_early_stop():
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    return early_stopper

def normalize_features(df, columns):

    df_copy = df.copy() 
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            df_copy[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df_copy
    
def encode_labels(df, columns):

    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, 1)
    return df

def map_attack (attack):
    if(type == 'multi'):
        if attack in dos: att_type = 1
        elif attack in probe: att_type = 2
        elif attack in user_to_root: att_type = 3
        elif attack in root_to_local: att_type = 4
        else: att_type = 0 
        
    if(type == 'binary'):
        if attack in dos: att_type = 1
        elif attack in probe: att_type = 1
        elif attack in user_to_root: att_type = 1
        elif attack in root_to_local: att_type = 1
        else: att_type = 0 
    
    return att_type

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
'dst_host_srv_rerror_rate', 'attack_type', 'difficulty']

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

categorical_columns = ['protocol_type','service','flag']

print(bcolors.OKBLUE + "Prep features and labels" + bcolors.ENDC)
# one hot encode labels
df_x = pd.concat([df_train, df_test])
df_copy = df_x.copy()

df_x = encode_labels(df_x, categorical_columns)

df_y = df_x.pop('attack_type')

X_train = normalize_features(df_x, df_x.columns)

print(bcolors.OKBLUE + "Mapping attack types" + bcolors.ENDC)
# categorize attack type map
dos = ("apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm")
probe = ("ipsweep","mscan","nmap","portsweep","saint","satan")
user_to_root = ("buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm")
root_to_local = ("ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop")

# map attack types to df, based off attack label
attack_map = df_copy.attack_type.apply(map_attack)
X_train['attack_map'] = attack_map

print(X_train['attack_map'].value_counts())

# check for null values
# print(df_train.isnull().values.any())
# print(df_test.isnull().values.any())

# y_train = X_train['attack_map']
# X_train_comb = X_train.drop('attack_map', 1)

y = X_train['attack_map']
X = X_train.drop('attack_map', 1)

kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
kfold.get_n_splits(X,y)

predictions = []

model = model_config()

for train_index, test_index in kfold.split(X, y):
    train_X, test_X = X.iloc[train_index], X.iloc[test_index]
    train_y, test_y = y.iloc[train_index], y.iloc[test_index]
    
    print("train index:",train_index)
    print("test index:",test_index)

    x_columns_train = X_train.columns.drop('attack_map')
    x_train_array = train_X[x_columns_train].values
    x_train_1=np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))
    
    dummies = pd.get_dummies(train_y) 
    outcomes = dummies.columns
    num_classes = len(outcomes)
    y_train_1 = dummies.values
    
    x_columns_test = X_train.columns.drop('attack_map')
    x_test_array = test_X[x_columns_test].values
    x_test_2=np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))
    
    dummies_test = pd.get_dummies(test_y) 
    outcomes_test = dummies_test.columns
    num_classes = len(outcomes_test)
    y_test_2 = dummies_test.values
    
    model.fit(x_train_1, y_train_1,validation_data=(x_test_2,y_test_2), epochs=10)
    
    pred = model.predict(x_test_2)
    pred = np.argmax(pred,axis=1)
    y_eval = np.argmax(y_test_2,axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    predictions.append(score)
    print("Validation score: {}".format(score))

print(predictions)

print(bcolors.OKBLUE + "Save model" + bcolors.ENDC)
model.save(MODEL_DIR + '/' + DATASET_NAME + '_' + type)