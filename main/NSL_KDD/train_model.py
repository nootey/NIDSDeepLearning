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
DATA_DIR = ROOT_DIR + 'data/'
type = 'binary'

# helper functions
def model_config(input_shape,output_shape):
    model = Sequential()
    model.add(Dense(42, input_dim=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape,activation='softmax'))

    if type == 'multi':
        print(bcolors.OKBLUE + 'Categorical Cross-Entropy Loss Function' + bcolors.ENDC)
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    if type == 'binary':
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
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    return early_stopper

# encode a numeric column as a zscore
def encode_numeric_zscore(df, name, mean=None, standard_deviation=None):
    
    # define mean
    if mean is None:
        mean = df[name].mean()
    
    # define standarad deviation
    if standard_deviation is None:
        standard_deviation = df[name].std()

    # calculate zscore
    df[name] = (df[name] - mean) / standard_deviation
    
# encode text values as dummy variables (RGB = [1,0,0], [0,1,0], [0,0,1])
def encode_text_dummy(df, name):
    # use built in pandas function for converting to dummiy
    dummies = pd.get_dummies(df[name])
    # extract values from entities and encode them 
    for value in dummies.columns:
        label = f"{name}-{value}"
        df[label] = dummies[value]
    df.drop(name, axis=1, inplace=True)

# classify attacks as possible attack types
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

# fetch the training file
print(bcolors.WARNING + "Reading file" + bcolors.ENDC)
df_train = pd.read_csv(DATASET_DIR + DATASET_NAME + '/KDDTrain+.txt')
df_test = pd.read_csv(DATASET_DIR + DATASET_NAME + '/KDDTest+.txt')

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
attack_map = base_df.attack.apply(map_attack)
base_df['label'] = attack_map

base_df.head()

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
    encode_numeric_zscore(base_df,column) 

# encode non numeric df items as dummie variables
for column in encode_non_numeric:
    encode_text_dummy(base_df,column)

# drop possible rows that are "NA" -> none in this dataset, but better safe than sorry
base_df.dropna(inplace=True,axis=1)


#base_df.groupby('label')['label'].count()

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
X_train, X_val, y_train, y_val = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y) 
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

X_test_name = type + '_X_test.pkl'
y_test_name = type + '_y_test.pkl'
print(bcolors.OKBLUE + "Saving test data" + bcolors.ENDC)
with open(DATA_DIR + DATASET_NAME + '/' + X_test_name, 'wb') as f:
    pickle.dump(X_val, f)

with open(DATA_DIR + DATASET_NAME + '/' + y_test_name, 'wb') as f:
    pickle.dump(y_val, f)

print(bcolors.WARNING + "Fit model" + bcolors.ENDC)
model = model_config(X.shape[1], y.shape[1])

# train model
history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1,
    batch_size=32, 
    epochs=25,
    callbacks=[model_early_stop()],
)

print(bcolors.WARNING + "Save model" + bcolors.ENDC)
model.save(MODEL_DIR + '/' + DATASET_NAME + '_' + type)

