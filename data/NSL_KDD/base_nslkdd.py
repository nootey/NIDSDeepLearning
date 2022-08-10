# Import modules
import numpy as np 
import pandas as pd 
import tensorflow as tf

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

#define paths
root_folder = "D:/School/diplomska_git/"
dataset_path = root_folder + 'datasets/'
dataset_name = "nsl-kdd/KDDTrain+.txt"
save_model_path = root_folder + 'saved_models/'

# fetch the training file
print(bcolors.WARNING + "Reading file" + bcolors.ENDC)
base_df = pd.read_csv(dataset_path+dataset_name)

# assign columns to the dataframe
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'])
base_df.columns = columns

# check initial df
base_df.head()

# define functions

# model configuration
def model_config(input_shape,output_shape):
    model = Sequential()
    # input layer
    model.add(Dense(10, input_dim=input_shape, activation='relu'))
    # hidden layer(s)
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # output layer
    model.add(Dense(output_shape,activation='softmax'))

    # Compile & Fit model
    model.compile (
    optimizer=Adam(learning_rate=0.0001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],
    )

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
    if attack in dos: type = 1
    elif attack in probe: type = 2
    elif attack in privilege: type = 3
    elif attack in remote_access: type = 4
    else: type = 0 
    return type

# categorize the possible attack types for a better prediction model

# define types of possible attacks
dos = ['apache2','back','land','neptune','mailbomb','pod','processtable','teardrop','udpstorm','worm']
probe = ['mscan','nmap','portsweep','saint','satan']
privilege = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm', 'ipsweep',]
remote_access = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop','smurf']
attack_label = ['normal', 'dos', 'probe', 'privilege', 'access']

# map attack types to df, based off attack label
attack_map = base_df.attack.apply(map_attack)
base_df['attack_map'] = attack_map

base_df.head()

# define arrays to split numeric/non-numeric data to pass to df
encode_non_numeric = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
encode_numeric = []  

# drop data about attacks
base_df.drop(columns=['attack'],inplace=True)

# get numeric items
for item in base_df.columns:
    if item not in encode_non_numeric and item != "attack_map": encode_numeric.append(item) 

print(bcolors.WARNING + "Normalizing data" + bcolors.ENDC)
# encode numeric df items as zscores
for column in encode_numeric:
    encode_numeric_zscore(base_df,column) 

# encode non numeric df items as dummie variables
for column in encode_non_numeric:
    encode_text_dummy(base_df,column)

# drop possible rows that are "NA" -> none in this dataset, but better safe than sorry
base_df.dropna(inplace=True,axis=1)

# check df
base_df.head()


#base_df.groupby('attack_map')['attack_map'].count()

print(bcolors.WARNING + "Prep X and y" + bcolors.ENDC)
# Convert to Numpy array
# values
X_columns = base_df.columns.drop('attack_map')
X = base_df[X_columns].values
# labels
y = pd.get_dummies(base_df['attack_map']).values

print(bcolors.WARNING + "Train test split" + bcolors.ENDC)
# Create a train/test split
# stratify makes sure that data is correctly proportionalized
X_train, X_val, y_train, y_val = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y) 
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# print(bcolors.WARNING + "Creating mirrored strategy" + bcolors.ENDC)
# # Create a MirroredStrategy.
# # strategy = tf.distribute.MirroredStrategy(["GPU:1","GPU:2","GPU:3"])
# strategy = tf.distribute.MirroredStrategy()
# # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# print(bcolors.WARNING + "Opening strategy scope" + bcolors.ENDC)
# # Open a strategy scope.
# with strategy.scope():
#     model = model_config(X.shape[1], y.shape[1])

print(bcolors.WARNING + "Training model" + bcolors.ENDC)
model = model_config(X.shape[1], y.shape[1])

# train model
history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1,
    batch_size=32, 
    epochs=15,
    callbacks=[model_early_stop()],
)

print(bcolors.WARNING + "Save model" + bcolors.ENDC)
model.save(save_model_path+dataset_name)

print(bcolors.WARNING + "Load model" + bcolors.ENDC)
model = keras.models.load_model(save_model_path+dataset_name)

print(bcolors.WARNING + "Predict result" + bcolors.ENDC)
# Measure model accuracy
predictions = model.predict(
    x=X_val,
    batch_size=32,
    verbose=1
)
rounded_predictions = np.argmax(predictions,axis=1)

print(bcolors.WARNING + "Plot confusion matrix" + bcolors.ENDC)
# create the confusion matrix

con_mat = tf.math.confusion_matrix(labels=y_val.argmax(axis=1), predictions=rounded_predictions).numpy()

con_mat_norm = np.around(con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=4)

con_mat_df = pd.DataFrame(con_mat_norm, index=attack_label, columns=attack_label)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.get_cmap("Reds"))
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()  