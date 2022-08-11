import pickle
# Import modules
import numpy as np 
import pandas as pd 
import tensorflow as tf

# Import processing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape
from tensorflow.keras.models import Sequential

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

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
root_folder = "D:/School/diplomska_git/"
dataset_path = root_folder + 'datasets/'
dataset_name = "CICIDS2017"
pickle_path = root_folder + dataset_name + "/"
proccessed_pickle_data = pickle_path + 'processed/'
pickle_name = dataset_name + "_pickle.pkl"
save_model_path = root_folder + 'saved_models/'

# init variables
batch_size = 32
n_hidden = 20


# helper functions
def model_config(input_shape,output_shape):
    model = Sequential()

    # input layer
    model.add(Dense(1024, input_dim=input_shape, activation='relu'))
    # hidden layer(s)
    model.add(Dropout(0.01))
    # output layer
    model.add(Dense(output_shape,activation='softmax'))

    # Compile & Fit model
    model.compile (
    optimizer=Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],
    )

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

def encode_numeric_zscore(df, name, mean=None, standard_deviation=None):
    
    # define mean
    if mean is None:
        mean = df[name].mean()
    
    # define standarad deviation
    if standard_deviation is None:
        standard_deviation = df[name].std()

    # calculate zscore
    df[name] = (df[name] - mean) / standard_deviation

#load pickle
print(bcolors.WARNING + "Loading pickle" + bcolors.ENDC)
# features
with open(proccessed_pickle_data + 'train_features.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open(proccessed_pickle_data + 'test_features.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open(proccessed_pickle_data + 'val_features.pkl', 'rb') as f:
    X_val = pickle.load(f)

#labels
with open(proccessed_pickle_data + 'train_labels.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open(proccessed_pickle_data + 'test_labels.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open(proccessed_pickle_data + 'val_labels.pkl', 'rb') as f:
    y_val = pickle.load(f)

# balanced data
with open(proccessed_pickle_data + 'train_features_balanced.pkl', 'rb') as f:
    X_train_bal = pickle.load(f)

with open(proccessed_pickle_data + 'train_labels_balanced.pkl', 'rb') as f:
    y_train_bal = pickle.load(f)

print(bcolors.WARNING + "Training model" + bcolors.ENDC)
model = model_config(X_train.shape[1], y_train.shape[1])


# train model
history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1,
    batch_size=batch_size, 
    epochs=5,
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
    batch_size=batch_size,
    verbose=1
)
rounded_predictions = np.argmax(predictions,axis=1)

print(bcolors.WARNING + "Plot confusion matrix" + bcolors.ENDC)
# create the confusion matrix

attack_label = ['Benign', 'DoS/DDoS', 'PortScan', 'Brute Force', 'Web Attack', 'Botnet ARES']

con_mat = tf.math.confusion_matrix(labels=y_val, predictions=rounded_predictions).numpy()

con_mat_norm = np.around(con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=4)

con_mat_df = pd.DataFrame(con_mat_norm, index=attack_label, columns=attack_label)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.get_cmap("Reds"))
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()  