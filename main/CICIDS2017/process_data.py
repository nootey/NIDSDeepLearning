import pickle
# Import modules
import numpy as np 
import pandas as pd 
import os
import tensorflow as tf

# Import processing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

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

#define paths
root_folder = "D:/School/diplomska_git/"
dataset_path = root_folder + 'datasets/'
dataset_name = "CICIDS2017"
old_pickle_path = dataset_path + dataset_name + "/"
new_pickle_path = root_folder + dataset_name + "/"
pickle_name = dataset_name + "_pickle.pkl"
save_model_path = root_folder + 'saved_models/'
data_dir  = os.path.join(os.path.abspath(".."), dataset_name)

#load pickle
print(bcolors.WARNING + "Loading pickle" + bcolors.ENDC)
with open(old_pickle_path + pickle_name, 'rb') as f:
    df = pickle.load(f)

# helper functions
def balance_dataset(X, y, undersampling_strategy, oversampling_strategy):

    under_sampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=0)
    X_under, y_under = under_sampler.fit_resample(X, y)

    over_sampler = SMOTE(sampling_strategy=oversampling_strategy)
    X_bal, y_bal = over_sampler.fit_resample(X_under, y_under)
    
    return X_bal, y_bal

# print(df.info())

labels = df['attack_map']
features = df.drop(labels=['label', 'attack_map', 'destination_port_category'], axis=1)

print(bcolors.WARNING + "Prep X and y" + bcolors.ENDC)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42, stratify=labels)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

print(bcolors.WARNING + "Pre-proccess data" + bcolors.ENDC)
categorical_features = features.select_dtypes(exclude=["int64", "float64"]).columns
numeric_features = features.select_dtypes(exclude=[object]).columns

preprocessor = ColumnTransformer(transformers=[
    ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),
    ('numericals', QuantileTransformer(), numeric_features)
])

# pre-proccessing values
columns = numeric_features.tolist()

X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=columns)
X_val = pd.DataFrame(preprocessor.transform(X_val), columns=columns)

# pre-processing labels
le = LabelEncoder()

y_train = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
y_test = pd.DataFrame(le.transform(y_test), columns=["label"])
y_val = pd.DataFrame(le.transform(y_val), columns=["label"])

undersampling_strategy = {
    0: 800000,
    3: 192161,
    4: 34383,
    2: 5131,
    5: 1271,
    1: 1166,
}

oversampling_strategy = {
    0: 800000,
    3: 212102,
    4: 44460,
    2: 50115,
    5: 50284,
    1: 50149,
}

# Balance the training set
X_train_bal, y_train_bal = balance_dataset(X_train, y_train, undersampling_strategy, oversampling_strategy)

print(X_train_bal.shape)
print(y_train_bal.shape)

# Save the balanced training set
print(bcolors.WARNING + "Saving as pickle" + bcolors.ENDC)
# balanced features and labels
with open(new_pickle_path + 'processed/train_features_balanced.pkl', 'wb') as f:
    pickle.dump(X_train_bal, f)

with open(new_pickle_path + 'processed/train_labels_balanced.pkl', 'wb') as f:
    pickle.dump(y_train_bal, f)

X_train.to_pickle(os.path.join(data_dir, 'processed', 'train_features.pkl'))
X_val.to_pickle(os.path.join(data_dir, 'processed', 'val_features.pkl'))
X_test.to_pickle(os.path.join(data_dir, 'processed', 'test_features.pkl'))

y_train.to_pickle(os.path.join(data_dir, 'processed', 'train_labels.pkl'))
y_val.to_pickle(os.path.join(data_dir, 'processed', 'val_labels.pkl'))
y_test.to_pickle(os.path.join(data_dir, 'processed', 'test_labels.pkl'))

print(bcolors.WARNING + "Done" + bcolors.ENDC)
