# imports
import pickle
import pandas as pd 
import numpy as np 
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

#helper functions
def clean_column(column):
    column = column.strip(' ')
    column = column.replace('/', '_')
    column = column.replace(' ', '_')
    column = column.lower()
    return column

#define paths
root_folder = "D:/School/diplomska_git/"
dataset_path = root_folder + 'datasets/'
dataset_name = "CICIDS2017"
pickle_path = dataset_path + dataset_name + "/"
pickle_name = dataset_name + "_pickle.pkl"
save_model_path = root_folder + 'saved_models/'

paths = [
    dataset_path + dataset_name + '/Monday-WorkingHours.pcap_ISCX.csv',
    dataset_path + dataset_name + '/Tuesday-WorkingHours.pcap_ISCX.csv',
    dataset_path + dataset_name + '/Wednesday-workingHours.pcap_ISCX.csv',
    dataset_path + dataset_name + '/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    dataset_path + dataset_name + '/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    dataset_path + dataset_name + '/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    dataset_path + dataset_name + '/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    dataset_path + dataset_name + '/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
]

# fetch the training file
print(bcolors.WARNING + "Reading files" + bcolors.ENDC)
df = pd.read_csv(paths[0])
for i in range(1,len(paths)):
    temp = pd.read_csv(paths[i])
    df = pd.concat([df,temp])

#re-format columns to make them more readable and usable
print(bcolors.WARNING + "Cleaning dataset" + bcolors.ENDC)
df.columns = [clean_column(column) for column in df.columns]
df.drop(labels=['fwd_header_length.1'], axis= 1, inplace=True)

#print(df.duplicated().any()) #there are duplicates
#print('NUM values before: ', df.shape[0], end='\n\n')
# drop duplicates
df.drop_duplicates(inplace=True, keep=False, ignore_index=True)
#print('NUM values after: ', df.shape[0])

# check for null values (prints out the columns with null values)
#print(df.columns[df.isnull().any()])

# drop null values
df.dropna(axis=0, inplace=True, how="any")

# check for infinite values
np.all(np.isfinite(df.drop(['label'], axis=1)))

# replace infinite values with NaN
df.replace([-np.inf, np.inf], np.nan, inplace=True)

# drop infinte values
df.dropna(axis=0, inplace=True, how='any')

# find features with null standard deviation
dataset_std = df.std(numeric_only=True)

# find the features that meet the threshold
constant_features = [column for column, std in dataset_std.iteritems() if std < 0.01]

# drop constant features
df.drop(labels=constant_features, axis=1, inplace=True)

# destination port range is too big, needs to be compressed into 3 categories
conditions = [
    (df['destination_port'] >= 1) & (df['destination_port'] < 1024),
    (df['destination_port'] >= 1024) & (df['destination_port'] < 49152),
    (df['destination_port'] >= 49152) & (df['destination_port'] <= 65535)
]

categories = [
    "1 - 1023", 
    "1024 - 49151",
    "49152 - 65535"
]

df.insert(1, 'destination_port_category', np.select(conditions, categories, default="0"))

#check if any values are highly corelated to eachother
print(bcolors.WARNING + "Removing features with tight corelation" + bcolors.ENDC)
dataset_corr = df.corr()
# #plot correlation matrix
# fig = plt.figure(figsize=(15, 15))
# sns.set(font_scale=1.0)
# ax = sns.heatmap(dataset_corr, annot=False)
# plt.show()

# create a mask
mask = np.triu(np.ones_like(dataset_corr, dtype=bool))
tri_df = dataset_corr.mask(mask)

# find correlated features
correlated_features = [c for c in tri_df.columns if any(tri_df[c] > 0.98)]

# drop the correlated features
df.drop(labels=correlated_features, axis=1, inplace=True)

print(bcolors.WARNING + "Mapping attacks" + bcolors.ENDC)
df['label'] = df['label'].str.replace('Web Attack �', 'Web Attack', regex=False)

# classify attacks as possible attack types
attack_group = {
    'BENIGN': 'Benign',
    'PortScan': 'PortScan',
    'DDoS': 'DoS/DDoS',
    'DoS Hulk': 'DoS/DDoS',
    'DoS GoldenEye': 'DoS/DDoS',
    'DoS slowloris': 'DoS/DDoS', 
    'DoS Slowhttptest': 'DoS/DDoS',
    'Heartbleed': 'DoS/DDoS',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Botnet ARES',
    'Web Attack Brute Force': 'Web Attack',
    'Web Attack Sql Injection': 'Web Attack',
    'Web Attack XSS': 'Web Attack',
    'Infiltration': 'Botnet ARES'
}

# Create grouped label column
df['attack_map'] = df['label'].map(lambda x: attack_group[x])
print(df['attack_map'].value_counts())

# # visualize attack type distribution
# fig = plt.figure(figsize=(12, 5))
# attack = df['attack_map'].value_counts()
# attack_count = attack.values
# attack_type = attack.index
# bar = plt.bar(attack_type, attack_count, align='center')

# for rect in bar:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2.0, height, format(height, ','), ha='center', va='bottom', fontsize=12)

# plt.title('Distribution of different type of network activity in the dataset', fontsize=18)
# plt.xlabel('Network activity', fontsize=16)
# plt.ylabel('Number of instances', fontsize=16)
# plt.grid(True)
# plt.show()
# #print(df_final.groupby('label')['label'].count())

print(bcolors.WARNING + "Saving as pickle" + bcolors.ENDC)
with open(pickle_path + pickle_name, 'wb') as f:
    pickle.dump(df, f)

print(bcolors.WARNING + "Done" + bcolors.ENDC)