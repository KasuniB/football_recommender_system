import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.spatial import distance
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans

redundant = ['Rk','Player','Nation','Pos','Squad','Comp','Age','Born','90s','Matches']

goalkeeping = pd.read_csv('goalkeeping.csv', header=0).drop(['Rk','Matches'], axis=1)
advanced_goalkeeping = pd.read_csv('advanced_goalkeeping.csv', header=0).drop(redundant, axis=1)

def renameColumns(table_no, df):
    num = str(table_no) + "_"
    return df.rename(columns=lambda x: num + x)

advanced_goalkeeping = renameColumns(8, advanced_goalkeeping)

grand = pd.concat([goalkeeping, advanced_goalkeeping], axis=1)
print(grand.head())
df = grand[grand['90s'] >= 3]
df = df[df['Pos']=='GK'].reset_index()
df.loc[:, 'Comp'] = df['Comp'].str.split(' ', expand=True, n=1)[1]

with open('gk.pkl', 'wb') as file:
    pickle.dump(df, file)

players = []
for idx in range(len(df)):
    players.append(df['Player'][idx] + '({})'.format(df['Squad'][idx]))
player_ID = dict(zip(players, np.arange(len(players))))

with open('gk_ID.pickle', 'wb') as file:
    pickle.dump(player_ID, file)

print(df)

stats = df.iloc[:, 11:-1]
labels = df['Pos']
data = StandardScaler().fit_transform(stats)

pca = decomposition.PCA()
pca.n_components = 39
pca_data = pca.fit_transform(data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_explained)

# The number of components should be the same as in the PCA
stats = pca_data[:, :39]

def getStats(name):
    idx = player_ID[name]
    return stats[idx, :]

def similarity(player1, player2):
   return 1- distance.cosine(getStats(player1),getStats(player2))

def normalize(array):
   return np.array([round(num, 2) for num in (array - min(array))*100/(max(array)-min(array))])

engine = {}
for query in tqdm(players):
    metric = []
    for player in players:
        value = similarity(query, player)
        metric.append(value)
    metric = normalize(metric)
    engine[query] = metric

with open('gk_engine.pickle', 'wb') as file:
    pickle.dump(engine, file)
# Apply K-means clustering to the data
kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(stats)

# Add cluster labels to the DataFrame
df['Cluster'] = cluster_labels

# Save the updated DataFrame with cluster labels
with open('gk_clustered.pkl', 'wb') as file:
    pickle.dump(df, file)
