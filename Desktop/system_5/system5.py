import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import decomposition
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

st.title('Football Player Recommender')
st.write('A Web App to recommend football players who play similar to your favorite players!')

redundant = ['Rk','Player','Nation','Pos','Squad','Comp','Age','Born','90s','Matches']

#@st.cache_data
def load_data():
    df = pd.read_csv('general.csv', header=0).drop(['Rk','Matches'], axis=1)
    shooting = pd.read_csv('shooting.csv', header=0).drop(redundant, axis=1)
    passing = pd.read_csv('passing.csv', header=0).drop(redundant, axis=1)
    passing_types = pd.read_csv('passing_types.csv', header=0).drop(redundant, axis=1)
    gca = pd.read_csv('gca.csv', header=0).drop(redundant, axis=1)
    defense = pd.read_csv('defense.csv', header=0).drop(redundant, axis=1)
    possession = pd.read_csv('possession.csv', header=0).drop(redundant, axis=1)
    misc = pd.read_csv('misc.csv', header=0).drop(redundant, axis=1)

    def renameColumns(table_no, df):
        num = str(table_no) + "_"
        return df.rename(columns=lambda x: num + x)

    shooting = renameColumns(2, shooting)
    passing = renameColumns(3, passing)
    passing_types = renameColumns(4, passing_types)
    gca = renameColumns(5, gca)
    defense = renameColumns(6, defense)
    possession = renameColumns(7, possession)
    misc = renameColumns(8, misc)

    grand = pd.concat([df , shooting, passing, passing_types, gca, defense, possession, misc], axis=1)

    # At least 3 90s played
    df = grand[grand['90s'] >= 3]

    # Excluding goalkeepers
    df = df[df['Pos'] != 'GK'].reset_index(drop=True)

    # Extracting player names
    df['Player'] = df['Player'].str.split('\\', expand=True)[0]

    # Removing country short forms
    df['Comp'] = df['Comp'].str.split(' ', expand=True, n=1)[1]
    return df

def similarity(player1, player2):
    p1 = stats[player_ID[player1], :]
    p2 = stats[player_ID[player2], :]
    sim = 100 - distance.cosine(p1, p2) * 100
    return round(sim, 2)

def normalize(metric):
    min_metric = min(metric)
    max_metric = max(metric)
    return [(m - min_metric) / (max_metric - min_metric) * 100 for m in metric]

df = load_data()

# Applying modifications here:
players = []
for idx, row in df.iterrows():
    players.append(row['Player'] + '({})'.format(row['Squad']))

player_ID = dict(zip(players, np.arange(len(players))))

# Save the player ID dictionary
with open('player_ID.pickle', 'wb') as file:
    pickle.dump(player_ID, file)

stats = df.iloc[:, 12:-1]
labels = df['Pos']
data = StandardScaler().fit_transform(stats)
model = TSNE(n_components=2, perplexity=30, random_state=0)
tsne_data = model.fit_transform(data)
tsne_data = np.vstack((tsne_data.T, labels)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension 1", "Dimension 2", "Positions"))

# Standardizing the data
data = StandardScaler().fit_transform(stats)

# Setting up and running PCA
pca = decomposition.PCA()
pca.n_components = 142
pca_data = pca.fit_transform(data)

# % Variance explained per components
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)

# Cumulative variance explained
cum_var_explained = np.cumsum(percentage_var_explained)

stats = pca_data[:, :90]

# Calculate player similarity
engine = {}
for query in tqdm(players):
    metric = []
    for player in players:
        value = similarity(query, player)
        metric.append(value)
    metric = normalize(metric)
    engine[query] = metric

# Save the similarity engine
with open('engine.pickle', 'wb') as file:
    pickle.dump(engine, file)

# Save the entire stats DataFrame
with open('outfield.pkl', 'wb') as file:
    pickle.dump(stats, file)

#@st.cache_data(show_spinner=False)
def getData():
    with open('outfield.pkl', 'rb') as file:
        player_df = pickle.load(file)
        player_df = pd.DataFrame(player_df)  # Convert player_df to DataFrame
    with open('player_ID.pickle', 'rb') as file:
        player_ID = pickle.load(file)
    with open('engine.pickle', 'rb') as file:
        engine = pickle.load(file)
    return player_df, player_ID, engine

def getRecommendations(df, metric, df_type, league, comparison, count=5):
    if df_type == 'outfield':
        df_res = pd.DataFrame(df).iloc[:, [-1]].copy()  # Convert df to DataFrame
    else:
        df_res = pd.DataFrame(df).iloc[:, [1, 3, 5, 6, 11]].copy()  # Convert df to DataFrame
    
    df_res['Player'] = list(player_ID.keys())
    df_res.insert(1, 'Similarity', metric)
    df_res = df_res.sort_values(by=['Similarity'], ascending=False)
    metric = [str(num) + '%' for num in df_res['Similarity']]
    df_res['Similarity'] = metric
    # Ignoring the top result
    df_res = df_res.iloc[1:, :]

    # Comparison filtering
    if comparison == 'Same position' and df_type == 'outfield':
        q_pos = list(df[df['Player'] == query.split(' (')[0]]['Pos'])[0]
        df_res = df_res[df_res['Pos'] == q_pos]

    # League filtering
    if league == 'All':
        pass
    else:
        df_res = df_res[df_res['Comp'] == league]

    # Limiting the result to the desired count
    df_res = df_res.head(count)
    return df_res  # Returning the final result
   

# Returning the final result
st.sidebar.markdown('### Input player info')
query = st.sidebar.selectbox('Select a player', players)
count = st.sidebar.slider('How many similar players do you want?', min_value=1, max_value=10, value=5)
comparison = st.sidebar.selectbox('Comparison', ['All positions', 'Same position'])
league = st.sidebar.selectbox('League', ['All', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'])

player_df, player_ID, engine = getData()

# Querying the engine
metric = engine[query]

# Generating the result
result = getRecommendations(player_df, metric, 'outfield', league, comparison, count)
st.markdown('### Here are players who play similar to {}'.format(query.split(' (')[0]))
st.dataframe(result)
