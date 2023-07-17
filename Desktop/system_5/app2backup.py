import pandas as pd
import pickle
import streamlit as st

def getData():
    with open('outfield.pkl', 'rb') as file:
        player_df = pickle.load(file)
        player_df = pd.DataFrame(player_df)  # Convert player_df to DataFrame
    with open('player_ID.pickle', 'rb') as file:
        player_ID = pickle.load(file)
    with open('engine.pickle', 'rb') as file:
        engine = pickle.load(file)
    
    with open('gk.pkl', 'rb') as file:
        gk_df = pickle.load(file)
        gk_df = pd.DataFrame(gk_df)  # Convert gk_df to DataFrame
    with open('gk_ID.pickle', 'rb') as file:
        gk_ID = pickle.load(file)
    with open('gk_engine.pickle', 'rb') as file:
        gk_engine = pickle.load(file)

    return player_df, player_ID, engine, gk_df, gk_ID, gk_engine

def getRecommendations(df, metric, df_type, league, comparison, count=5):
    if df_type == 'outfield players':
        df_res = pd.DataFrame(df).iloc[:, [1, 3, 5, 6, 11,-1]].copy()  # Convert df to DataFrame
    else:
        df_res = pd.DataFrame(df).iloc[:, [1, 3, 5, 6, 11]].copy()  # Convert df to DataFrame
    
    df_res['Player'] = list(ID.keys())
    df_res.insert(1, 'Similarity', metric)
    df_res = df_res.sort_values(by=['Similarity'], ascending=False)
    metric = [str(num) + '%' for num in df_res['Similarity']]
    df_res['Similarity'] = metric
    # Ignoring the top result
    df_res = df_res.iloc[1:, :]

    # Comparison filtering
    if comparison == 'Same position' and df_type == 'outfield players':
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

# Start the Streamlit app
st.title('Football Player Recommender')
st.write('A Web App to recommend football players who play similar to your favorite players!')

player_df, player_ID, engine, gk_df, gk_ID, gk_engine = getData()
player_type = st.sidebar.radio('Player type', ['Outfield players', 'Goal Keepers'])

if player_type == 'Outfield players':
    df, ID, engine = player_df, player_ID, engine
else:
    df, ID, engine = gk_df, gk_ID, gk_engine

# Returning the final result
st.sidebar.markdown('### Input player info')
players = list(ID.keys())  # Change from player_ID to ID
query = st.sidebar.selectbox('Select a player', players)
count = st.sidebar.slider('How many similar players do you want?', min_value=1, max_value=10, value=5)
comparison = st.sidebar.selectbox('Comparison', ['All positions', 'Same position'])
league = st.sidebar.selectbox('League', ['All', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'])

# Querying the engine
metric = engine[query]

# Generating the result
result = getRecommendations(df, metric, player_type.lower(), league, comparison, count)
st.markdown('### Here are players who play similar to {}'.format(query.split(' (')[0]))
st.dataframe(result)
