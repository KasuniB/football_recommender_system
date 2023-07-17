import pandas as pd
import pickle
import streamlit as st
import numpy as np
import os.path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

def trainModel(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 11:-2], df['Cluster'], test_size=0.2, random_state=42)

    # Define the logistic regression model
    model = LogisticRegression(solver='liblinear', max_iter=5000)

    # Define the grid of hyperparameters to search
    param_grid = {'C': [0.1, 1, 10]}

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the best model on the test data
    y_pred = best_model.predict(X_test)

    return best_model, best_params, y_test, y_pred

def getData():
    with open(r'outfield_clustered.pkl', 'rb') as file:
        player_df = pd.DataFrame(pickle.load(file))
    with open(r'player_ID.pickle', 'rb') as file:
        player_ID = pickle.load(file)
    with open(r'engine.pickle', 'rb') as file:
        engine = pickle.load(file)
    
    with open(r'gk_clustered.pkl', 'rb') as file:
        gk_df = pd.DataFrame(pickle.load(file))
    with open(r'gk_ID.pickle', 'rb') as file:
        gk_ID = pickle.load(file)
    with open(r'gk_engine.pickle', 'rb') as file:
        gk_engine = pickle.load(file)
    
    outfield_model = None
    gk_model = None

    if not os.path.isfile('outfield_model.pickle') or os.path.getsize('outfield_model.pickle') == 0:
        outfield_model, _, _, _ = trainModel(player_df)
        with open('outfield_model.pickle', 'wb') as file:
            pickle.dump(outfield_model, file)
    else:
        with open('outfield_model.pickle', 'rb') as file:
            outfield_model = pickle.load(file)

    if not os.path.isfile('gk_model.pickle') or os.path.getsize('gk_model.pickle') == 0:
        gk_model, _, _, _ = trainModel(gk_df)
        with open('gk_model.pickle', 'wb') as file:
            pickle.dump(gk_model, file)
    else:
        with open('gk_model.pickle', 'rb') as file:
            gk_model = pickle.load(file)

    return player_df, player_ID, engine, gk_df, gk_ID, gk_engine, outfield_model, gk_model

   
def getOutfieldRecommendations(df, ID, metric, league, comparison, query, count=5, model=None):
    if model is not None:
        # Apply model to get predictions
        X = df.iloc[:, 11:-2]
        y_pred = model.predict(X)
        df['Cluster'] = y_pred

    df_res = df.iloc[:, [1, 3, 5, 6, 11]].copy()

    df_res['Player'] = list(ID.keys())
    df_res.insert(1, 'Similarity', metric)
    df_res = df_res.sort_values(by=['Similarity'], ascending=False)
    metric = [str(num) + '%' for num in df_res['Similarity']]
    df_res['Similarity'] = metric
    df_res = df_res.iloc[1:, :]

    if comparison == 'Same position':
        q_pos = list(df[df['Player'] == query.split(' (')[0]]['Pos'])[0]
        df_res = df_res[df_res['Pos'] == q_pos]

    if league != 'All':
        df_res = df_res[df_res['Comp'] == league]

    df_res = df_res.head(count)
    return df_res

def getGoalkeeperRecommendations(df, ID, metric, league, comparison, query, count=5, model=None):
    if model is not None:
        # Apply model to get predictions
        if isinstance(model, LogisticRegression):
            X_train = df.iloc[:, 11:-1]
        else:
            X_train = df.iloc[:, 11:-2]
        selected_features = model.coef_[0] != 0

        if len(selected_features) < X_train.shape[1]:
            selected_features = np.concatenate((selected_features, np.zeros(X_train.shape[1] - len(selected_features), dtype=bool)))

        selected_columns = X_train.columns[selected_features]
        X_pred = X_train[selected_columns]
        y_pred = model.predict(X_pred)
        df['Cluster'] = y_pred

    df_res = df.iloc[:, [1, 3, 5, 6, 11]].copy()

    df_res['Player'] = list(ID.keys())
    df_res.insert(1, 'Similarity', metric)
    df_res = df_res.sort_values(by=['Similarity'], ascending=False)
    metric = [str(num) + '%' for num in df_res['Similarity']]
    df_res['Similarity'] = metric
    df_res = df_res.iloc[1:, :]

    if comparison == 'Same position':
        q_pos = list(df[df['Player'] == query.split(' (')[0]]['Pos'])[0]
        df_res = df_res[df_res['Pos'] == q_pos]

    if league != 'All':
        df_res = df_res[df_res['Comp'] == league]

    df_res = df_res.head(count)
    return df_res




st.title('Football Player Recommender')
st.write('A Web App to recommend football players who play similar to your favorite players!')

player_df, player_ID, engine, gk_df, gk_ID, gk_engine, outfield_model, gk_model = getData()

player_type = st.sidebar.radio('Player type', ['Outfield players', 'Goal Keepers'])

if player_type == 'Outfield players':
    df, ID, engine = player_df, player_ID, engine
else:
    df, ID, engine = gk_df, gk_ID, gk_engine

st.sidebar.markdown('### Input player info')
players = list(ID.keys())
query = st.sidebar.selectbox('Select a player', players)
count = st.sidebar.slider('How many similar players do you want?', min_value=1, max_value=10, value=5)
comparison = ('Comparison', ['All positions', 'Same position'])
league = st.sidebar.selectbox('League', ['All', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'])

metric = engine[query]

result = None
if player_type == 'Outfield players':
    result = getOutfieldRecommendations(df, ID, metric, league, comparison, query, count, model=outfield_model)
else:
    result = getGoalkeeperRecommendations(df, ID, metric, league, comparison, query, count, model=gk_model)



st.markdown('### Here are players who play similar to {}'.format(query.split(' (')[0]))
st.dataframe(result)