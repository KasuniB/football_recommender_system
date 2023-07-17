import pandas as pd

player_df = pd.read_pickle('outfield.pkl')
print(type(player_df))

gk_df = pd.read_pickle('gk.pkl')
print(type(gk_df))
