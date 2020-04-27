import pandas as pd

def get_all_data(filename):
    df = pd.read_csv(filename)
    print(df)

get_all_data("finalized_players.csv")