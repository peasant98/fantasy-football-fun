"""

API for play index from pro
football reference.

"""

from fantasy_football_fun.pfr.pfr_query import PFRQuery
import pandas as pd
import time


def create_data():
    query = PFRQuery(year_start=1900, order_by='fantasy_points')
    offset = 0
    dfs = []
    while True:
        query.offset = offset
        time.sleep(2)
        df = query.table
        if df is not None:
            dfs.append(df)
        else:
            break
        inc = offset + 100
        print(f"Got players {offset} to {inc}")
        offset += len(df)
    final_df = pd.concat(dfs)
    final_df.to_csv("players.csv")

# make sure to
#  rerun things


if __name__ == '__main__':
    create_data()
