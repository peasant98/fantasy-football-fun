"""

API for play index from pro
football reference.

"""

from fantasy_football_fun.pfr.pfr_query import PFRQuery
import fantasy_football_fun.const as C
import pandas as pd
import time


def create_data():
    query = PFRQuery(year_start=1900, order_by='fantasy_points')
    offset = 0
    dfs = []
    while True:
        query.offset = offset
        time.sleep(1)
        df = query.table
        dfs.append(df)
        inc = offset + 100
        print(f"Got players {offset} to {inc}")
        offset += len(df)
        if offset > C.MAX_FF_POINTS_VAL:
            print("Enough data obtained!")
            break
    final_df = pd.concat(dfs)
    final_df.to_csv("players.csv")


if __name__ == '__main__':
    create_data()
