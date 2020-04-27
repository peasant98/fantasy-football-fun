import pandas as pd
import requests
from bs4 import BeautifulSoup

import fantasy_football_fun.const as C


def add_positions(filename, col_name):
    """
    given the links within the dataframe, get the positions
    """

    df = pd.read_csv(filename)
    positions = []
    sorted_df = df.sort_values(['Player', 'Year'])
    cur_name = sorted_df['Player'].values[0]
    prev_name = None
    links = sorted_df[col_name].values
    cur_position = None
    for idx, link in enumerate(links):
        cur_name = sorted_df['Player'].values[idx]
        if prev_name != cur_name:
            url = f'{C.BASE_URL}{link}'
            cur_position = extract_position_from_link(url)
            positions.append(cur_position)
            print(f'{cur_name} with {cur_position}')
        else:
            positions.append(cur_position)
        prev_name = cur_name

    sorted_df['position'] = pd.Series(positions, index=sorted_df.index)
    sorted_df.to_csv('finalized_players.csv')


def extract_position_from_link(link):
    """
    gets the html from `link` and parses it for the
    position.
    """
    response = requests.get(link)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    mydivs = soup.find_all("div", {"id": "meta"})
    if len(mydivs) != 1:
        return "unknown"

    pars = mydivs[0].find_all("p")
    if len(mydivs) < 1:
        return "unknown"

    for par in pars:
        if "Position:" in par.text:
            start_idx = par.text.find("Position:")
            strings = par.text[start_idx:17].split()
            if len(strings) < 2:
                return "unknown"
            position = strings[1]
            return position
    return "unknown"


add_positions('players.csv', col_name='link')
