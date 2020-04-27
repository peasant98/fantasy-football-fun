import requests
from bs4 import BeautifulSoup
import pandas as pd


def create_df(query_string):
    """
    creates a dataframe based the specified query.
    """
    response = requests.get(query_string)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    headers = []
    all_players_info = []
    player_links = []
    theads = soup.find_all('thead')
    if len(theads) == 0:
        return None

    thead = theads[0]

    trs = thead.find_all('tr')
    if len(trs) < 2:
        return None

    ths = trs[1].find_all('th')
    for th in ths:
        headers.append(str(th.text))

    tbody = soup.find_all('tbody')[0]
    # also length 1
    trs = tbody.find_all('tr')
    # every player in the table
    for tr in trs:
        rank = tr.find_all('th')[0].text
        if rank == "Rk":
            pass
        else:
            player_info = [int(rank)]
            tds = tr.find_all("td")
            for td in tds:
                # each column, e.g., fantasy points
                if td.text is None:
                    val = -1
                else:
                    val = td.text
                try:
                    val = float(val)
                    if val == int(val):
                        val = int(val)
                except Exception as e:
                    # failed to convert. go with string instead
                    val = str(val)
                    if td['data-stat'] == 'player':
                        links = td.find_all("a")
                        if len(links) != 0:
                            link = links[0]['href']
                            player_links.append(link)
                player_info.append(val)
            # an actual player exists here.
            all_players_info.append(player_info)
    df = pd.DataFrame(all_players_info, columns=headers)
    df['link'] = pd.Series(player_links, index=df.index)
    return df
