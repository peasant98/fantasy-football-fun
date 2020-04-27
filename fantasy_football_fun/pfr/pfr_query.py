"""
class to allow users to easily query
for what search terms they want.

"""
import fantasy_football_fun.const as C
import requests
import pandas as pd
from bs4 import BeautifulSoup


class PFRQuery():
    def __init__(self, year_start=2000, year_end=2019, season_start=1,
                 season_end=-1, age_min=18, age_max=48,
                 positions=C.POSITIONS,
                 draft_year_min=1936, draft_year_max=2019,
                 draft_slot_min=1, draft_slot_max=500,
                 draft_pick_in_round="pick_overall",
                 is_hof=None,
                 draft_positions=C.POSITIONS,
                 order_by="pass_td", offset=0, order_by_asc=False):

        self.base_string = "https://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single"

        self.year_start, self.year_end = self.validate_range(year_start, year_end, "year")

        self.season_start, self.season_end = self.validate_range(season_start, season_end, "season", minus_one_valid=True)

        self.draft_year_min, self.draft_year_max = self.validate_range(draft_year_min, draft_year_max, "draft_year")

        self.draft_slot_min, self.draft_slot_max = self.validate_range(draft_slot_min, draft_slot_max, "draft_slot")

        self.age_min, self.age_max = self.validate_range(age_min, age_max, "age")

        self.positions = self.validate_list(C.POSITIONS, positions)

        self.draft_positions = self.validate_list(C.POSITIONS, positions)

        # order_by is the most important term here
        self.order_by = self.validate__item_in_list(C.ORDER_BY_TERMS, order_by)

        self.is_hof_string = ""
        self.offset = 0
        if is_hof is not None:
            self.is_hof_string = "Y" if is_hof else "N"
        self.construct_query_string()

    def construct_query_string(self):
        """
        Constructs the query string to be used for webscraping.
        """
        self.standard_string = f"{self.base_string}&year_min=" \
                               f"{self.year_start}&year_max=" \
                               f"{self.year_end}&season_start=" \
                               f"{self.season_start}&season_end=" \
                               f"{self.season_end}&draft_year_min=" \
                               f"{self.draft_year_min}&draft_year_max=" \
                               f"{self.draft_year_max}&draft_slot_min=" \
                               f"{self.draft_slot_min}&draft_slot_max=" \
                               f"{self.draft_slot_max}&draft_pick_in_round=" \
                               f"pick_overall&conference=any&c5val=1.0&order_by=" \
                               f"{self.order_by}"
        # positions to look on
        self.pos_string = "".join([f"&pos[]={pos}" for pos in self.positions])
        # positions drafted at
        self.draft_pos_string = "".join([f"&draft_pos[]={pos}" for pos in self.draft_positions])
        self.base_string = f"{self.standard_string}{self.pos_string}{self.draft_pos_string}"
        self.query_string = self.base_string

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        """
        Adds an offset to a string
        """
        self._offset = offset
        self.query_string = f'{self.base_string}&offset={self.offset}'

    @property
    def table(self):
        """
        to be done - get the whole table of data.

        # Parameters:
        """
        response = requests.get(self.query_string)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        headers = []
        all_players_info = []
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
                    player_info.append(val)
                # an actual player exists here.
                all_players_info.append(player_info)
        df = pd.DataFrame(all_players_info, columns=headers)
        return df

    def validate__item_in_list(self, master_list, user_item):
        if user_item not in master_list:
            raise ValueError("Wrong vaues supplied! Please refer to const.py.")
        else:
            return user_item

    def validate_list(self, const_list, user_list):
        wrong_vals_amt = len(set(user_list) - set(const_list))
        if wrong_vals_amt > 0:
            raise ValueError("Wrong vaues supplied! Please refer to const.py.")
        else:
            return user_list

    def validate_range(self, min_val, max_val, param_name, minus_one_valid=False):
        """
        validates range of min and max.
        """
        if not minus_one_valid:
            # two positive numbers must be provided
            assert max_val >= min_val, f"""{param_name} end should be the same or higher as {param_name} start"""
        return min_val, max_val

# https://www.pro-football-reference.com/play-index/psl_finder.cgi?
# request=1&match=single&year_min=2019&year_max=2019&
# season_start=1&season_end=-1&pos%5B%5D=qb&pos%5B%5D=rb&pos%5B%5D=wr&pos%5B%5D=te&pos%5B%5D=e&pos%5B%5D=t&pos%5B%5D=g&pos%5B%5D=c&pos%5B%5D=ol&pos%5B%5D=dt&pos%5B%5D=de&pos%5B%5D=dl&pos%5B%5D=ilb&pos%5B%5D=olb&pos%5B%5D=lb&pos%5B%5D=cb&pos%5B%5D=s&pos%5B%5D=db&pos%5B%5D=k&pos%5B%5D=p&draft_year_min=1936&draft_year_max=2019&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=pick_overall&conference=any&draft_pos%5B%5D=qb&draft_pos%5B%5D=rb&draft_pos%5B%5D=wr&draft_pos%5B%5D=te&draft_pos%5B%5D=e&draft_pos%5B%5D=t&draft_pos%5B%5D=g&draft_pos%5B%5D=c&draft_pos%5B%5D=ol&draft_pos%5B%5D=dt&draft_pos%5B%5D=de&draft_pos%5B%5D=dl&draft_pos%5B%5D=ilb&draft_pos%5B%5D=olb&draft_pos%5B%5D=lb&draft_pos%5B%5D=cb&draft_pos%5B%5D=s&draft_pos%5B%5D=db&draft_pos%5B%5D=k&draft_pos%5B%5D=p&c5val=1.0&order_by=pass_td

