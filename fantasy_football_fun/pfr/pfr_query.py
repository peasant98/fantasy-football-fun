"""
class to allow users to easily query
for what search terms they want.

"""
import fantasy_football_fun.const as C


class PRFQuery():
    def __init__(self, year_start=2000, year_end=2019, season_start=1,
                 season_end=-1, age_min=18, age_max=48,
                 positions=["qb", "wr", "te", "rb"],
                 draft_year_min=1936, draft_year_max=2019,
                 draft_slot_min=1, draft_slot_max=500,
                 draft_pick_in_round="pick_overall",
                 is_hof=None,
                 draft_positions=["qb", "wr", "te", "rb"],
                 order_by="pass_td", offset=0, order_by_asc=False):
        
        self.base_string = "https://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single"
        self.year_start, self.year_end = self.validate_range(year_start, year_end)
        self.season_start, self.season_end = self.validate_range(season_start, season_end)

        self.draft_year_min, self.draft_year_max = self.validate_range(draft_year_min, draft_year_max)

        self.draft_slot_min, self.draft_slot_max = self.validate_range(draft_slot_min, draft_slot_max)

        self.age_min, self.age_max = self.validate_range(age_min, age_max)

        self.positions = self.validate_list(C.POSITIONS, positions)

        self.draft_positions = self.validate_list(C.POSITIONS, positions)

        # order_by is the most important term here
        self.order_by = self.validate_list(C.ORDER_BY_TERMS, order_by)

        provided_wrong_positions = 0 == len(set(positions) - set(C.POSITIONS))
        if provided_wrong_positions:
            raise ValueError("Wrong positions supplied! Please refer to const.py.")
        else:
            self.positions = positions
        self.is_hof_string = ""

        if is_hof is not None:
            self.is_hof_string = "Y" if is_hof else "N"


    def construct_query_string(self):
        """
        Constructs the query string to be used for webscraping.
        """
        self.standard_string = f"""{self.base_string}&year_min=
                                   {self.year_start}&year_max=
                                   {self.year_end}&season_start=
                                   {self.season_start}&season_end=
                                   {self.season_end}&draft_year_min=
                                   {self.draft_year_min}&draft_year_max=
                                   {self.draft_year_max}&draft_slot_min=
                                   {self.draft_slot_min}&draft_slot_max=
                                   {self.draft_slot_max}&draft_pick_in_round=
                                   pick_overall&conference=any&c5val=1.0%&order_by=
                                   """
        self.pos_string = "".join([f"&pos[]={pos}" for pos in self.positions])

        self.draft_pos_string = "".join([f"&draft_pos[]={pos}" for pos in self.draft_positions])
        self.query_string = f"{self.standard_string}{self.pos_string}{self.draft_pos_string}"

    def validate_list(self, const_list, user_list):
        provided_wrong_vals = 0 == len(set(user_list) - set(const_list))
        if provided_wrong_vals:
            raise ValueError("Wrong vaues supplied! Please refre to const.py.")
        else:
            return user_list

    def validate_range(self, min, max, param_name, minus_one_valid=False):
        if not minus_one_valid:
            # two positive numbers must be provided
            assert max >= min, f"""{param_name} end  xshould be the same of higher
                                as {param_name} s"""
            return min, max

# https://www.pro-football-reference.com/play-index/psl_finder.cgi?
# request=1&match=single&year_min=2019&year_max=2019&
# season_start=1&season_end=-1&pos%5B%5D=qb&pos%5B%5D=rb&pos%5B%5D=wr&pos%5B%5D=te&pos%5B%5D=e&pos%5B%5D=t&pos%5B%5D=g&pos%5B%5D=c&pos%5B%5D=ol&pos%5B%5D=dt&pos%5B%5D=de&pos%5B%5D=dl&pos%5B%5D=ilb&pos%5B%5D=olb&pos%5B%5D=lb&pos%5B%5D=cb&pos%5B%5D=s&pos%5B%5D=db&pos%5B%5D=k&pos%5B%5D=p&draft_year_min=1936&draft_year_max=2019&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=pick_overall&conference=any&draft_pos%5B%5D=qb&draft_pos%5B%5D=rb&draft_pos%5B%5D=wr&draft_pos%5B%5D=te&draft_pos%5B%5D=e&draft_pos%5B%5D=t&draft_pos%5B%5D=g&draft_pos%5B%5D=c&draft_pos%5B%5D=ol&draft_pos%5B%5D=dt&draft_pos%5B%5D=de&draft_pos%5B%5D=dl&draft_pos%5B%5D=ilb&draft_pos%5B%5D=olb&draft_pos%5B%5D=lb&draft_pos%5B%5D=cb&draft_pos%5B%5D=s&draft_pos%5B%5D=db&draft_pos%5B%5D=k&draft_pos%5B%5D=p&c5val=1.0&order_by=pass_td