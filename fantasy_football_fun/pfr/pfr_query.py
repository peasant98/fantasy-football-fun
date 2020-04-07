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
                 conference="any",
                 draft_pos=["qb", "wr", "te", "rb"],
                 c5val=1.0, order_by="pass_td", offset=0):
        self.base_string = "https://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single"
        assert year_end >= year_end, "year_end should be the same or higher as year_start."
        if season_end != -1:
            assert season_end >= season_start, "season_end should be the same or higher as season_start"
        provided_wrong_positions = 0 == len(set(positions) - set(C.POSITIONS))
        if provided_wrong_positions:
            raise ValueError("Wrong positions supplied! Please refer to const.py.")

    def validate_range(self, min, max):
        pass


# https://www.pro-football-reference.com/play-index/psl_finder.cgi?
# request=1&match=single&year_min=2019&year_max=2019&
# season_start=1&season_end=-1&pos%5B%5D=qb&pos%5B%5D=rb&pos%5B%5D=wr&pos%5B%5D=te&pos%5B%5D=e&pos%5B%5D=t&pos%5B%5D=g&pos%5B%5D=c&pos%5B%5D=ol&pos%5B%5D=dt&pos%5B%5D=de&pos%5B%5D=dl&pos%5B%5D=ilb&pos%5B%5D=olb&pos%5B%5D=lb&pos%5B%5D=cb&pos%5B%5D=s&pos%5B%5D=db&pos%5B%5D=k&pos%5B%5D=p&draft_year_min=1936&draft_year_max=2019&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=pick_overall&conference=any&draft_pos%5B%5D=qb&draft_pos%5B%5D=rb&draft_pos%5B%5D=wr&draft_pos%5B%5D=te&draft_pos%5B%5D=e&draft_pos%5B%5D=t&draft_pos%5B%5D=g&draft_pos%5B%5D=c&draft_pos%5B%5D=ol&draft_pos%5B%5D=dt&draft_pos%5B%5D=de&draft_pos%5B%5D=dl&draft_pos%5B%5D=ilb&draft_pos%5B%5D=olb&draft_pos%5B%5D=lb&draft_pos%5B%5D=cb&draft_pos%5B%5D=s&draft_pos%5B%5D=db&draft_pos%5B%5D=k&draft_pos%5B%5D=p&c5val=1.0&order_by=pass_td