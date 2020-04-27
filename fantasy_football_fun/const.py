"""
This module serves to provide a variety of constants
for the Fantasy Football Fun package.

"""

POSITIONS = ["qb", "rb", "wr", "te", "e",
             "t", "g", "c", "ol", "dt", "de",
             "dl", "ilb", "olb", "lb", "cb", "s",
             "db", "k", "p"]

# all of the terms users can order by
ORDER_BY_TERMS = ["player", "year_id", "age",
                  "height_in", "weight", "bmi",
                  "g", "gs", "pass_cmp", "pass_att",
                  "pass_inc", "pass_cmp_perc", "pass_yds",
                  "pass_td", "pass_int", "pass_int_td",
                  "pass_td_perc", "pass_int_perc", "pass_rating",
                  "pass_sacked", "pass_sacked_yds", "pass_sacked_perc",
                  "pass_yds_per_att", "pass_adj_yds_per_att", "rush_yds",
                  "rec_yds", "scoring", "fantasy_points", "sacks",
                  "seasons"]
MAX_FF_POINTS_VAL = 35850
BASE_URL = 'https://www.pro-football-reference.com'
