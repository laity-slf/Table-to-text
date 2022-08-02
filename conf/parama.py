import argparse
import os
from datetime import datetime

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MAP_K = {
    'AST': 'Player assists',
    'BLK': 'Player blocks ',
    'DREB': 'Player defensive rebounds',
    'FG3A': 'Player 3-pointers attempted',
    'FG3M': 'Player 3-pointers made',
    'FG3_PCT': ' Player 3-pointer percentage',
    'FGA': 'Player field goals attempted',
    'FGM': 'Player field goals made',
    'FG_PCT': 'Player field goal percentage',
    'FTA': 'Player free throws attempted',
    'FTM': 'Player free throws made',
    'FT_PCT': 'Player free throw percentage',
    'MIN': 'Player minutes played',
    'OREB': 'Player offensive rebounds',
    'PF': 'Player personal fouls',
    'PTS': 'Player points',
    'REB': 'Player total rebounds',
    'START_POSITION': 'Player position',
    'STL': 'Player steals',
    'TO': 'Player turnovers'
}

