"""
In this file we build the RotoWire dataset so that it can be used in OpenNMT
and it can be used by our proposed hierarchical model.

All tables are represented as a sequence, where every ENT_SIZE tokens are one
entity, so that seq.view(ENT_SIZE, -1) separates all entities.
Each entity starts with <ent> token, for learning entity repr

A lot of this file comes from previous work on this dataset:
https://github.com/ratishsp/data2text-plan-py/blob/master/scripts/create_dataset.py
"""

from more_itertools import collapse

import pkg_resources
import json, os, re
import argparse

# OpenNMT has a fancy pipe
DELIM = "|"
ENT = "<ent>|"
# I manually checked and there are at most 24 elements in an entity
ENT_SIZE = 24

bs_keys = ['START_POSITION', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
           'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
           'AST', 'TO', 'STL', 'BLK', 'PF', 'FIRST_NAME', 'SECOND_NAME']

ls_keys = ['PTS_QTR1', 'PTS_QTR2', 'PTS_QTR3', 'PTS_QTR4', 'PTS', 'FG_PCT',
           'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'WINS', 'LOSSES', 'CITY',
           'NAME']
ls_keys = [f'TEAM-{key}' for key in ls_keys]
f = open('data/rotowire/config.json')
MAP = json.load(f)
POS_MAP = MAP['POS_MAP']
K_MAP = MAP["K_MAP"]


def _build_home(entry):
    """The team who hosted the game"""
    records = []
    for key in ls_keys:
        records.append(
            ENT + DELIM.join([entry['home_name'], key, entry['home_line'][key].replace(' ', '_'), 'Home']) + DELIM)
    return records


def _build_vis(entry):
    """The visiting team"""
    records = []
    for key in ls_keys:
        # Contrary to previous work, home is now a unique token at the end
        records.append(
            ENT + DELIM.join([entry['vis_name'], key, entry['vis_line'][key].replace(' ', '_'), 'Visit']) + DELIM)
    return records


def get_player_idxs(entry):
    # In 4 instances the Clippers play against the Lakers
    # Both are from LA... We simply devide in half the players
    # In all 4, there are 26 players so we return 13-25 & 0-12
    # as it is always visiting first and home second.
    if entry['home_city'] == entry['vis_city']:
        assert entry['home_city'] == 'Los Angeles'
        return ([str(idx) for idx in range(13, 26)],
                [str(idx) for idx in range(13)])

    nplayers = len(entry['box_score']['PTS'])
    home_players, vis_players = list(), list()
    for i in range(nplayers):
        player_city = entry['box_score']['TEAM_CITY'][str(i)]
        if player_city == entry['home_city']:
            home_players.append(str(i))
        else:
            vis_players.append(str(i))
    return home_players, vis_players


def box_preprocess(entry, remove_na=False):
    home_players, vis_players = get_player_idxs(entry)

    all_entities = list()  # will contain all records of the input table
    score = list()  # will contain score of each k-v pair
    for is_home, player_idxs in enumerate([vis_players, home_players]):
        for player_idx in player_idxs:
            for key in bs_keys:
                player = []
                # player = [DELIM.join(['<ent>', '<ent>'])]
                tmp = entry['box_score'][key]
                val = tmp[player_idx]
                entity = entry['box_score']['PLAYER_NAME'][player_idx]
                if remove_na and val == 'N/A': continue
                m_val = max([0 if item == 'N/A' else int(item) for item in tmp.values()]) if K_MAP[key] else None
                is_home_str = 'Home' if is_home else 'Visit'
                score.append(get_scores(entity, key, val, m_val,
                                        ' '.join(entry['summary']).split('.')[:-1]))
                player.extend([entity, key, val.replace(' ', '_'), is_home_str])
                all_entities.append(ENT + DELIM.join(player) + DELIM)
            # We pad the entity to size ENT_SIZE with OpenNMT <blank> token
            # player.extend([DELIM.join(['<blank>', '<blank>'])] * (ENT_SIZE - len(player)))
            # all_entities.append(player)
    all_entities.extend(_build_home(entry))
    all_entities.extend(_build_vis(entry))
    score.extend([-1] * (len(all_entities) - len(score)))
    return list(collapse(all_entities)), score


def get_scores(entity, key, value, m_val, summary):
    """
    :param entity:
    :param key:
    :param value:
    :param m_val:
    :param summary:
    :return: score which ranges from 0 to 1
    """

    def _check(a, s):
        for i in a:
            if i in s:
                return True
        return False

    if value == 'N/A':
        return 0
    score = 0
    ent_list = entity.split(' ') + [entity]
    for n, sentence in enumerate(summary):
        base = (1 - n / len(summary))
        for ent in ent_list:
            if ent in sentence or ' he ' in sentence:
                # calculate the score
                if K_MAP[key]:
                    if _check(K_MAP[key], sentence):
                        return base * 0.5 + 0.5 if value in sentence else base * 0.4 + 0.5
                    elif score == 0:
                        score = base * 0.3 + 0.2
                else:
                    if key == 'START_POSITION' and _check(POS_MAP[value], sentence) or value in sentence:
                        return base * 0.5 + 0.5
                    elif score == 0:
                        score = base * 0.3 + 0.2
    # if key is not matched , will return the first score
    if score:
        return score
    else:
        return (int(value) / (m_val + 1e-3)) * 0.2 if K_MAP[key] else 0.05  # ent is not matched


def _clean_summary(summary, tokens):
    """
    In here, we slightly help the copy mechanism
    When we built the source sequence, we took all multi-words value
    and repalaced spaces by underscores. We replace those as well in
    the summaries, so that the copy mechanism knows it was a copy.
    It only happens with city names like "Los Angeles".
    """
    summary = ' '.join(summary)
    for token in tokens:
        val = token.split(DELIM)[0]
        if '_' in val:
            val_no_underscore = val.replace('_', ' ')
            summary = summary.replace(val_no_underscore, val)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', dest='folder', default='./', required=False,
                        help='Save the preprocessed dataset to this folder')
    parser.add_argument('--keep-na', dest='keep_na', default=True, action='store_true',
                        help='Activate to keep NA in the dataset')

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print('Creating folder to store preprocessed dataset at:')
        print(args.folder)
        os.mkdir(args.folder)

    for setname in ['train', 'valid', 'test']:
        filename = f'rotowire/{setname}.json'
        filename = pkg_resources.resource_filename(__name__, filename)
        with open(filename, encoding='utf8', mode='r') as f:
            data = json.load(f)

        input_filename = os.path.join(args.folder, f'{setname}_src.txt')
        output_filename = os.path.join(args.folder, f'{setname}_tgt.txt')
        lable_filename = os.path.join(args.folder, f'{setname}_label.txt')
        with open(input_filename, mode='w', encoding='utf8') as inputf:
            with open(output_filename, mode='w', encoding='utf8') as outputf:
                with open(lable_filename, mode='w', encoding='utf8') as labelf:
                    for entry in data:
                        input, score = box_preprocess(entry)
                        inputf.write(''.join(input) + '\n')
                        summary = _clean_summary(entry['summary'], input)
                        outputf.write(summary + '\n')
                        score = list(map(str, score))
                        labelf.write(' '.join(score) + '\n')
