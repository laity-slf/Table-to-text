import os
import numpy
import numpy as np
import torch
from tqdm import tqdm
from conf.parama import MAP_K

RECORD = "<record>"


def prepare_for_bert(input_file, output_file, tokenizer, example_type='train', do_norm=False, cut_label=True):
    data_file = f'{output_file}/data_{example_type}.npy'
    print(f"*****create {example_type}*****")
    # 读取 label
    tgt = example_type.lower() + '_label.txt'
    with open(os.path.join(input_file, tgt), 'r') as f:
        lines = f.readlines()
        y = [line.strip('\n').split(' ') for line in lines]
        y = [yy[:-30] for yy in y]

    if cut_label:
        # cut label to 20
        # remove list
        r = [22 * i for i in range(1, 31)] + [22 * i - 1 for i in range(1, 31)]
        for i, yy in enumerate(y):
            tmp = []
            for j, _y in enumerate(yy):
                if (j + 1) not in r:
                    tmp.append(_y)
            y[i] = tmp
    # rotowire 数据
    if os.path.exists(data_file):
        entity, team = np.load(data_file, allow_pickle=True)
    else:
        src = example_type.lower() + '_src.txt'
        with open(os.path.join(input_file, src), 'r') as f:
            lines = f.readlines()
            _x = [line.strip('\n').split('<ent>|')[1:] for line in lines]
            x_team = [xx[-30:] for xx in _x]
            x = [xx[:-30] for xx in _x]
        # TODO 加入team 的信息
        #  先获取Home 和 Visit的队伍信息列表
        team = []
        for x_t in x_team:
            tmp = {'Home': [], 'Visit': [], 'Win': ''}
            horv = 0
            for i, s in enumerate(x_t):
                s = s.split('|')
                tmp[s[-2]].append(' '.join(s[1:3]))
                # 记录胜利的队伍
                if s[1] == 'TEAM-PTS':
                    horv = max(horv, int(s[2]))
                    if horv == int(s[2]):
                        tmp['Win'] = s[3]
                if i == len(x_t) - 1:
                    tmp['Home'] = '<record>' + ' ' + ' <record> '.join(tmp['Home'])
                    tmp['Visit'] = '<record>' + ' ' + ' <record> '.join(tmp['Visit'])
                    team.append(tmp)

        if do_norm:
            entity = example_norm(x, team)
        else:
            record_num = 22
            # 循环表格得到数据
            entity = [[] for _ in range(len(x))]  # 初始化固定长度
            n = 0
            for i, table in enumerate(x):
                tmp = []
                for record in table:
                    r = record.split('|')
                    if r[1] in ['FIRST_NAME', 'SECOND_NAME']:
                        n += 1
                        continue
                    if n % record_num == 0:
                        n = 0
                        if tmp:
                            entity[i].append(' '.join(tmp[:-1]))
                        win = 'won' if team[i]['Win'] == r[-2] else 'lost'
                        tmp = [f'{r[0]} is a {r[-2]} player and he {win} the game' + ' ' + RECORD]
                        # tmp = [r[0] + ' ' + r[-2] + ' ' + RECORD]
                    r[-2] = RECORD
                    r[1] = MAP_K[r[1]] + ' is'
                    tmp.extend(r[1:-1])
                    n += 1
                entity[i].append(' '.join(tmp[:-1]))
        # 保存文件为 npy
        np.save(data_file, (entity, team))
    print(f"finishing creating {example_type} data")
    return convert_example2feature_noteam(list(entity), list(team), y, tokenizer, example_type=example_type,
                                          cut_label=cut_label)


# example  归一化
def example_norm(x, team):
    # 先做一个字典，记录下每个球队对应的key最大值
    # dic={,'MIN': [],'PTS': [],'FGM': [],'FGA': [],
    #      'FG_PCT': [],'FG3M': [],'FG3A': [],'FG3_PCT': [],
    #      'FTM': [],'FTA': [],'FT_PCT': [],'OREB': [],'DREB': [],
    #      'REB': [],'AST': [],'TO': [],'STL': [],'BLK': [],'PF': [],}
    dic = {}
    for e in x[0]:
        e = e.split('|')
        if e[1] in ['FIRST_NAME', 'SECOND_NAME', 'START_POSITION', 'FG_PCT', 'FG3_PCT', 'FT_PCT']:
            continue
        elif f'{e[1]}_{e[3]}' not in dic:
            dic[f'{e[1]}_{e[3]}'] = [0 for _ in range(len(x))]
    # 求每个key 对应的最大值
    # e=['DeJuan Blair', 'SECOND_NAME', 'Blair', 'Home', '']
    for i, xx in enumerate(x):
        for e in xx:
            e = e.split('|')
            if e[1] in ['FIRST_NAME', 'SECOND_NAME', 'START_POSITION', 'FG_PCT', 'FG3_PCT', 'FT_PCT'] or e[2] == 'N/A':
                continue
            dic[f'{e[1]}_{e[3]}'][i] = max(dic[f'{e[1]}_{e[3]}'][i], int(e[2]))
    # 对value进行归一化,并生成entity
    n = 0
    entity = [[] for _ in range(len(x))]
    record_num = 20
    for i, xx in enumerate(x):
        tmp = []
        for e in xx:
            e = e.split('|')
            if e[1] in ['FIRST_NAME', 'SECOND_NAME']:
                continue
            if e[2] != 'N/A':
                if e[1] == 'START_POSITION':
                    e[2] = 1.0 if e[2] in ['F', 'G', 'C'] else 0.0
                elif e[1] == 'PF':
                    e[2] = float(e[2]) / 6
                elif e[1] in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    e[2] = float(e[2]) / 100
                else:
                    e[2] = float(e[2]) / dic[f'{e[1]}_{e[3]}'][i] if dic[f'{e[1]}_{e[3]}'][i] != 0 else 0.0
                # 转为可分词的str，保留前两位
                e[2] = str(round(e[2], 2))
            # add <record>
            if n % record_num == 0:
                n = 0
                if tmp:
                    entity[i].append(' '.join(tmp[:-1]))
                win = 'won' if team[i]['Win'] == e[-2] else 'lost'
                # tmp = [f'{e[0]} is a {e[-2]} player and he {win} the game' + ' ' + RECORD
                tmp = [e[0] + ' ' + e[-2] + ' ' + win + ' ' + RECORD]
            e[-2] = RECORD
            tmp.extend(e[1:-1])
            n += 1
        entity[i].append(' '.join(tmp[:-1]))
    return entity


def convert_example2feature(entity, team, label, tokenizer, example_type=None, cut_label=False):
    entity_ids = []
    entity_type_ids = []
    entity_lens = []
    tab_lens = []
    max_len = 0
    loop = tqdm(enumerate(entity), desc='正在分词', total=len(entity))
    for i, tab in loop:
        for ent in tab:
            k = 'Home' if 'Home' in ent else 'Visit'
            # 分词
            token = tokenizer(ent, team[i][k])
            entity_token_id = token['input_ids']
            entity_type_id = token['token_type_ids']
            entity_len = len(entity_token_id)
            # 记录最大长度
            max_len = max(max_len, entity_len)
            # get id,type_id
            entity_ids.append(entity_token_id)
            entity_type_ids.append(entity_type_id)
            entity_lens.append(entity_len)
        tab_lens.append(len(tab))
    entity_ids_padded = []
    entity_type_ids_padded = []
    record_pos = []

    for i in range(len(entity_ids)):
        # item_len = len(entity_item_ids)
        padding = [0] * (max_len - entity_lens[i])
        entity_ids_padded.append(entity_ids[i] + padding)
        entity_type_ids_padded.append(entity_type_ids[i] + padding)
        record_p = (np.array(entity_ids[i] + padding) == tokenizer.additional_special_tokens_ids[0])
        record_pos.append(record_p)

    # 转换为tensor，注意类型为 long
    entity_ids_padded = torch.tensor(entity_ids_padded, dtype=torch.long)
    entity_type_ids_padded = torch.tensor(entity_type_ids_padded, dtype=torch.long)
    record_pos = torch.tensor(record_pos)
    entity_mask = (entity_ids_padded > 0)
    # resize label
    record_num = 20 if cut_label else 22
    y = []
    for yy in label:
        yy = list(map(float, yy))
        y.extend(np.array(yy).reshape(-1, record_num))
    y = torch.tensor(np.array(y), dtype=torch.float)
    return entity_ids_padded, entity_type_ids_padded, entity_mask, y, record_pos, tab_lens


def convert_example2feature_noteam(entity, team, label, tokenizer, example_type=None, cut_label=False):
    entity_ids = []
    entity_type_ids = []
    entity_lens = []
    tab_lens = []
    max_len = 0
    loop = tqdm(enumerate(entity), desc='正在分词_no_team', total=len(entity))
    for i, tab in loop:
        for ent in tab:
            # 分词
            token = tokenizer(ent)
            entity_token_id = token['input_ids']
            entity_type_id = token['token_type_ids']
            entity_len = len(entity_token_id)
            # 记录最大长度
            max_len = max(max_len, entity_len)
            # get id,type_id
            entity_ids.append(entity_token_id)
            entity_type_ids.append(entity_type_id)
            entity_lens.append(entity_len)
        tab_lens.append(len(tab))
    entity_ids_padded = []
    entity_type_ids_padded = []
    record_pos = []

    for i in range(len(entity_ids)):
        # item_len = len(entity_item_ids)
        padding = [0] * (max_len - entity_lens[i])
        entity_ids_padded.append(entity_ids[i] + padding)
        entity_type_ids_padded.append(entity_type_ids[i] + padding)
        record_p = (np.array(entity_ids[i] + padding) == tokenizer.additional_special_tokens_ids[0])
        record_pos.append(record_p)

    # 转换为tensor，注意类型为 long
    entity_ids_padded = torch.tensor(entity_ids_padded, dtype=torch.long)
    entity_type_ids_padded = torch.tensor(entity_type_ids_padded, dtype=torch.long)
    record_pos = torch.tensor(record_pos)
    entity_mask = (entity_ids_padded > 0)
    # resize label
    y = []
    record_num = 20 if cut_label else 22
    for yy in label:
        yy = list(map(float, yy))
        y.extend(np.array(yy).reshape(-1, record_num))
    y = torch.tensor(np.array(y), dtype=torch.float)
    return entity_ids_padded, entity_type_ids_padded, entity_mask, y, record_pos, tab_lens


if __name__ == '__main__':
    for pred_type in ['train', 'valid', 'test']:
        prepare_for_bert('../data', example_type=pred_type)
    print(1)
