import os
import numpy
import numpy as np
import torch
from tqdm import tqdm

RECORD = "<record>"


def prepare_for_bert(input_file, tokenizer, example_type='train'):
    data_file = f'{input_file}/bert_data/data_{example_type}.npy'
    if os.path.exists(os.path.join(input_file, data_file)):
        entity, team = np.load(data_file, allow_pickle=True)
    else:
        print(f"*****create {example_type}*****")
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
            tmp = {'Home': [], 'Visit': []}
            for i, s in enumerate(x_t):
                s = s.split('|')
                tmp[s[-2]].append(' '.join(s[1:3]))
                if i == len(x_t) - 1:
                    tmp['Home'] = '<record>' + ' ' + ' <record> '.join(tmp[s[-2]])
                    tmp['Visit'] = '<record>' + ' ' + ' <record> '.join(tmp[s[-2]])
                    team.append(tmp)

        # 循环表格得到数据
        entity = [[] for _ in range(len(x))]  # 初始化固定长度
        record_num = 22
        n = 0
        for i, table in enumerate(x):
            tmp = []
            for record in table:
                r = record.split('|')
                if n % record_num == 0:
                    n = 0
                    if tmp:
                        entity[i].append(' '.join(tmp[:-1]))
                    tmp = [r[0] + ' ' + r[-2] + ' ' + RECORD]
                else:
                    r[-2] = RECORD
                tmp.extend(r[1:-1])
                n += 1
            entity[i].append(' '.join(tmp[:-1]))

        # 保存文件为 npy
        np.save(data_file, (entity, team))
    # 读取 label
    tgt = example_type.lower() + '_label.txt'
    with open(os.path.join(input_file, tgt), 'r') as f:
        lines = f.readlines()
        y = [line.strip('\n').split(' ') for line in lines]
        y = [yy[:-30] for yy in y]
    print("finishing creating")

    return convert_example2feature(list(entity), list(team), y, tokenizer)


def convert_example2feature(entity, team, label, tokenizer):
    entity_ids = []
    entity_type_ids = []
    entity_lens = []
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

    entity_ids_padded = []
    entity_type_ids_padded = []
    record_pos = []

    for i in range(len(entity_ids)):
        # item_len = len(entity_item_ids)
        padding = [0] * (max_len - entity_lens[i])
        entity_ids_padded.append(entity_ids[i] + padding)
        entity_type_ids_padded.append(entity_type_ids[i] + padding)
        record_p = (np.array(entity_ids[i]+padding) == tokenizer.additional_special_tokens_ids[0])
        record_pos.append(record_p)

    # 转换为tensor，注意类型为 long
    entity_ids_padded = torch.tensor(entity_ids_padded, dtype=torch.long)
    entity_type_ids_padded = torch.tensor(entity_type_ids_padded, dtype=torch.long)
    record_pos = torch.tensor(record_pos)
    entity_mask = (entity_ids_padded > 0)
    # resize label
    y = []
    for yy in label:
        yy = list(map(float, yy))
        y.extend(np.array(yy).reshape(-1, 22))
    y = torch.tensor(np.array(y), dtype=torch.float)
    return entity_ids_padded, entity_type_ids_padded, entity_mask, y, record_pos


if __name__ == '__main__':
    dic = {}
    for pred_type in ['train', 'valid', 'test']:
        dic[pred_type] = prepare_for_bert('../data', example_type=pred_type)
    print(1)
