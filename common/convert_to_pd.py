import numpy as np
import pandas as pd
from tqdm import tqdm

RECORD = "<record>"


def convert_to(t):
    src = f'train_{t}.txt'
    with open(src, 'r') as f:
        lines = f.readlines()
        _x = [line.strip('\n').split('<ent>')[1:] for line in lines]
        x_team = [xx[-30:] for xx in _x]
        x = [xx[:-30] for xx in _x]
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
                #             tmp['Home'] = '<record>' + ' ' + ' <record> '.join(tmp['Home'])
                tmp['Visit'] = '<record>' + ' ' + ' <record> '.join(tmp['Visit'])
                team.append(tmp)
        # 先做一个字典，记录下每个球队对应的key最大值
        # dic={,'MIN': [],'PTS': [],'FGM': [],'FGA': [],
        #      'FG_PCT': [],'FG3M': [],'FG3A': [],'FG3_PCT': [],
        #      'FTM': [],'FTA': [],'FT_PCT': [],'OREB': [],'DREB': [],
        #      'REB': [],'AST': [],'TO': [],'STL': [],'BLK': [],'PF': [],}
        dic = {}
        for e in x[0]:
            e = e.split('|')
            if e[1] in ['FIRST_NAME', 'SECOND_NAME', 'START_POSITION']: continue
            dic[f'{e[1]}_{e[3]}'] = [0 for _ in range(len(x))]

        for i, xx in enumerate(x):
            for e in xx:
                e = e.split('|')
                if e[1] in ['FIRST_NAME', 'SECOND_NAME', 'START_POSITION'] or e[2] == 'N/A':
                    continue
                dic[f'{e[1]}_{e[3]}'][i] = max(dic[f'{e[1]}_{e[3]}'][i], int(e[2]))
        n = 0
        entity = [[] for _ in range(len(x))]
        record_num = 20
        # 对value进行归一化,并生成entity,N/A -> 0
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
                #             e[2] = str(round(e[2], 2))
                else:
                    e[2] = 0.0
                # add <record>
                if n % record_num == 0 and n != 0:
                    n = 0
                    if tmp: entity[i].append(tmp)
                    tmp = []
                tmp.append(e[:-1])
                n += 1
            entity[i].append(tmp)

        df = pd.DataFrame(index=list(range(len(entity * 30))),
                          columns=['Name', 'vorh', 'worl', 'START_POSITION', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
                                   'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
                                   'TO', 'STL', 'BLK', 'PF'])
        r = 0  # 行
        for i, tab in tqdm(enumerate(entity), desc='创建csv数据表', total=len(entity), colour='green'):
            for e in tab:
                for j, ee in enumerate(e):
                    if j == 0:
                        # name
                        df['Name'][r] = ee[0]
                        # 是否主场 1 主场 0 客场
                        df['vorh'][r] = 1 if ee[-1] == 'Home' else 0
                        # 是否 win
                        df['worl'][r] = 1 if team[i]['Win'] == ee[-1] else 0
                    df[ee[1]][r] = ee[2]
                r += 1
        df = df.loc[:r - 1]
        tgt = f'{t}_label.txt'
        with open(tgt, 'r') as f:
            lines = f.readlines()
            y = [line.strip('\n').split(' ') for line in lines]
            y = [yy[:-30] for yy in y]
        # cut label to 20
        # remove list
        r = [22 * i for i in range(1, 31)] + [22 * i - 1 for i in range(1, 31)]
        for i, yy in enumerate(y):
            tmp = []
            for j, _y in enumerate(yy):
                if (j + 1) not in r:
                    tmp.append(float(_y))
            y[i] = tmp
        a = np.array(y[0]).reshape(-1, 20)
        for i, yy in enumerate(y):
            if i == 0: continue
            a = np.vstack((a, np.array(yy).reshape(-1, 20)))

        # save
        np.save('valid.npy', a)
        df.to_csv(f'rotowire_{t}.csv', index=False)
    pass




if __name__ == '__main__':
    for t in ['train','valid','test']:
        convert_to(t)