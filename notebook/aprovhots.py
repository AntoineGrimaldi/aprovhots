import glob, ast, os
import pandas as pd
from tqdm import tqdm
import numpy as np


def csv2npy(path):
    os.chdir(path)
    list_csv = glob.glob('*.csv')
    list_npy = glob.glob('*.npy')
    print(f'list of all .csv files : \n {list_csv} \n')

    index = 'xytp'

    for number, name in enumerate(list_csv):
        if name[:-4]+'.npy' not in list_npy:
            print(f'loading: {name}')
            df = pd.read_csv(path+name)
            events = None
            index = 0
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                text = row['.events']
                text = text.replace('[', '[{').replace(']', '}]').replace('\n', ', ').replace(' ts: ,', '').replace(", x", "}, {x").replace("x", "'x'").replace(" y", " 'y'").replace(" secs", " 'secs'").replace(" nsecs", " 'nsecs'").replace(" polarity", " 'polarity'")
                mydict = ast.literal_eval(text)
                x = pd.DataFrame.from_dict(mydict)['x'].values
                y = pd.DataFrame.from_dict(mydict)['y'].values
                t = pd.DataFrame.from_dict(mydict)['secs'].values*1e6+pd.DataFrame.from_dict(mydict)['nsecs'].values*1e-3
                p = pd.DataFrame.from_dict(mydict)['polarity'].values
                if events is not None:
                    events = np.vstack((events, np.array([x,y,t,p]).T))
                else:
                    events = np.array([x,y,t,p]).T
            np.save(path+name[:-4], events)
        else: print(f'{name} was already loaded saved as .npy file')