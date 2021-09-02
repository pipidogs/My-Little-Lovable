# -*- coding: utf-8 -*-

"""
    @Name:         csv_to_tsv
    @Date:         2021/8/23
    @Description:  将csv文件转化为tsv文件
"""
import numpy as np

import pandas as pd

filename = '此处文件名'
train_set = pd.read_csv(f"data/{filename}.csv", sep=',', header=0)
train_set.dropna(inplace=True)
train_set[['label']] = train_set[['label']].astype(np.int)
# csv与tsv的列名对应关系
train_df_bert = pd.DataFrame({
    'label': train_set['label'],
    'text1': train_set['sentence1'].replace(r'\n', ' ', regex=True),
    'text2': train_set['sentence2'].replace(r'\n', ' ', regex=True)

})
train_df_bert.to_csv(f'data/{filename}_pair.tsv', sep='\t', index=False, header=True)

if __name__ == '__main__':
    pass
