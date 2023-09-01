import os

import pandas as pd

df = pd.read_csv('../data/big-vul_dataset/train.csv')
# 假设你的 DataFrame 叫做 df，CWE Type 列的名称是 'CWE Type'
# 统计 CWE Type 列中每个类型的出现频率
cwe_type_counts = df['CWE ID'].value_counts()

# 获取出现频率前10的 CWE Type
top_10_cwe_types = cwe_type_counts.head(10).index.tolist()

for file in os.listdir('../data/big-vul_dataset'):
    df_tofilter = pd.read_csv('../data/big-vul_dataset/'+file)
    df_tofilter = df_tofilter[df_tofilter['CWE ID'].isin(top_10_cwe_types)]
    df_tofilter['CWE ID Label'] = pd.factorize(df_tofilter['CWE ID'])[0]
    df_tofilter.to_csv('../data/big-vul_dataset/cwe_'+file)
# 现在，filtered_df 包含了出现频率前10的 CWE Type 的记录
