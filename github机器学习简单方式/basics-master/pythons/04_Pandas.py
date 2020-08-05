'''
@File       :   04_Pandas.py
@Author     :   Jiang Fubang
@Time       :   2020/7/27 15:02
@Version    :   1.0
@Contact    :   luckybang@163.com
@Dect       :   None
'''

import numpy as np
import pandas as pd

np.random.seed(seed=1234)

df = pd.read_csv('../data/titanic.csv', header=0)

# print(df)
# print(df.describe())
# print(df['age'].hist())
# 去重
# print(df['embarked'].unique())
# 查看某一列
# print(df['name'].head())
# 按某列筛选
# print(df[df['sex']=='female'].head())
# 按某列排序
# print(df.sort_values('age', ascending=False).head())
# 分组
# survived_group = df.groupby('survived')
# print(survived_group.mean())
# 索引取值
# print(df.iloc[0, :])
# print(df.iloc[0, 1])
# 具有至少一个null值的行
# print(df[pd.isnull(df).any(axis=1)].head())
# 删除nan的行, 重置索引
df = df.dropna()
df = df.reset_index()
# print(df)
# 删除多列
df = df.drop(['name', 'cabin', 'ticket'], axis=1)
# print(df.head())
# 映射特征值
df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
df['embarked'] = df['embarked'].dropna().map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# print(df.head())
# 特征工程
def get_family_size(sibsp, parch):
    family_size = sibsp + parch
    return family_size
df['family_size'] = df[['sibsp', 'parch']].apply(lambda x: get_family_size(x['sibsp'], x['parch']), axis=1)
# print(df.head())
# 重新整理header
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'family_size', 'fare', 'embarked', 'survived']]
df.to_csv('../data/processed_titanic.csv', index=False)


df = pd.read_excel("../data/apply案例数据.xlsx", sheet_name='成绩表')
print(df)
max_score = df.groupby(by="姓名")["综合成绩"].apply(max).reset_index()
print(max_score)
min_score = df.groupby(by="姓名")["综合成绩"].apply(min).reset_index()
print(min_score)
score_combine = pd.merge(max_score, min_score, how="inner", on="姓名")
print(score_combine)


order = pd.read_excel('../data/apply案例数据.xlsx', sheet_name="省市销售数据")
print(order.info())