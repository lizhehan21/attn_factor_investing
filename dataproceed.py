# main.py
# 预先计算了ret rank
import os
import torch
import time
from model_utils import *

import torch
import time
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


factor_dir = './alpha_transformer'
factor_names = (
        [f'factor_pyt_{i:03d}' for i in range(1, 31)] +  # 001~030
        [f'L2_fac{i}' for i in range(1, 21)]             # 1~20
    )
print("因子名：", factor_names)
close_path = './close.pkl'
seq_len = 21
lookforward = 10

# 1. 数据加载
df = merge_all_optimized(factor_dir, factor_names, close_path) #######
print("merged")
factor_cols = factor_names
print(df)
# 2. 预处理
df_preprocess_factors = cross_section_preprocess_factors(df, factor_cols)
#df_preprocess_label = preprocess_label(df_preprocess_factors)
print("preproceed")
factor_cols = factor_names + ['close']

df_preprocess_factors.to_csv('fully_proceed_df_ori_nan.csv')
print("处理数据已保存: fully_proceed_df_ori.pkl")

#df_preprocess_label[factor_cols] = df_preprocess_label[factor_cols].astype(np.float16)

#df_preprocess_label.to_pickle('fully_proceed_df_16.pkl')
#print("处理数据已保存: fully_proceed_df_16.pkl")

#df_preprocess_label.to_csv('fully_proceed_df.csv')

# 保存label_rank横截面宽表
#pivot_label_rank = df_preprocess_label.pivot(index='date', columns='stock_id', values='label_rank')
#pivot_label_rank.to_pickle(f'{lookforward}_label_rank_wide.pkl')
#print(f"横截面label_rank宽表已保存: {lookforward}_label_rank_wide.pkl")