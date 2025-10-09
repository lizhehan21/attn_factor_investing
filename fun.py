import os
import glob
import random
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch import amp

from tqdm import tqdm

# ===================== #
#   1. 数据读取与整理   #
# ===================== #

def load_factors(factor_dir, factor_names, dates, stock_codes):
    dfs = []
    for fname in factor_names:
        df = pd.read_csv(f'{factor_dir}/{fname}.csv', index_col=0)
        df = df.reindex(index=dates, columns=stock_codes)
        df = df.stack().rename(fname)
        dfs.append(df)
    factors = pd.concat(dfs, axis=1)
    factors.index.names = ['date', 'stock_id']
    return factors.reset_index()

def standardize_columns(columns):
    columns = pd.Index(columns)  # 兼容传入list或Index
    columns = (columns
        .str.replace('.SZ', '.XSHE', regex=False)
        .str.replace('.SH', '.XSHG', regex=False)
    )
    return columns

def load_factors_fast(factor_dir, factor_names, dates, stock_codes):
    idx = pd.MultiIndex.from_product([dates, stock_codes], names=['date', 'stock_id'])
    dfs = []
    for fname in factor_names:
        df = pd.read_csv(os.path.join(factor_dir, f'{fname}.csv'), index_col=0)
        df.index = df.index.astype(str)
        df.columns = standardize_columns(df.columns.astype(str))
        s = df.stack().reindex(idx).rename(fname)
        dfs.append(s)
    all_factors = pd.concat(dfs, axis=1)
    all_factors.index.names = ['date', 'stock_id']
    all_factors = all_factors.reset_index()
    return all_factors

    
def calc_label(close_df, lookforward=10):
    # close_df: 行是日期，列是股票代码
    df = close_df.copy()
    df = df.sort_index()
    ret = (df.shift(-lookforward-1) / df.shift(-1) - 1)
    label = ret.stack().rename('label')
    label.index.names = ['date', 'stock_id']
    return label.reset_index()

def merge_all_optimized(factor_dir, factor_names, close_path,lookforward=10):
    close = pd.read_pickle(close_path)
    dates = close.index.astype(str)
    stock_codes = close.columns.astype(str)

    factors = load_factors_fast(factor_dir, factor_names, dates, stock_codes)
    label = calc_label(close, lookforward=lookforward)
    factors['date'] = factors['date'].astype(str)
    label['date'] = label['date'].astype(str)
    df = pd.merge(factors, label, on=['date', 'stock_id'], how='left')
    df = df.sort_values(['stock_id', 'date']).reset_index(drop=True)
    return df




# ===================== #
#   2. 数据预处理       #
# ===================== #
def mad_based_outlier(series, n=5.0):
    median = series.median()
    mad = np.median(np.abs(series - median))
    upper = median + n * mad
    lower = median - n * mad
    return np.clip(series, lower, upper)

def preprocess_factors(df, factor_cols):
    for col in factor_cols:
        df[col] = mad_based_outlier(df[col], n=5)
    scaler = StandardScaler()
    df[factor_cols] = scaler.fit_transform(df[factor_cols].fillna(0))
    df[factor_cols] = df[factor_cols].fillna(0)
    return df

def cross_section_preprocess_factors(df, factor_cols):
    df = df.copy()
    for factor in factor_cols:
        # 先按date分组，对每一天做MAD缩尾
        df[factor] = df.groupby('date')[factor].transform(lambda x: mad_based_outlier(x, n=5))
        # 再按date分组，对每一天做z-score标准化，NaN填0
        df[factor] = df.groupby('date')[factor].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() and x.std() != 0 else 0
        )
        #df[factor] = df[factor].fillna(0)
    return df

def preprocess_label(df):
    # 删除label缺失
    df = df[~df['label'].isna()].copy()
    # 截面排序标准化 
    df['label_rank'] = df.groupby('date')['label'].rank(pct=True)
    return df

# ===================== #
#   3. 时序样本构建     #
# ===================== #
class StockSequenceDatasetFast(Dataset):
    def __init__(
        self,
        df,
        factor_cols,
        seq_len=21,
        return_meta=False,
        float_dtype=np.float32,
        impute=True,          # 是否在窗口内按均值填补
        tol_ratio=0.5         # 每个因子在窗口内允许的缺失比例阈值（<= 阈值则填补，> 阈值则丢弃窗口）
    ):
        self.sequences = []
        self.targets = []
        self.meta = []
        self.return_meta = return_meta
        self.factor_cols = factor_cols
        self.seq_len = seq_len
        self.float_dtype = float_dtype
        self.impute = impute
        self.tol_ratio = tol_ratio

        # 预处理一次日期为统一字符串，避免重复转换
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 按股票分组
        for sid, subdf in df.groupby('stock_id', group_keys=False):
            subdf = subdf.sort_values('date')

            if len(subdf) < seq_len:
                continue

            X_all = subdf[factor_cols].to_numpy(dtype=float_dtype, copy=True)
            y_all = subdf['label_rank'].to_numpy(dtype=float_dtype, copy=True)
            dates_all = subdf['date'].to_numpy()

            # 滑窗
            for i in range(seq_len - 1, len(subdf)):
                x_seq = X_all[i - seq_len + 1:i + 1].copy()  # [seq_len, n_factor]
                y_seq = y_all[i]
                if np.isnan(y_seq):
                    continue  # 目标必须非NaN

                # 逐列统计缺失
                nan_mask = np.isnan(x_seq)  # [seq_len, n_factor]
                if not nan_mask.any():
                    # 无缺失，直接收集
                    self.sequences.append(x_seq.astype(float_dtype, copy=False))
                    self.targets.append(y_seq)
                    if return_meta:
                        self.meta.append({'date': dates_all[i], 'stock_id': sid})
                    continue

                # 有缺失：按列处理
                n_missing_per_col = nan_mask.sum(axis=0)                   # [n_factor]
                allow_missing_max = int(np.floor(seq_len * tol_ratio))     # 阈值（<= 该阈值可填补）
                # 如果任何一列超过阈值，丢弃窗口
                if (n_missing_per_col > allow_missing_max).any():
                    continue

                if self.impute:
                    # 按窗口内该因子非NaN均值进行填补
                    # 对每列独立求均值
                    col_means = np.nanmean(x_seq, axis=0)                 # [n_factor]
                    # 将均值广播到缺失位置
                    # 避免均值仍为nan的极端情况（全NaN会被上面阈值挡掉，这里只是稳妥处理）
                    col_means = np.where(np.isnan(col_means), 0.0, col_means)
                    # 用列均值填补
                    rows_idx, cols_idx = np.where(nan_mask)
                    x_seq[rows_idx, cols_idx] = col_means[cols_idx]
                else:
                    # 如果不允许填补，但窗口存在NaN，则丢弃
                    continue

                # 收集
                self.sequences.append(x_seq.astype(float_dtype, copy=False))
                self.targets.append(y_seq)
                if return_meta:
                    self.meta.append({'date': dates_all[i], 'stock_id': sid})

        self.sequences = np.asarray(self.sequences, dtype=float_dtype)
        self.targets = np.asarray(self.targets, dtype=float_dtype)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx])
        y = torch.tensor(self.targets[idx])
        if self.return_meta:
            m = self.meta[idx]
            return x, y, m['date'], m['stock_id']
        else:
            return x, y


class StockSequenceDatasetFast0(Dataset):
    def __init__(self, df, factor_cols, seq_len=21, return_meta=False, float_dtype=np.float32):
        self.sequences = []
        self.targets = []
        self.meta = []
        gb = df.groupby('stock_id', group_keys=False)
        for sid, subdf in gb:
            subdf = subdf.sort_values('date')
            subdf['date'] = pd.to_datetime(subdf['date']).dt.strftime('%Y-%m-%d')

            X = subdf[factor_cols].values.astype(float_dtype)
            y = subdf['label_rank'].values.astype(float_dtype)
            dates = subdf['date'].values

            if len(subdf) < seq_len:
                continue
            
            for i in range(seq_len - 1, len(subdf)):
                x_seq = X[i - seq_len + 1:i + 1]
                y_seq = y[i]
                # 只保留无nan的序列 ！！！有待调整
                if np.isnan(x_seq).any() or np.isnan(y_seq):
                    continue
                self.sequences.append(x_seq)
                self.targets.append(y_seq)
                if return_meta:
                    self.meta.append({'date': dates[i], 'stock_id': sid})

        self.sequences = np.array(self.sequences, dtype=float_dtype)
        self.targets = np.array(self.targets, dtype=float_dtype)
        self.return_meta = return_meta


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.return_meta:
            meta = self.meta[idx]
            return (
                torch.from_numpy(self.sequences[idx]),
                torch.tensor(self.targets[idx]),
                meta['date'],  # str 类型
                meta['stock_id']
            )
        else:
            return torch.from_numpy(self.sequences[idx]), torch.tensor(self.targets[idx])



# ===================== #
#   4. 模型结构定义     #
# ===================== #


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class LSTMNet(nn.Module): #没有维护amp
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze(-1)

class ALSTMNet(nn.Module): #没有维护amp
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim*2, 1)
    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)
        attn_scores = torch.tanh(self.attn(lstm_out))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_vec = (attn_weights * lstm_out).sum(dim=1)
        concat = torch.cat([attn_vec, hn[-1]], dim=1)
        out = self.fc(concat)
        return out.squeeze(-1)

class TransformerNet(nn.Module):
    def __init__(self, input_dim=42, d_model=64, num_heads=2, num_layers=2, ff_dim=256, dropout=0.1, max_len=128):
        super().__init__()
        
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
                
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'  # 确保ReLU
        )
        
        for _ in range(num_layers):
            encoder_layer.linear1 = nn.Linear(d_model, ff_dim)
            encoder_layer.linear2 = nn.Linear(ff_dim, d_model)
            encoder_layer.dropout1 = nn.Dropout(dropout)
            encoder_layer.dropout2 = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq, input_dim]
        x = self.input_fc(x)  # [batch, seq, 64]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)    # 池化（全局平均）
        out = self.fc_out(pooled)
        return out.squeeze(-1)





# ===================== #
# 5. 划分数据集 有好多版本 # 
# ===================== #
def split_by_ratio(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):#大划分，不滚动
    """
    将DataFrame按比例分为train/val/test三部分（默认顺序分割）
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.sort_values('date')  # 按日期升序分割，确保无数据穿越
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df

def split_by_year(df, train_years, val_years, test_years): #大划分，不滚动
    df['year'] = df['date'].astype(str).str[:4]  # 假设date为字符串'YYYY-MM-DD'
    train_df = df[df['year'].isin([str(y) for y in train_years])].copy()
    val_df = df[df['year'].isin([str(y) for y in val_years])].copy()
    test_df = df[df['year'].isin([str(y) for y in test_years])].copy()
    df = df.drop(columns='year')
    return train_df, val_df, test_df

def split_by_date(df, train_range, val_range, test_range): #大划分，不滚动
    """
    df: 包含 'date' 列，类型为 str 或 datetime
    train_range, val_range, test_range: (start_date, end_date)，格式'YYYY-MM-DD'
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    train_start, train_end = pd.to_datetime(train_range[0]), pd.to_datetime(train_range[1])
    val_start, val_end = pd.to_datetime(val_range[0]), pd.to_datetime(val_range[1])
    test_start, test_end = pd.to_datetime(test_range[0]), pd.to_datetime(test_range[1])

    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
    return train_df, val_df, test_df

def get_first_fridays(dates):
    """获取所有月第一个周五的日期"""
    
    all_dates = pd.Series(dates.unique()).sort_values()
    all_dates = pd.to_datetime(all_dates)
    months = all_dates.dt.to_period('M').unique()
    fridays = []
    for m in months:
        month_dates = all_dates[all_dates.dt.to_period('M') == m]
        friday = month_dates[month_dates.dt.weekday == 4]  # 0=Mon,...,4=Fri
        if not friday.empty:
            fridays.append(friday.iloc[0])
    return fridays

def get_first_days(dates):
    """获取所有月第一天的日期"""
    all_dates = pd.Series(dates.unique()).sort_values()
    all_dates = pd.to_datetime(all_dates)
    # 转为年月Period，再转回日期就是每月第一天
    months = all_dates.dt.to_period('M').unique()
    first_days = [pd.Period(m).to_timestamp() for m in months]
    return first_days

def get_last_days(dates):
    """获取所有月最后一天的日期"""
    all_dates = pd.Series(dates.unique()).sort_values()
    all_dates = pd.to_datetime(all_dates)
    # 转为年月Period，再转回日期就是每月最后一天
    months = all_dates.dt.to_period('M').unique()
    last_days = [pd.Period(m).to_timestamp(how='end') for m in months]
    return last_days


def split_rolling_with_valid(df, lookback=252, valid_days=42):
    # 滚动训练
    df = df.sort_values('date')
    last_days = get_last_days(df['date']) #可以换时间点
    split_results = []
    for i in range(len(last_days) - 1):
        train_end = last_days[i]
        test_end = last_days[i+1]
        date_arr = df[df['date'] <= train_end]['date'].unique()
        if len(date_arr) < lookback:
            continue
        train_start = date_arr[-lookback]
        train_end_true = date_arr[-11]
        if valid_days > 0:
            valid_start = date_arr[-valid_days-11]
            split_results.append({
                "train_start": str(train_start)[:10],
                "train_end": str((valid_start - pd.Timedelta(days=1)))[:10],
                "valid_start": str(valid_start)[:10],
                "valid_end": str(train_end_true)[:10],
                "test_start": str((train_end + pd.Timedelta(days=1)))[:10],
                "test_end": str(test_end)[:10],
            })
        else:
            split_results.append({
                "train_start": str(train_start)[:10],
                "train_end": str(train_end_true)[:10],
                "valid_start": None,
                "valid_end": None,
                "test_start": str((train_end + pd.Timedelta(days=1)))[:10],
                "test_end": str(test_end)[:10],
            })
    return split_results

# ===================== #
#   6. 训练主循环       #
# ===================== #

#  6.1 Loss function

class ICLoss(nn.Module):
    def __init__(self, method='pearson'):
        """
        method: 'pearson' 计算线性IC, 'spearman' 计算秩相关(RankIC)
        """
        super().__init__()
        assert method in ('pearson', 'spearman')
        self.method = method

    def forward(self, y_pred, y_true):
        """
        y_pred, y_true: [batch_size] 或 [N]
        """
        if self.method == 'spearman':
            # Rank transform
            y_pred = y_pred.argsort().argsort().float()
            y_true = y_true.argsort().argsort().float()
        # 皮尔森相关系数计算
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        # 加个eps防止分母为零
        eps = 1e-8
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) ) * torch.sqrt(torch.sum(vy ** 2) ))
        # Loss 通常用 (1-corr) 或 -corr（相关性越大越好）
        loss = 1 - corr
        return loss

class RankICLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        y_pred, y_true: shape [batch_size] or [N]
        计算Spearman秩相关系数，并以1-秩相关作为loss
        """
        # 转换为秩(rank)
        y_pred_rank = y_pred.argsort().argsort().float()
        y_true_rank = y_true.argsort().argsort().float()
        # 去均值
        vx = y_pred_rank - y_pred_rank.mean()
        vy = y_true_rank - y_true_rank.mean()
        eps = 1e-8
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2) ))
        loss = 1 - corr
        return loss

class WeightedMSELoss(nn.Module):
    """
    加权MSE损失函数，权重为e^(真实标签y)
    公式: loss = mean(e^y * (y_pred - y_true)^2)
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        squared_error = torch.pow(y_pred - y_true, 2)
        weights = torch.exp(y_true)
        weighted_loss = weights * squared_error
        return torch.mean(weighted_loss)
    
class CustomSigmoidWeightedMSELoss(nn.Module):
    """
    使用 custom_sigmoid(y_true) 作为权重的加权 MSE：
    loss = mean(custom_sigmoid(y_true) * (y_pred - y_true)^2)
    """
    def __init__(self):
        super(CustomSigmoidWeightedMSELoss, self).__init__()

    def custom_sigmoid(self, x): #可以改
        z = 15 * (x - 0.5)
        y = 1 / (1 + torch.exp(-z))
        y_fixed = 0.5 + 0.5 * y
        return y_fixed

    def forward(self, y_pred, y_true):
        squared_error = torch.pow(y_pred - y_true, 2)
        weights = self.custom_sigmoid(y_true)
        weighted_loss = weights * squared_error
        return torch.mean(weighted_loss)
    


def train_model(
    model,
    model_type,
    train_loader,
    val_loader,
    test_loader,
    n_epochs=100,
    lr=1e-4,
    patience=20,
    seed=42,
    split_num=0,
    criterion=CustomSigmoidWeightedMSELoss(),
    use_amp=True  
):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)    # <--- 新增
    best_val = float('inf')
    best_weights = None
    no_improve = 0

    save_dir = os.path.abspath(os.path.join('..', 'my_results', model_type.lower()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建文件夹: {save_dir}")

    for epoch in range(n_epochs):
        
        model.train()
        tloss = 0
        with tqdm(train_loader, disable=True,desc=f"[{model_type}] Epoch {epoch+1}/{n_epochs}", ncols=100) as pbar:
            for xb, yb, _, _ in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                

                loss.backward()
                optimizer.step()
                tloss += loss.item() * len(xb)
                pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

        tloss = tloss / len(train_loader.dataset)
        
        if len(val_loader.dataset) == 0: #可能会有逻辑漏洞，只是对无验证集报错的补漏
            print("该分割下没有验证集，用train loss早停")
            vloss = tloss
        else:
            model.eval()
            vloss = 0
            with torch.no_grad():
                for xb, yb, _, _ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    with amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        pred = model(xb)
                        loss = criterion(pred, yb)
                    vloss += loss.item() * len(xb)
            vloss = vloss / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}: TrainLoss={tloss:.4f}  ValLoss={vloss:.4f}')

        if vloss < best_val:
            best_val = vloss
            best_weights = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early Stopping.")
                break
    model.load_state_dict(best_weights)
    model_save_path = os.path.join(save_dir, f"best_model_split_{split_num}.pth")
    torch.save(model.state_dict(), model_save_path)
    return model

def train_model_no_val(
    model,
    model_type,
    train_loader,
    val_loader,
    test_loader,
    n_epochs=100,
    lr=1e-4,
    patience=20,
    seed=42,
    split_num=0,
    criterion=CustomSigmoidWeightedMSELoss(),
    use_amp=True
):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion
    optimizer1 = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float('inf')
    #best_weights = None
    #no_improve = 0

    save_dir = os.path.abspath(os.path.join('..', 'my_results', model_type.lower()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建文件夹: {save_dir}")

    for epoch in range(n_epochs):
        model.train()
        tloss = 0
        if epoch<=3:
            optimizer = optimizer2
        else:
            optimizer = optimizer1
        with tqdm(train_loader, disable=True,desc=f"[{model_type}] Epoch {epoch+1}/{n_epochs}", ncols=100) as pbar:
            for xb, yb, _, _ in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with amp.autocast("cuda", dtype=torch.bfloat16,enabled=use_amp):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    #print("autocast启用?:", torch.is_autocast_enabled())
                    #print("输出dtype:", out.dtype)            # 预期: torch.float16 或 torch.bfloat16
                    #print("显存(MB):", torch.cuda.memory_allocated()/1024/1024)
                #if use_amp:
                    #scaler.scale(loss).backward()
                    #scaler.step(optimizer)
                    #scaler.update()
                    #print("当前GradScaler缩放因子:", scaler.get_scale())
                    
                #else:
                loss.backward()
                optimizer.step()
                     # 更新参数
                tloss += loss.item() * len(xb)
                pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        tloss = tloss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}: TrainLoss={tloss:.4f}')

    #torch.save(model.state_dict(), model_save_path)
    return model

def predict_as_factor(model, test_loader,use_amp=True):
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_dates, all_sids = [], [], []
    with torch.no_grad():
        for xb, _, dateb, sidb in test_loader:
            xb = xb.to(device)
            with amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(xb).cpu().numpy()
            all_preds.append(pred)
            all_dates += list(dateb)
            all_sids += list(sidb)
    all_preds = np.concatenate(all_preds)
    df = pd.DataFrame({'date': all_dates, 'stock_id': all_sids, 'pred': all_preds})
    df_wide = df.pivot(index='date', columns='stock_id', values='pred')
    df_wide = df_wide.sort_index()  # 行按日期排序
    return df_wide









    
