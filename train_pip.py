import os
import torch
import time
from torch.utils.data import DataLoader
from fun import *


def train_predict_pipeline(
    input= './fully_proceed_df_32_10lookforward.csv',
    factor_names='factor_pyt_001',
    model_size=None,
    lookback=252,
    seq_len=21,
    model_type='Transformer',  # 可选 'LSTM'、'ALSTM'、'Transformer'
    batch_size=128,
    n_epochs=30,
    lr=1e-4,
    patience=10,
    valid_days =42,
    pred_factor_name='pred_factor',
    criterion = None,
    use_amp=True,
    date_start=None,           # 例: '2021-01-01'
    date_end=None,             # 例: '2022-12-01'
):
    # 设备自动选择
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"自动检测到训练设备: {device}")

    output_dir = os.path.abspath(os.path.join('.', 'output'))
    os.makedirs(output_dir, exist_ok=True)

    #print("自动识别因子名factor_names：", factor_names)
    factor_cols = factor_names if isinstance(factor_names, list) else [factor_names]

    
    df = pd.read_csv(input)
    
    float_dtype=np.float32
    print(f"模型已选择并加载至设备: {model_type}")
    df['date'] = pd.to_datetime(df['date'])

    start = time.time()

    all_set = StockSequenceDatasetFast(df, factor_cols, seq_len=seq_len, return_meta=True, float_dtype=float_dtype)

    end = time.time()

    print(f"构建 StockSequenceDatasetFast 耗时: {end - start:.2f} 秒")


    splits = split_rolling_with_valid(df, lookback=lookback, valid_days=valid_days)
    
    if date_start and date_end:
        splits = [s for s in splits
                  if (s['test_start'] >= date_start) and
                     (s['test_start'] <= date_end)]
        print(f"筛选区间 [{date_start} ~ {date_end}")

    all_preds_wide = []  # 拼接所有预测结果
    train_t0 = time.time()
    for i, split in tqdm(enumerate(splits), total=len(splits)):
        start = time.time()
        val_loader =None
        train_start, train_end = split['train_start'], split['train_end']
        print("train_start:",train_start, "train_end:",train_end)
        #valid_start, valid_end = split['valid_start'], split['valid_end']
        test_start, test_end = split['test_start'], split['test_end']
        # 2. 提取每个样本的标签日、stock_id
        meta_dates = np.array([s['date'] for s in all_set.meta])  # 或all_set.samples等
        meta_sids  = np.array([s['stock_id'] for s in all_set.meta])
        # 3. 用时间区间筛选索引
        train_idx = np.where((meta_dates >= train_start) & (meta_dates <= train_end))[0]
        #val_idx   = np.where((meta_dates >= valid_start) & (meta_dates <= valid_end))[0]
        test_idx  = np.where((meta_dates >= test_start) & (meta_dates <= test_end))[0]
        # 4. 构造子集
        train_set = torch.utils.data.Subset(all_set, train_idx)
        #val_set   = torch.utils.data.Subset(all_set, val_idx)
        test_set  = torch.utils.data.Subset(all_set, test_idx)



        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        

        if valid_days>0:
            valid_start, valid_end = split['valid_start'], split['valid_end']
            val_idx   = np.where((meta_dates >= valid_start) & (meta_dates <= valid_end))[0]
            val_set   = torch.utils.data.Subset(all_set, val_idx)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        end = time.time()
        
        print(f"构建单轮训练集耗时: {end - start:.2f} 秒")
        print(f"{pred_factor_name} 第{i+1}轮 训练/验证/测试集加载完成")

        # === 训练模型 ===
        if model_size == 'ori':
            model = TransformerNet(input_dim=len(factor_cols)) 
        elif model_size == 'large':
            model = TransformerNet(input_dim=50,d_model=256, num_heads=4, num_layers=6, ff_dim=1024)
        model = model.to(device)
        
        start = time.time()
        if valid_days == 0:
            model = train_model_no_val(model, model_type, train_loader, val_loader, test_loader, n_epochs=n_epochs, lr=lr, patience=patience, split_num=i,criterion=criterion,use_amp=use_amp )
        else:
            model = train_model(model, model_type, train_loader, val_loader, test_loader, n_epochs=n_epochs, lr=lr, patience=patience, split_num=i,criterion=criterion, use_amp=use_amp )
        end = time.time()
        print(f"训练结束，耗时 {end - start:.2f} 秒") 

        # === 预测 ===
        # 假设test_loader输出(batch, seq_len, n_factor), y (batch, ), meta (包含date/stock_id)
        y_pred_list, y_true_list, date_list, stock_id_list = [], [], [], []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, date,  stock_id in test_loader:
                batch_x = batch_x.to(device)
                with amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    output = model(batch_x).cpu().numpy().squeeze()
                #print("output:", output, "shape:", np.shape(output), "type:", type(output))

                y_pred_list.extend(np.atleast_1d(output).tolist())
                #y_true_list.extend(batch_y.numpy().squeeze())
                date_list.extend(date)
                stock_id_list.extend(stock_id)
        
        pred_df = pd.DataFrame({
            'date': date_list,
            'stock_id': stock_id_list,
            'y_pred': y_pred_list
        })
        pred_df_wide = pred_df.pivot(index='date', columns='stock_id', values='y_pred')
        pred_df_wide = pred_df_wide.sort_index()
        all_preds_wide.append(pred_df_wide) #list
        all_preds_df = pd.concat(all_preds_wide)
        all_preds_df = all_preds_df.sort_index()
        all_preds_df.to_csv(os.path.join(output_dir, f'{pred_factor_name}_result.csv'))

    elapsed = time.time() - train_t0
    print(f"所有滚动窗口预测结果已合并输出；总耗时 {elapsed/60:.1f} 分钟")
    all_preds_df = pd.concat(all_preds_wide)
    all_preds_df = all_preds_df.sort_index()
    all_preds_df.to_csv(os.path.join(output_dir, f'{pred_factor_name}_result.csv'))

    print("所有滚动窗口预测结果（宽表）已合并输出")
    return elapsed

