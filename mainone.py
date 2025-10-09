# --- main.py ---
#日期使用 'YYYY-MM-DD'
from train_pip import *
import pandas as pd

if __name__ == "__main__":
    model_type = 'Transformer'
    factor_names = (
        [f'factor_pyt_{i:03d}' for i in range(1, 31)] +
        [f'L2_fac{i}' for i in range(1, 21)]
    )
    input_file = './fully_proceed_df_32_10lookforward_nan.csv'

    

    date_start = None
    date_end   = None
    # 训练月份范围
    #date_start = "2022-01-01"
    #date_end   = "2022-12-31"

    # 其它超参
    model_size='ori'
    n_epochs   = 30
    lr         = 1e-5
    patience   = 20
    valid_days = 3
    seq_len    = 5
    use_amp    = True
    batch_size=20000
    pred_factor_name = f"test_{model_size}_lr{lr:.0e}_seq{seq_len}_bs{batch_size}_amp{use_amp}_epochs{n_epochs}"
    records = []
    
    
    elapsed = train_predict_pipeline(
                    input=input_file,
                    factor_names=factor_names,
                    model_size=model_size,
                    seq_len=seq_len,
                    model_type=model_type,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    patience=patience,
                    valid_days=valid_days,
                    pred_factor_name=pred_factor_name,
                    criterion= ICLoss(method='pearson'),
                    use_amp=use_amp,
                    date_start=date_start,
                    date_end=date_end,
                )                           
                
    # 如需保存：
    # df_res.to_csv("./output/bench_time_bs_loss.csv", index=False)
