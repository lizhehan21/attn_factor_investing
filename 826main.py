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

    #  batch sizes
    #batch_grid = [1000,2000,10000,20000]

    # 两个损失函数
    loss_grid = {
        "ICLoss": ICLoss(method='pearson'),
        # "ICLoss(spearman)": ICLoss(method='spearman'),   # 若需一并对比可解注
        #"WeightedMSE": CustomSigmoidWeightedMSELoss(),
    }
   
    model_size = 'ori'
    #amp = True
    # 训练月份范围
    date_start = None
    date_end   = None
    date_start = "2018-01-01"
    date_end   = "2024-12-31"

    # 其它超参
    n_epochs   = 30
    lr         = 1e-5
    patience   = 20
    valid_days = 0
    lookback_grid = [63,126,189,252]
    seq_len    = 5
    use_amp    = True
    batch_size=20000
    
    records = []
    for lookback in lookback_grid:
        for j in range(1):
            for i in range(1):
                print(f"\n========== 运行: lookback={lookback}, lookback={lookback} ==========")
                elapsed = train_predict_pipeline(

                    input=input_file,
                    factor_names=factor_names,
                    model_size=model_size,
                    lookback=lookback,
                    seq_len=seq_len,
                    model_type=model_type,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    patience=patience,
                    valid_days=valid_days,
                    pred_factor_name=f"30e_lookback{lookback}",
                    criterion= ICLoss(method='pearson'),
                    use_amp=use_amp,
                    date_start=date_start,
                    date_end=date_end,
                )
                
    # 如需保存：
    # df_res.to_csv("./output/bench_time_bs_loss.csv", index=False)
