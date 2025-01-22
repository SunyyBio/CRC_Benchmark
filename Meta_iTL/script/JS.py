import os

import numpy as np
import pandas as pd


# ========== 1. 定义 KL 与 JS 散度函数 ==========

def kl_divergence(p, q):
    """
    计算 KL 散度: KL(P || Q) = sum( p_i * log(p_i / q_i) ).
    输入:
        p, q: 一维向量 (numpy array), 已经确保 sum(p)=1, sum(q)=1。
    输出:
        KL 散度的值 (float)
    """
    # 防止出现 log(0) 的情况，在 p, q 上加一个极小值
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    """
    计算 Jensen-Shannon 散度:
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), 其中 M = 0.5 * (P + Q).
    输入:
        p, q: 一维向量 (numpy array), 已经确保 sum(p)=1, sum(q)=1。
    输出:
        JS 散度的值 (float)
    """
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ========== 2. 将一组数据合并/平均为“一维分布向量”的辅助函数 ==========

def pooling_distribution(data):
    """
    将 data (形状: [样本数, 特征数]) 合并为一个“一维分布”。
    做法: 在样本维度上对每个特征求和, 最后再归一化使向量和为 1。
    这相当于把所有样本当作一个大的 pooled sample。
    注意: data 的每一行还没有被归一化也没关系，这里会直接求和。
    """
    # 如果 data 是 dataframe, 先转为 numpy
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 在样本维度(行)上对每列(特征)求和, 得到 1 x 特征数
    summed = np.sum(data, axis=0)
    summed += 1e-10  # 防止出现全零特征
    # 归一化
    pooled_dist = summed / np.sum(summed)
    return pooled_dist


def averaging_distribution(data):
    """
    将 data (形状: [样本数, 特征数]) 合并为一个“一维分布”。
    做法:
      1) 每个样本先进行行归一化(使它成为概率分布)
      2) 对所有样本的行分布进行均值
      3) 对结果再次归一化

    相比于 pooling_distribution，这种方法会让每个样本的权重相同，
    而不是按测序深度(或原始总和)来加权。
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 1) 对每个样本的行向量先做归一化
    # 防止 log(0)，先加个极小值
    data = data + 1e-10
    row_sums = np.sum(data, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10  # 避免除 0
    data_normed = data / row_sums

    # 2) 对所有样本的行分布进行均值 => 得到 1 x 特征数
    avg_dist = np.mean(data_normed, axis=0)

    # 3) 再一次归一化，确保总和为 1
    avg_dist = avg_dist / np.sum(avg_dist)
    return avg_dist


# ========== 3. 计算“整组分布”间 KL / JS 的函数 ==========

def compute_group_kl_js(data1, data2, method='pooling'):
    """
    给定两组数据 (各自形状: [样本数, 特征数])，先合并成一维分布，再计算 KL/JS。
    参数:
        data1, data2: 两组数据, 可以是 pd.DataFrame 或 numpy.ndarray。
        method: 'pooling' 或 'averaging'，用哪种方式把组内样本合并成一个向量。
    返回:
        kl, js: 分别是 KL(P || Q) 和 JS(P || Q) 的值。
    """
    # 1) 根据指定方式，获取 data1, data2 的“一维分布”向量
    if method == 'pooling':
        p = pooling_distribution(data1)
        q = pooling_distribution(data2)
    elif method == 'averaging':
        p = averaging_distribution(data1)
        q = averaging_distribution(data2)
    else:
        raise ValueError("method 必须为 'pooling' 或 'averaging'")

    # 2) 计算 KL / JS
    kl_value = kl_divergence(p, q)
    js_value = js_divergence(p, q)

    return kl_value, js_value


# ========== 4. 示例调用 ==========

for ratio in ["S0.2", "S0.3", "S0.4", "S0.5", "S0.6"]:
    # Create a directory for the current ratio
    ratio_dir = f"F:/bioinfoclub/benchmark/result/TL/Figure/TL/验证TL/KL/{ratio}/"
    os.makedirs(ratio_dir, exist_ok=True)

    # Initialize an empty DataFrame to store results for this ratio
    all_results = pd.DataFrame()

    for seed in range(1):  # Run 50 iterations
        save_dir = "MNN_all_studys"
        target_study = "CHN_WF-CRC"
        #source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-3_US-CRC-3"
        source_study="FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-2_CHN_SH-CRC-3"

        # Load the target dataset
        target = pd.read_csv(
            f'F:/bioinfoclub/benchmark/result/TL/TL/{save_dir}/data/{source_study}_{target_study}/{target_study}_{ratio}_test.csv',
            index_col=0
        )
        target = target[target['Group'].isin(["ADA", "CTR"])]
        #target = target.sample(n=25,random_state=seed)
        target = target.drop(columns=['Study'], errors='ignore')

        # Load the feature dataset
        Meta_iTL = pd.read_csv(
            f'F:/bioinfoclub/benchmark/result/TL/TL/{save_dir}/data/{source_study}_{target_study}/filtered_{source_study}_{target_study}_{ratio}_train.csv',
            index_col=0
        )
        Meta_iTL = Meta_iTL[Meta_iTL['Group'].isin(["ADA", "CTR"])]
        Meta_iTL = Meta_iTL.drop(columns=['Study'], errors='ignore')

        feature_path = 'F:/bioinfoclub/benchmark/result/TL/TL/MNN_all_studys/feature/finally_feature/' + source_study + "_" + target_study + "/" + ratio + "/MNN_all_studys_feature.csv"
        feature = pd.read_csv(feature_path)
        feature_path_target = 'F:/bioinfoclub/benchmark/result/TL/TL/study_to_study/feature/finally_feature/FR-CRC_' + target_study + "/" + ratio + "/study_to_study_feature.csv"
        feature_target = pd.read_csv(feature_path_target)
        frequency_feature_name2 = "Target_FR-CRC_" + target_study + "_rf_optimal"
        finally_rf_optimal_feature2 = feature_target[frequency_feature_name2].dropna().tolist()
        frequency_feature_name = source_study + "_" + target_study + "_rf_optimal"
        finally_rf_optimal_feature = feature[frequency_feature_name].dropna().tolist()
        optimal_features_set = list(dict.fromkeys(finally_rf_optimal_feature2 + finally_rf_optimal_feature))

        feature_finally = pd.read_csv(f"F:/bioinfoclub/benchmark/result/TL/raw/MNN_all_studys/feature/rf_optimal_finally.csv")
        Raw_feature=pd.Series(feature_finally[f"{source_study}_rf_optimal"].tolist()).dropna().tolist()
        # Load the original dataset
        meta_all = pd.read_csv("C:/Users/sunny/Desktop/论文/meta2.csv")
        meta_all = meta_all[
            meta_all['Study'].isin(source_study.split("_")) & meta_all['Group'].isin(["ADA", "CTR"])
            ]
        feature_all = pd.read_csv("G:/deeplearning/CRC/benchmarker/species/data/new2/feature_rare_CTR_ADA.csv",
                                  sep='\t', index_col=0)
        #feature_all = np.log(feature_all + 1)
        feature_all = feature_all.loc[meta_all['Sample_ID']]

        # Combine data
        target['batch'] = "target"
        Meta_iTL['batch'] = "TL"
        feature_all['batch'] = "Raw"
        combined_df = pd.concat([target, Meta_iTL, feature_all], axis=0, ignore_index=True)

        # Normalize the data
        def normalize(row):
            return row / np.sum(row)

        target_data = combined_df[combined_df['batch'] == "target"].drop(columns=['batch', 'Group']).apply(normalize,
                                                                                                           axis=1)
        TL_data = combined_df[combined_df['batch'] == "TL"].drop(columns=['batch', 'Group']).apply(normalize,
                                                                                                   axis=1)
        Raw_data = combined_df[combined_df['batch'] == "Raw"].drop(columns=['batch', 'Group']).apply(normalize,
                                                                                                     axis=1)

        # 方法一: pooling
        kl_pooling, js_pooling = compute_group_kl_js(target_data, TL_data, method='pooling')
        print(">>> [Pooling] KL={}, JS={}".format(kl_pooling, js_pooling))

        # 方法二: averaging
        kl_averaging, js_averaging = compute_group_kl_js(target_data, TL_data, method='averaging')
        print(">>> [Averaging] KL={}, JS={}".format(kl_averaging, js_averaging))

        # Append the current iteration's results
        iteration_results = pd.DataFrame([
            {
                "Iteration": seed,
                "Comparison": "target_vs_TL",
                "KL_Divergence": js_pooling,
                "JS_Divergence": js_averaging,
                "Ratio": ratio
            }
            # ,
            # {
            #     "Iteration": seed,
            #     "Comparison": "target_vs_Raw",
            #     "KL_Divergence": kl_target_Raw,
            #     "JS_Divergence": js_target_Raw,
            #     "Ratio": ratio
            # }
        ])
        all_results = pd.concat([all_results, iteration_results], ignore_index=True)

    all_results.to_csv(
        os.path.join(ratio_dir, f"{source_study}_{target_study}_{ratio}_all_divergence_results.csv"),
        index=False
    )
    # Save all results for this ratio to a single CSV file
    all_results.to_csv(
        os.path.join(ratio_dir, f"{source_study}_{target_study}_{ratio}_all_divergence_results.csv"),
        index=False
    )

    # Print completion message
    print(f"Results for ratio {ratio} have been saved.")