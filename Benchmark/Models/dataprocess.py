import os
import sys
import numpy as np
import pandas as pd

new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

def split_data(analysis_level, groups, Raw):
    if Raw in ["Raw", "Raw_log"]:
        data = data_path + f"{analysis_level}/feature_rare_{groups}.csv"
        feature = pd.read_csv(data, sep='\t', index_col=0)
        if Raw == "Raw_log":
            feature = np.log(feature+1)
    else:
        data = data_path + f"{analysis_level}/{groups}_adj_batch.csv"
        feature_abundance = pd.read_csv(data, sep=',', index_col=0)
        feature = feature_abundance.T

    meta = pd.read_csv(data_path+"/meta.csv")
    merged_df = pd.merge(feature, meta[["Group", "Sample_ID", "Study"]], left_index=True, right_on='Sample_ID', how="inner")
    merged_df.set_index('Sample_ID', inplace=True)
    return merged_df
def select_feature(analysis_level, group_name, feature_name,Raw):
    if Raw == "Raw":
        if feature_name == "new_method":
            feature_dir = (
                    feature_path+f"{analysis_level}/Raw/{group_name}/feature.csv")
        else:
            feature_dir = (
                feature_path+f"{analysis_level}/Raw/{group_name}/adj_p_{group_name}.csv")
    elif Raw == "Raw_log":
        if feature_name == "new_method":
            feature_dir = (
                    feature_path+f"{analysis_level}/Raw_log/{group_name}/feature.csv")
        else:
            feature_dir = (
            feature_path+f"{analysis_level}/Raw_log/{group_name}/adj_p_{group_name}.csv")
    else:
        if feature_name == "new_method":
            feature_dir = (
                    feature_path+f"{analysis_level}/Batch/{group_name}/feature.csv")
        else:
            feature_dir = (
                feature_path+f"{analysis_level}/Batch/{group_name}/adj_p_{group_name}.csv")
    feature = pd.read_csv(feature_dir, index_col=0)

    if feature_name == "all":
        feature.fillna(1, inplace=True)
        feature_select = feature.index.tolist()
    elif feature_name == "union":
        feature.fillna(1, inplace=True)
        feature_select = feature[feature.lt(0.01).all(axis=1)].index.tolist()
    elif feature_name=="new_method":
        if group_name=="CTR_CRC":
            feature_select = feature['FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_CHN_WF-CRC_CHN_SH-CRC_CHN_HK-CRC_DE-CRC_IND-CRC_US-CRC_rf_optimal'].dropna().tolist()
        if group_name == "CTR_ADA":
            feature_select=feature['AT-CRC_JPN-CRC_FR-CRC_CHN_WF-CRC_ITA-CRC_CHN_SH-CRC-2_US-CRC-3_CHN_SH-CRC-4_US-CRC-2_rf_optimal'].dropna().tolist()
    elif feature_name=="lefse":
        feature.fillna(1, inplace=True)
        feature_select = feature[feature[feature_name] < 0.05].index.tolist()
    else:
        feature_select = feature[feature[feature_name] < 0.01].index.tolist()
    return feature_select

def balance_samples(meta_feature, group1, group2):
    grouped = meta_feature.groupby("Study")
    selected_samples = pd.DataFrame()
    for name, group in grouped:
        ada_samples = group[group["Group"] == group1]
        ctr_samples = group[group["Group"] == group2]
        min_count = min(len(ada_samples), len(ctr_samples))
        if min_count > 0:
            ada_samples = ada_samples.sample(n=min_count, replace=False)
            ctr_samples = ctr_samples.sample(n=min_count, replace=False)
            selected_samples = selected_samples.append(ada_samples)
            selected_samples = selected_samples.append(ctr_samples)
    selected_samples.reset_index(drop=False, inplace=True)
    selected_samples.set_index('Sample_ID', inplace=True)
    return selected_samples

def change_group(meta_feature,group_name):
    """
      Convert group labels to binary.
      :param meta_feature: Metadata dataframe with group column.
      :param group_name: Name of the target group.
      :return: Updated metadata dataframe with binary group labels.
      """
    if group_name == 'CTR_ADA':
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "ADA" else 0)
    elif group_name == "CTR_CRC":
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    else:
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    return meta_feature

