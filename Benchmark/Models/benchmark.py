import os
import sys

import pandas as pd
from dataprocess import split_data, select_feature
from train_module import train_module

new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
# Define paths for figures, features, and data
figure_path = sys.path[1] + '\\Result\\figures\\Fig04\\'
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'
analysis = ["class"]
alpha_param = "N"
host_param = "N"
data_type="Batch"
for analysis_level in analysis:
    for group_name in ["CTR_CRC"]:
        #for feature_name in ["all","ancom","maaslin2","metagenomeSeq","ttest","wilcoxon","lefse"]:
        #for feature_name in ["new_method"]:
        for feature_name in ["RFECV"]:
            comb_dir=figure_path+data_type+"/" + analysis_level + "/"
            meta_feature= split_data(analysis_level,group_name,data_type)
            studies_of_interest = ['DE-CRC', 'AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC', "AT-CRC", "CHN_HK-CRC",
                                   "CHN_SH-CRC", "IND-CRC", "US-CRC"]
            meta_feature = meta_feature[meta_feature["Study"].isin(studies_of_interest)]
            sig_feature = select_feature(analysis_level, group_name, feature_name,data_type)
            sig_feature_columns = set(sig_feature)
            meta_feature_columns = set(meta_feature.columns)
            intersection_columns = sig_feature_columns.intersection(meta_feature_columns)
            meta_sig_feature = meta_feature.loc[:, list(intersection_columns)]
            meta_feature = meta_feature[["Group", "Study"]].join(meta_sig_feature)
            param_df = pd.read_csv(sys.path[1]+"/Result/param1/"+analysis_level+"_best_params.csv")
            train_module(meta_feature,comb_dir, group_name, feature_name,param_df)
