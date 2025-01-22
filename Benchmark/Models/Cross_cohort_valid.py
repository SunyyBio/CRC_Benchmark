import ast
import pickle
from tqdm import tqdm
import pandas as pd
from LODO import Lodo_study_study_fig
import os
from Benchmark.Models.dataprocess import split_data
current_file_path = __file__
# 切割路径并获取上级的上级目录
new_path = os.path.abspath(current_file_path + "/../..")
model_type="rf"
def param(analysis_level,group):
    with open(new_path+"/Result/param/"+model_type+group+analysis_level + '_best_params.pkl', 'rb') as f:
        loaded_best_param = pickle.load(f)
    return loaded_best_param
def LODO(source_study):
    # """LODO"""
    for analysis_level in ["species"]:
        for group in tqdm(["CTR_CRC"]):
            meta_feature, _, _ = split_data(analysis_level, group,"Raw_log")
            df = pd.read_csv(f"{new_path}/Result/feature/{analysis_level}/Raw_log/{group}/feature.csv",
                index_col=0)
            features = df['FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_CHN_WF-CRC_CHN_SH-CRC_CHN_HK-CRC_DE-CRC_IND-CRC_US-CRC_rf_optimal'].dropna().tolist()
            meta_feature = meta_feature[(meta_feature['Study'].str.contains("|".join(source_study)))]
            sig_feature_columns = set(features)
            meta_feature_columns = set(meta_feature.columns)
            intersection_columns = sig_feature_columns.intersection(meta_feature_columns)
            finally_frequency_feature = ["Group", "Study"] + list(intersection_columns)
            meta_sig_feature = meta_feature.loc[:,finally_frequency_feature]
            studies=source_study
            studies_num = len(studies)
            output_dir =new_path+"/Result/figers"+"/"+analysis_level+"_"+model_type+"_" +group + "_LODO.pdf"
            model_param=param(analysis_level,group)
            Lodo_study_study_fig(meta_sig_feature, studies, studies_num, output_dir, group,42,model_type,**model_param)
        print(analysis_level, group)
source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC","IND-CRC","CHN_WF-CRC","CHN_SH-CRC","CHN_HK-CRC","DE-CRC"]
LODO(source_study)