import ast
from collections import Counter
import pymrmr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import sys
import os
from sklearn.preprocessing import MinMaxScaler

from Benchmark.Models.dataprocess import split_data,change_group

new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
# Define paths for figures, features, and data
figure_path = sys.path[1] + '\\Result\\figures\\Fig03\\'
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

def select_all_study_feature(analysis_level, group_name, feature_name,data_type):
    """
       Select features for all studies.
       :param analysis_level: Taxonomic level (e.g., species).
       :param group_name: Group for comparison (e.g., CTR_CRC).
       :param feature_name: Name of the feature selection method (e.g., all, union).
       :param data_type: Data type (e.g., Raw, Raw_log).
       :return: List of selected features.
       """
    feature_dir = f"{feature_path}/{analysis_level}/{data_type}/{group_name}/adj_p_{group_name}.csv"
    feature = pd.read_csv(feature_dir, index_col=0)
    feature.fillna(1, inplace=True)
    if feature_name == "all":
        feature_select = feature.index.tolist()
    elif feature_name == "union":
        feature_select = feature[feature.lt(0.05).all(axis=1)].index.tolist()
    else:
        feature_select = feature[feature[feature_name] < 0.05].index.tolist()
    return feature_select
def select_single_study_feature(analysis_level, group_name, feature_name,study,data_type):
    feature_dir = f"{feature_path}/{analysis_level}/{data_type}/{group_name}/{study}/adj_p_{group_name}.csv"
    feature = pd.read_csv(feature_dir, index_col=0)
    feature.fillna(1, inplace=True)
    if feature_name == "all":
        feature_select = feature.index.tolist()
    elif feature_name == "union":
        feature_select = feature[feature.lt(0.05).all(axis=1)].index.tolist()
    else:
        feature_select = feature[feature[feature_name] < 0.05].index.tolist()
    return feature_select
def raw_all_studys(analysis_level, group_name, source_study,data_type):
    study_single_lefse_MaAsLin = []
    feature_study_mapping = {}
    for study in source_study:
        lefse_single_feature = select_single_study_feature(analysis_level, group_name, "lefse",study,data_type)
        MaAsLin_feature = select_single_study_feature(analysis_level, group_name, "maaslin2",study,data_type)
        ancom_feature = select_single_study_feature(analysis_level, group_name, "ancom", study,data_type)
        study_single_lefse_MaAsLin += lefse_single_feature + MaAsLin_feature +ancom_feature
    lefse_all_feature = select_all_study_feature(analysis_level, group_name,"lefse",data_type)
    MaAsLin_all_feature =select_all_study_feature(analysis_level, group_name, "maaslin2",data_type)
    ancom_all_feature=select_all_study_feature(analysis_level, group_name, "ancom",data_type)
    study_single_lefse_MaAsLin +=lefse_all_feature+MaAsLin_all_feature+ancom_all_feature
    feature_frequencies = Counter(study_single_lefse_MaAsLin)
    df_frequencies = pd.DataFrame(list(feature_frequencies.items()), columns=["Feature", "Frequency"])

    df_frequencies["Frequency"] = df_frequencies["Feature"].map(feature_frequencies)
    base_directory = sys.path[1] + '/Result/feature/'+analysis_level+"/"+data_type+"/"+group_name
    print(base_directory)
    output_freq_file = base_directory+"/feature_frequencies.csv"
    directory = os.path.dirname(output_freq_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_frequencies.to_csv(output_freq_file, index=False)
def frequnence_main(data_type):
    source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "CHN_WF-CRC", "CHN_SH-CRC", "CHN_HK-CRC", "DE-CRC", "IND-CRC","US-CRC"]
    raw_all_studys("class","CTR_CRC", source_study,data_type)
    raw_all_studys("order","CTR_CRC", source_study,data_type)
    raw_all_studys("family","CTR_CRC", source_study,data_type)
    raw_all_studys("genus", "CTR_CRC", source_study, data_type)
    raw_all_studys("species", "CTR_CRC", source_study, data_type)
    raw_all_studys("ko_gene", "CTR_CRC", source_study, data_type)
    raw_all_studys("uniref_family", "CTR_CRC", source_study, data_type)
if __name__ == "__main__":
    # Calculate feature frequencies across studies
    frequnence_main(data_type="Batch")
    frequnence_main(data_type="Raw")
    frequnence_main(data_type="Raw_log")