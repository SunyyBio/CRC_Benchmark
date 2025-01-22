import os
import sys

import pandas as pd
from collections import Counter
from tqdm import tqdm

new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

def raw_target_studys(source_study, target_study, ratio):
    """
    Calculate the frequency of selected features for target domain data.

    :param source_study: Source study identifier
    :param target_study: Target study identifier
    :param save_dir: Directory to save the output
    :param ratio: Ratio value for directory path
    :return: None
    """
    # File paths
    feature_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Target_{target_study}_feature.csv"
    feature = pd.read_csv(feature_file)
    MaAsLin2_feature_select = feature["Target_MaAsLin"].dropna().tolist()
    lefse_feature_select =  feature["Target_lefse"].dropna().tolist()
    Ancom_feature_select =  feature["Target_Ancom"].dropna().tolist()
    # Combine features and calculate frequencies
    combined_features = MaAsLin2_feature_select + lefse_feature_select + Ancom_feature_select
    feature_frequencies = Counter(combined_features)
    # Create DataFrame for feature frequencies
    df_frequencies = pd.DataFrame(feature_frequencies.items(), columns=["Feature", "Frequency"])
    # Save the frequency DataFrame to CSV
    output_freq_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/target_{target_study}_feature_frequencies.csv"
    directory = os.path.dirname(output_freq_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_frequencies.to_csv(output_freq_file, index=False)
def cal_feature_frequency(feature_file,feature_name,source_study,output_file,):

    feature = pd.read_csv(feature_file)
    study_single_lefse_MaAsLin = []
    feature_study_mapping = {}
    for study in source_study:
        lefse_single_feature = study + "_lefse"
        MaAsLin_feature = study + "_MaAsLin"
        ancom_feature = study + "_Ancom"
        study_single_lefse_MaAsLin += lefse_single_feature + MaAsLin_feature + ancom_feature
        for feature_name in study_single_lefse_MaAsLin:
            if feature_name in feature_study_mapping:
                feature_study_mapping[feature_name].append(study)
            else:
                feature_study_mapping[feature_name] = [study]
    lefse_all_feature = feature_name + "_lefse"
    lefse_all_feature = feature[lefse_all_feature].dropna().tolist()
    MaAsLin_all_feature = feature_name + "_MaAsLin"
    MaAsLin_all_feature = feature[MaAsLin_all_feature].dropna().tolist()
    Ancom_all_feature = feature_name+ "_Ancom"
    Ancom_all_feature = feature[Ancom_all_feature].dropna().tolist()
    study_single_lefse_MaAsLin += lefse_all_feature + MaAsLin_all_feature + Ancom_all_feature
    feature_frequencies = Counter(study_single_lefse_MaAsLin)

    df_frequencies = pd.DataFrame(list(feature_frequencies.items()), columns=["Feature", "Frequency"])
    df_frequencies["Frequency"] = df_frequencies["Feature"].map(feature_frequencies)
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_frequencies.to_csv(output_file, index=False)
    df = pd.read_csv(feature_file)
    df["Feature"] = df_frequencies["Feature"]
    df.to_csv(feature_file)

def raw_MNN_all_studys(source_study):
    """
    原始数据进行计算特征的频率
    :param source_study:
    :param save_dir:
    :return:
    """
    feature = pd.read_csv(f"{feature_path}raw/merge_all_cohort_feature.csv")
    study_single_lefse_MaAsLin = []
    feature_study_mapping = {}
    for study in source_study:
        lefse_single_feature = study + "_lefse"
        MaAsLin_feature = study + "_MaAsLin"
        ancom_feature = study + "_Ancom"
        study_single_lefse_MaAsLin += lefse_single_feature + MaAsLin_feature +ancom_feature
        for feature_name in study_single_lefse_MaAsLin:
            if feature_name in feature_study_mapping:
                feature_study_mapping[feature_name].append(study)
            else:
                feature_study_mapping[feature_name] = [study]
    lefse_all_feature='_'.join(source_study)+"_lefse"
    lefse_all_feature = feature[lefse_all_feature].dropna().tolist()
    MaAsLin_all_feature = '_'.join(source_study) + "_MaAsLin"
    MaAsLin_all_feature = feature[MaAsLin_all_feature].dropna().tolist()
    Ancom_all_feature = '_'.join(source_study) + "_Ancom"
    Ancom_all_feature = feature[Ancom_all_feature].dropna().tolist()
    study_single_lefse_MaAsLin +=lefse_all_feature+MaAsLin_all_feature+Ancom_all_feature
    for feature_name in lefse_all_feature:
        if feature_name in feature_study_mapping:
            feature_study_mapping[feature_name].append("all_lefse")
        else:
            feature_study_mapping[feature_name] = ["all_lefse"]
    for feature_name in MaAsLin_all_feature:
        if feature_name in feature_study_mapping:
            feature_study_mapping[feature_name].append("all_MaAsLin")
        else:
            feature_study_mapping[feature_name] = ["all_MaAsLin"]
    feature_frequencies = Counter(study_single_lefse_MaAsLin)

    # 创建特征频率表格
    df_frequencies = pd.DataFrame(list(feature_frequencies.items()), columns=["Feature", "Frequency"])
    # 添加特征对应的研究数量列
    df_frequencies["Study Count"] = df_frequencies["Feature"].map(lambda x: len(set(feature_study_mapping.get(x, []))))
    # 将频率列修改为特征出现的次数
    df_frequencies["Frequency"] = df_frequencies["Feature"].map(feature_frequencies)
    # 保存特征频率表格为CSV文件
    file_name="_".join(source_study)
    output_freq_file = f"{feature_path}raw/{file_name}_feature_frequencies.csv"
    directory = os.path.dirname(output_freq_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_frequencies.to_csv(output_freq_file, index=False)

    # 将rf以及lefse以及MaAsLin的筛选的总特征放在study_to_study里面
    df = pd.read_csv(f"{feature_path}raw/merge_all_cohort_feature.csv")
    df["Raw_"+"_".join(source_study) + "_all_lefse_MaAsLin_Ancom"] = df_frequencies["Feature"]
    df.to_csv(f"{feature_path}raw/merge_all_cohort_feature.csv")
def TL_MNN_all_studys(source_study,target_study,ratio):
    """
    经过迁移学习的数据计算频率
    :param source_study:
    :param target_study:
    :param ratio:
    :param save_dir:
    :return:
    """
    feature_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_feature.csv"
    feature = pd.read_csv(feature_file)

    study_single_lefse_MaAsLin = []
    feature_study_mapping = {}  # 存储特征及其对应的研究名称列表

    for study in source_study+[target_study]:
        lefse_MaAsLin_Ancom_feature_name = study + "_lefse_MaAsLin_Ancom"
        lefse_MaAsLin_single_feature = feature[lefse_MaAsLin_Ancom_feature_name].dropna().tolist()
        study_single_lefse_MaAsLin += lefse_MaAsLin_single_feature
        for feature_name in lefse_MaAsLin_single_feature:
            if feature_name in feature_study_mapping:
                feature_study_mapping[feature_name].append(study)
            else:
                feature_study_mapping[feature_name] = [study]  # 记录特征对应的研究名称列表
    lefse_all_feature='Meta_iTL_all_cohort_lefse'
    lefse_all_feature = feature[lefse_all_feature].dropna().tolist()
    MaAsLin_all_feature = 'Meta_iTL_all_cohort_MaAsLin'
    MaAsLin_all_feature = feature[MaAsLin_all_feature].dropna().tolist()
    Ancom_all_feature = 'Meta_iTL_all_cohort_Ancom'
    Ancom_all_feature = feature[Ancom_all_feature].dropna().tolist()

    study_single_lefse_MaAsLin +=lefse_all_feature+MaAsLin_all_feature+Ancom_all_feature
    for feature_name in lefse_all_feature:
        if feature_name in feature_study_mapping:
            feature_study_mapping[feature_name].append("all_lefse")
        else:
            feature_study_mapping[feature_name] = ["all_lefse"]
    for feature_name in MaAsLin_all_feature:
        if feature_name in feature_study_mapping:
            feature_study_mapping[feature_name].append("all_MaAsLin")
        else:
            feature_study_mapping[feature_name] = ["all_MaAsLin"]

    feature_frequencies = Counter(study_single_lefse_MaAsLin)

    df_frequencies = pd.DataFrame(list(feature_frequencies.items()), columns=["Feature", "Frequency"])

    df_frequencies["Study Count"] = df_frequencies["Feature"].map(lambda x: len(set(feature_study_mapping.get(x, []))))

    df_frequencies["Frequency"] = df_frequencies["Feature"].map(feature_frequencies)

    # 保存特征频率表格为CSV文件
    output_freq_file = (f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_feature_frequencies.csv")
    directory = os.path.dirname(output_freq_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_frequencies.to_csv(output_freq_file, index=False)
    #将rf以及lefse以及MaAsLin的筛选的总特征放在study_to_study里面
    df=pd.read_csv(f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_feature.csv")
    df["Meta_iTL_feature_all_lefse_MaAsLin_Ancom"]=df_frequencies["Feature"]
    df.to_csv(f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_feature.csv")

def TL_function(target_study):
    """
    迁移学习的样本进行求频率
    :return:
    """
    for ratio in tqdm(["S0.2","S0.3","S0.4","S0.5","S0.6"]):
        if target_study == "CHN_SH-CRC-2":
            source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "CHN_WF-CRC", "CHN_SH-CRC-3",
                            "US-CRC-3"]
        elif target_study == "CHN_WF-CRC":
            source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-2",
                            "CHN_SH-CRC-3"]
        TL_MNN_all_studys(source_study,target_study,ratio=ratio)

def raw_funtion(target_study):
    """
    原始数据求特征的频率
    不论是哪个ratio，筛选出来的特征是一定的
    :return:
    """
    if target_study == "CHN_SH-CRC-2":
        source_study = ["FR-CRC","AT-CRC","ITA-CRC","JPN-CRC","US-CRC-2","CHN_WF-CRC","CHN_SH-CRC-3","US-CRC-3"]
    elif target_study == "CHN_WF-CRC":
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-2", "CHN_SH-CRC-3"]
    raw_MNN_all_studys(source_study)

def raw_target_funtion(target_study):
    """
    Find the frequency of features of sample screening in the target domain of transfer learning
    :return:
    """
    if target_study == "CHN_SH-CRC-2":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-3_US-CRC-3"
    elif target_study == "CHN_WF-CRC":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-2_CHN_SH-CRC-3"
    for ratio in tqdm(["S0.2","S0.3","S0.4","S0.5","S0.6"]):
        raw_target_studys(source_study, target_study,ratio)
if __name__ == "__main__":
    target_study="CHN_SH-CRC-2"
    raw_funtion(target_study)
    raw_target_funtion(target_study)
    TL_function(target_study)