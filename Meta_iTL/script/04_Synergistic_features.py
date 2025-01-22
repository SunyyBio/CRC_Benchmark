import argparse
import ast
import os
import sys
import pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from Meta_iTL.function.MNN_function import split_data,change_group
from Meta_iTL.function.Optimal_feature_function import Synergistic_select_feature

new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

def raw_main(target_study,frequnency_thre):
    if target_study=="CHN_WF-CRC":
        source_study = ["FR-CRC","AT-CRC","ITA-CRC","JPN-CRC","US-CRC-2","US-CRC-3","CHN_SH-CRC-4","CHN_SH-CRC-2"]
    if target_study=="CHN_SH-CRC-4":
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "CHN_WF-CRC", "CHN_SH-CRC-2", "US-CRC-3"]
    file_name="_".join(source_study)
    feature_file = f"{feature_path}raw/{file_name}_optimal.csv"
    frequence_file = f"{feature_path}raw/{file_name}_feature_frequencies.csv"
    meta_feature= split_data("species", "CTR_ADA","Raw_log")
    meta_feature = meta_feature[meta_feature['Study'].str.contains("|".join(source_study))]
    meta_feature = change_group(meta_feature, "CTR_ADA")

    synergistic_feature_name="_".join(source_study) + "_synergistic"
    Synergistic_select_feature(meta_feature, frequence_file, feature_file,frequnency_thre,synergistic_feature_name)

def target_main(ratio,target_study,frequnency_thre):
    if target_study == "CHN_WF-CRC":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"
    if target_study == "CHN_SH-CRC-4":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"

    feature_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/target_optimal.csv"
    frequence_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/target_{target_study}_feature_frequencies.csv"

    train_data_file = (data_path + source_study + "_" + target_study
                       + "/filtered_" + source_study + "_" + target_study + "_" + ratio + "_train.csv")
    train_data = pd.read_csv(train_data_file)
    train_data = train_data[train_data['Study'].str.contains(target_study)]
    meta_feature = change_group(train_data, "CTR_ADA")
    synergistic_feature_name="Target_" + source_study + "_" + target_study + "_synergistic"
    Synergistic_select_feature(meta_feature, frequence_file, feature_file,frequnency_thre,synergistic_feature_name)
def MNN_main(ratio,target_study,frequnency_thre):

    if target_study == "CHN_WF-CRC":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"
    if target_study == "CHN_SH-CRC-4":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"
    feature_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_optimal.csv"
    frequence_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_feature_frequencies.csv"
    train_data_file = (data_path+ source_study + "_" + target_study
                + "/filtered_" + source_study + "_" + target_study + "_" + ratio + "_train.csv")
    train_data = pd.read_csv(train_data_file)
    meta_feature = change_group(train_data, "CTR_ADA")
    synergistic_feature_name = source_study + "_" + target_study + "_synergistic"
    Synergistic_select_feature(meta_feature, frequence_file, feature_file, frequnency_thre,
                               synergistic_feature_name)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synergistic feature.")
    parser.add_argument("-t", "--target_study", required=True, help="Target study (e.g., CHN_WF-CRC or CHN_SH-CRC-4 ).")
    parser.add_argument("-d", "--data_type", required=True, help="Data Types (e.g., Meta_iTL or Raw).")
    parser.add_argument("-r", "--ratio", required=True, help="ratio (e.g., S0.2 or S0.3 or S0.4 ...).")
    parser.add_argument("-f", "--frequency_threshold", required=True, help="Frequency Threshold (e.g., 0 or 1...).")

    args = parser.parse_args()
    target_study = args.target_study
    data_type=args.data_type
    ratio = args.ratio
    try:
        frequnency_thre = float(args.frequency_threshold)
    except ValueError:
        raise ValueError("Frequency threshold (-f) must be a numeric value.")

    if data_type=="Meta_iTL":
        target_main(ratio, target_study, frequnency_thre)
        MNN_main(ratio, target_study, frequnency_thre)
    if data_type=="Raw":
        raw_main(target_study,frequnency_thre)
    print(f"{data_type} Synergistic select feature completed.")
