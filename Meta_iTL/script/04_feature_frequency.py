import argparse
import os
import sys
import pandas as pd
from collections import Counter
from tqdm import tqdm
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'
def cal_feature_frequency(feature_file,feature_name,source_study,output_file):
    feature = pd.read_csv(feature_file)
    study_single_lefse_MaAsLin = []
    if source_study!="Target":
        for study in source_study:
            lefse_single_feature = feature[study + "_lefse"].dropna().tolist()
            MaAsLin_feature = feature[feature_name + "_MaAsLin"].dropna().tolist()
            ancom_feature = feature[feature_name + "_Ancom"].dropna().tolist()
            study_single_lefse_MaAsLin += lefse_single_feature + MaAsLin_feature + ancom_feature

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
def TL_function(ratio,target_study):
    """
    Transfer learning samples to find frequency
    :return:
    """

    if target_study == "CHN_SH-CRC-4":
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "CHN_WF-CRC", "CHN_SH-CRC-2",
                        "US-CRC-3"]
    elif target_study == "CHN_WF-CRC":
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-4",
                        "CHN_SH-CRC-2"]
    feature_file=f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_feature.csv"
    feature_name='Meta_iTL_all_cohort'
    output_file=f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_feature_frequencies.csv"
    cal_feature_frequency(feature_file,feature_name,source_study,output_file)
def raw_funtion(target_study):
    """
    The frequency of the original data feature
    No matter which ratio is used, the features selected are certain
    :return:
    """
    if target_study == "CHN_SH-CRC-4":
        source_study = ["FR-CRC","AT-CRC","ITA-CRC","JPN-CRC","US-CRC-2","CHN_WF-CRC","CHN_SH-CRC-2","US-CRC-3"]
    elif target_study == "CHN_WF-CRC":
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-4", "CHN_SH-CRC-2"]
    feature_file=f"{feature_path}raw/merge_all_cohort_feature.csv"
    feature_name='_'.join(source_study)
    output_file=f"{feature_path}raw/{feature_name}_feature_frequencies.csv"
    cal_feature_frequency(feature_file,feature_name,source_study,output_file)
def raw_target_funtion(ratio,target_study):
    """
    Find the frequency of features of sample screening in the target domain of transfer learning
    :return:
    """
    feature_file = f"{feature_path}raw/merge_all_cohort_feature.csv"
    feature_name = target_study
    output_file = f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/target_{target_study}_feature_frequencies.csv"
    cal_feature_frequency(feature_file, feature_name, "Target", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate feature frequencies.")
    parser.add_argument("-t", "--target_study", required=True, help="Target study (e.g., CHN_SH-CRC-4 or CHN_WF-CRC).")
    parser.add_argument("-d", "--data_type", required=True, help="Data type (e.g., Meta_iTL or Target or Raw).")
    parser.add_argument("-r", "--ratio", required=True, help="ratio (e.g., S0.2 or S0.3 or S0.4 ...).")
    args = parser.parse_args()

    target_study = args.target_study
    data_type=args.data_type
    ratio = args.ratio
    if data_type=="Meta_iTL":
        raw_target_funtion(ratio,target_study)
        TL_function(ratio,target_study)
    if data_type=="Raw":
        raw_funtion(target_study)
    print(f"{data_type} feature frequency calculation completed.")
