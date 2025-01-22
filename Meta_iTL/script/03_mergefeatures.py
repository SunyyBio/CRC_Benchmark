import argparse
import csv
import os
import pickle
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd

new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

def target_rf_feature(dir_):
    """
    目标域随机森林筛选的特征

    :param dir_: Path to the CSV file containing the feature selection results.
    :return: A Pandas Series of selected features.
    """
    feature_select = pd.read_csv(dir_)
    return feature_select["Feature"]

def lefse_feature(dir_):
    """
    筛选用来下游分析的LEfSe特征

    :param dir_: Path to the LEfSe results file.
    :return: A list of selected features.
    """
    feature_select = pd.read_csv(dir_, sep=' ')
    if not feature_select.empty:
        filtered_features = feature_select[feature_select['P.unadj'] < 0.05]
        return filtered_features.index.str.split('|').str[7].tolist()
    return []

def MaAslin_feature(dir_):
    """
    筛选用来下游分析的MaAsLin显著特征

    :param dir_: Path to the MaAsLin results file.
    :return: A list of significant features (p-value < 0.05).
    """
    sig_feature = pd.read_csv(dir_, sep='\t')
    return sig_feature[sig_feature["pval"] < 0.05]['feature'].tolist()

def Ancom_feature(dir_):
    """
    筛选用来下游分析的ANCOM特征

    :param dir_: Path to the ANCOM results file.
    :return: A list of detected taxa IDs.
    """

    sig_feature = pd.read_csv(dir_, sep='\t')
    if sig_feature.shape[0] > 1:
        return sig_feature[sig_feature["detected_0.6"]]['taxa_id'].tolist()
    return []

    return feature

def select_feature(input_file, output_file, feature_name, feature_function):
    """
    Generic function to select features based on the provided feature function.

    :param input_file: Path to the input file containing feature selection results.
    :param output_file: Path to the output CSV file where features will be stored.
    :param feature_name: The name of the feature column to add to the output file.
    :param feature_function: The function used to extract features from input_file.
    """
    # Extract features using the provided function
    feature_select = feature_function(input_file)
    feature_series = pd.Series(feature_select)

    # Load the existing output DataFrame
    df = pd.read_csv(output_file)

    # Ensure sufficient rows in the DataFrame
    rows_to_add = len(feature_series) - len(df)
    if rows_to_add > 0:
        extra_df = pd.DataFrame(np.nan, index=range(rows_to_add), columns=df.columns)
        df = pd.concat([df, extra_df], ignore_index=True)

    # Add feature series to the DataFrame
    df[feature_name] = feature_series
    df.to_csv(output_file, index=False)

def select_lefse_feature(input_file, output_file, feature_name):
    """Select LEfSe features and save them to the output file."""
    select_feature(input_file, output_file, feature_name, lefse_feature)

def select_MaAsLin2_feature(input_file, output_file, feature_name):
    """Select MaAsLin features and save them to the output file."""
    select_feature(input_file, output_file, feature_name, MaAslin_feature)

def select_Ancom_feature(input_file, output_file, feature_name):
    """Select ANCOM features and save them to the output file."""
    select_feature(input_file, output_file, feature_name, Ancom_feature)


def select_all_feature(MaAsLin2_select_file, lefse_select_file, Ancom_select_file, studys, study, feature_file):
    """
    Select and combine features from various methods for downstream analysis.

    :param MaAsLin2_select_file: Path to the MaAsLin2 results file.
    :param lefse_select_file: Path to the LEfSe results file.
    :param Ancom_select_file: Path to the ANCOM results file.
    :param studys: List of study names for melting DataFrame.
    :param study: Name of the current study.
    :param feature_file: Path to the output feature CSV file.
    """
    # Extract features from the various input files
    MaAsLin2_feature_select = MaAslin_feature(MaAsLin2_select_file)
    lefse_feature_select = lefse_feature(lefse_select_file)
    Ancom_feature_select = Ancom_feature(Ancom_select_file)

    # Read the existing CSV file
    df = pd.read_csv(feature_file)

    # Initialize a new column for combined features
    column_name = f"{study}_lefse_MaAsLin_Ancom"
    df[column_name] = np.nan

    # Create a unique list of features from existing studies
    if studys:
        melted_df = pd.melt(df, value_vars=studys)
        unique_list = list(set(melted_df['value'].tolist()))
    else:
        unique_list = []

    # Combine all feature sets while ensuring uniqueness
    combined_feature_set = set(MaAsLin2_feature_select).union(
        lefse_feature_select, Ancom_feature_select, unique_list
    )

    # Convert to list and filter out any empty strings
    combined_feature_list = [feature for feature in combined_feature_set if feature]

    # Adjust DataFrame size as necessary
    max_length = max(len(combined_feature_list), len(df))
    if len(df) < max_length:
        extra_rows = max_length - len(df)
        extra_df = pd.DataFrame(np.nan, index=range(extra_rows), columns=df.columns)
        df = pd.concat([df, extra_df], ignore_index=True)

    # Fill in the new feature column
    #df.loc[:len(combined_feature_list) - 1, column_name] = combined_feature_list


    df[column_name] = df[column_name].astype('object')
    df.loc[:len(combined_feature_list) - 1, column_name] = combined_feature_list

    # Save the updated DataFrame to the CSV file
    df.to_csv(feature_file, index=False)
def select_target_feature(MaAsLin2_select_file, lefse_select_file, Ancom_select_file, studys, feature_file):
    """
    Select target features for downstream analysis.

    :param MaAsLin2_select_file: Path to the MaAsLin2 results file.
    :param lefse_select_file: Path to the LEfSe results file.
    :param Ancom_select_file: Path to the ANCOM results file.
    :param studys: Name of the current study.
    :param feature_file: Path to the output feature CSV file.
    """
    # Extract features from the various input files
    MaAsLin2_feature_select = MaAslin_feature(MaAsLin2_select_file)
    lefse_feature_select = lefse_feature(lefse_select_file)
    Ancom_feature_select = Ancom_feature(Ancom_select_file)

    # Combine all feature sets while ensuring uniqueness
    combined_feature_set = set(MaAsLin2_feature_select).union(
        lefse_feature_select, Ancom_feature_select
    )

    combined_feature_list = list(combined_feature_set)
    lefse_feature_series = pd.Series(combined_feature_list)

    # Prepare to save features
    column_name = studys
    if not os.path.exists(feature_file):
        # Create a new DataFrame and save if file doesn't exist
        selected_features_df = pd.DataFrame({column_name: lefse_feature_series})
        selected_features_df.to_csv(feature_file, index=False)
    else:
        # Update existing DataFrame
        df = pd.read_csv(feature_file)
        rows_to_add = len(lefse_feature_series) - len(df)
        if rows_to_add > 0:
            extra_df = pd.DataFrame(np.nan, index=range(rows_to_add), columns=df.columns)
            df = pd.concat([df, extra_df], ignore_index=True)

        df[column_name] = lefse_feature_series  # Assign values to the new column
        df.to_csv(feature_file, index=False)
def select_single_study_feature(lefse_study_dir, MaAslin_study_dir, Ancom_study_dir, all_studys, output_file):
    for study in all_studys:
        # source_lefse_MaAsLin
        lefse_file = lefse_study_dir + study + "_lefse.csv"
        MaAsLin_file = MaAslin_study_dir + study + "_MaAslin/all_results.tsv"
        Ancom_file = Ancom_study_dir + study + "_Ancom.csv"
        studys = []
        select_all_feature(MaAsLin_file, lefse_file, Ancom_file, studys, study, output_file)
def raw_study_select_feature(source_study, all_studys,save_dir):
    """
    Merge features from the raw dataset without sample selection.

    :param study: The current study to process.
    :param all_studys: A list of all studies for which features should be gathered.
    :param save_dir: The directory where features will be saved.
    :return: None
    """
    base_dir_input_file =feature_path+f"/raw/all_feature/"
    output_file = feature_path+f"/raw/{save_dir}.csv"

    # Input file for LEfSe features
    lefse_input_file = os.path.join(base_dir_input_file, f"{source_study}_lefse.csv")

    # Create output CSV file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([source_study])  # Write header

    # Selecting features from LEfSe
    select_lefse_feature(lefse_input_file, output_file, f"{source_study}_lefse")

    # Selecting features from MaAsLin
    maaslin_input_file = os.path.join(base_dir_input_file, f"{source_study}_MaAslin/all_results.tsv")
    select_MaAsLin2_feature(maaslin_input_file, output_file, f"{source_study}_MaAsLin")

    # Selecting features from ANCOM
    ancom_input_file = os.path.join(base_dir_input_file, f"{source_study}_Ancom.csv")
    select_Ancom_feature(ancom_input_file, output_file, f"{source_study}_Ancom")



    # # File paths for combined features
    # maaslin2_select_file = os.path.join(base_dir_input_file, f"{source_study}_MaAslin/all_results.tsv")
    # lefse_select_file = os.path.join(base_dir_input_file, f"{source_study}_lefse.csv")
    # ancom_select_file = os.path.join(base_dir_input_file, f"{source_study}_Ancom.csv")
    #
    # # Combine features into a single output
    # combined_study_names = [f"{s}_lefse_MaAsLin_Ancom" for s in all_studys]
    # select_all_feature(maaslin2_select_file, lefse_select_file, ancom_select_file, combined_study_names,
    #                    f"Raw_{source_study}_all", output_file)

def MNN_all_studys(source_study, target_study, all_studys, ratio):
    """
    Merge features from multiple source studies.

    :param source_study: Name of the source study.
    :param target_study: Name of the target study.
    :param all_studys: List of all studies used.
    :param ratio: Ratio for the merge.
    :param save_dir: Directory to save merged features.
    :return: None
    """
    # Create directory for merging features
    directory = Path(f"{feature_path}/Meta_iTL/{source_study}_{target_study}/{ratio}/")
    feature_dir = Path(f"{feature_path}/Meta_iTL/merge_feature/target_{target_study}/{ratio}/")
    feature_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    feature_file = feature_dir / f"Meta_iTL_feature.csv"
    if not feature_file.exists():
        df = pd.DataFrame({'target_study': [target_study]})
        df.to_csv(feature_file, index=False)
    output_file = feature_file
    # Selecting features from LEfSe
    input_file = directory/f'Meta_iTL_all_cohort_lefse.csv'
    select_lefse_feature(input_file, output_file, "Meta_iTL_all_cohort_lefse")
    # Selecting features from MaAsLin
    input_file = directory/f'Meta_iTL_all_cohort_MaAslin/all_results.tsv'
    select_MaAsLin2_feature(input_file, output_file, f"Meta_iTL_all_cohort_MaAsLin")
    # Selecting features from ANCOM
    input_file =  directory/f'Meta_iTL_all_cohort_Ancom.csv'
    select_Ancom_feature(input_file, output_file, f"Meta_iTL_all_cohort_Ancom")

    # # Analyze each study separately
    # slect_feature_dir=f"{feature_path}/Meta_iTL/{source_study}_{target_study}/{ratio}/"
    # select_single_study_feature(slect_feature_dir,slect_feature_dir,slect_feature_dir, all_studys+[target_study],output_file)

    for study in all_studys:

        # Selecting features from LEfSe
        input_file = directory/f'{study}_lefse.csv'
        select_lefse_feature(input_file, feature_file, f"{study}_lefse")
        # Selecting features from MaAsLin
        input_file = directory/f'{study}_MaAslin/all_results.tsv'
        select_MaAsLin2_feature(input_file, feature_file, f"{study}_MaAsLin")
        # Selecting features from ANCOM
        input_file = directory/f'{study}_Ancom.csv'
        select_Ancom_feature(input_file, feature_file, f"{study}_Ancom")

    # Merge all found features
    # MaAsLin2_select_file = directory/f'Meta_iTL_all_cohort_MaAslin/all_results.tsv'
    # lefse_select_file = directory/f'Meta_iTL_all_cohort_lefse.csv'
    # Ancom_select_file =  directory/f'Meta_iTL_all_cohort_Ancom.csv'
    #
    # studys = [f"{study}_lefse_MaAsLin_Ancom" for study in all_studys]
    # select_all_feature(MaAsLin2_select_file, lefse_select_file, Ancom_select_file, studys,
    #                    f"Meta_iTL_feature_all", output_file)
def Target_feature(ratio, source_study, target_study):
    """
    Merge features for the target dataset from all studies for transfer learning.

    :param ratio: The target domain's ratio.
    :param source_study: Source domain name.
    :param target_study: Target domain name.
    :param save_dir: Directory for saving data (MNN_all_studys or study_to_study).
    :return: None
    """

    base_path =  f"{feature_path}/Meta_iTL/{source_study}_{target_study}/{ratio}/"
    # Create target directory
    feature_dir = Path(f"{feature_path}/Meta_iTL/merge_feature/target_{target_study}/{ratio}/")
    feature_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    feature_file = feature_dir / f"target_{target_study}_feature.csv"
    if not feature_file.exists():
        df = pd.DataFrame({'target_study': [target_study]})
        df.to_csv(feature_file, index=False)
    # Selecting features from LEfSe
    input_file = f'{base_path}{target_study}_lefse.csv'
    select_lefse_feature(input_file, feature_file, f"{target_study}_lefse")
    # Selecting features from MaAsLin
    input_file = f'{base_path}{target_study}_MaAslin/all_results.tsv'
    select_MaAsLin2_feature(input_file, feature_file, f"{target_study}_MaAsLin")
    # Selecting features from ANCOM
    input_file = f'{base_path}{target_study}_Ancom.csv'
    select_Ancom_feature(input_file,feature_file, f"{target_study}_Ancom")

    # MaAsLin2_select_file = f"{base_path}{target_study}_MaAslin/all_results.tsv"
    # lefse_select_file = f"{base_path}{target_study}_lefse.csv"
    # Ancom_select_file = f"{base_path}{target_study}_Ancom.csv"
    # studys = f'target_feature_all'
    #
    # if not feature_file.exists():
    #     df = pd.DataFrame({'target_study': [target_study]})
    #     df.to_csv(feature_file, index=False)
    # select_target_feature(MaAsLin2_select_file, lefse_select_file, Ancom_select_file, studys, feature_file)
def MNN_all_studys_function(ratio,target_study):
    """
    Combine features corresponding to transfer learning based on all research
    :return:
    """
    if target_study=="CHN_SH-CRC-4":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"
        target_study = "CHN_SH-CRC-4"
        all_studys = ["FR-CRC","AT-CRC","ITA-CRC","JPN-CRC","US-CRC-2","US-CRC-3","CHN_WF-CRC","CHN_SH-CRC-2"]
    elif target_study=="CHN_WF-CRC":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"
        all_studys = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-4",
                      "CHN_SH-CRC-2"]

    Target_feature(ratio, source_study, target_study)
    MNN_all_studys(source_study, target_study, all_studys, ratio)

def raw_select_feature_function(ratio,target_study):
    """
    Characteristics of the original data of the sample without Meta_iTL
    :return:
    """
    def merge_all_cohrts(target_study):
        if target_study == "CHN_SH-CRC-4":
            source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"
            all_studys = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_WF-CRC",
                          "CHN_SH-CRC-2"]
        elif target_study == "CHN_WF-CRC":
            source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"
            all_studys = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-4",
                          "CHN_SH-CRC-2"]
        save_dir="merge_all_cohort_feature"
        raw_study_select_feature(source_study, all_studys,save_dir)

    def merge_single_cohrts():
        source_studys = [["FR-CRC"], ["ITA-CRC"], ["JPN-CRC"], ["US-CRC-2"], ["US-CRC-3"], ["CHN_WF-CRC"], ["AT-CRC"],
                         ['CHN_SH-CRC-4'], ['CHN_SH-CRC-2']]
        for source_study in source_studys:
            save_dir = "merge_all_cohort_feature"
            raw_study_select_feature(source_study[0], source_study, save_dir)

    merge_all_cohrts(target_study)
    merge_single_cohrts()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge feature.")
    parser.add_argument("-t", "--target_study", required=True, help="Target study (e.g., CHN_SH-CRC-4 or CHN_WF-CRC).")
    parser.add_argument("-d", "--data_type", required=True, help="Meta_iTL (e.g., Meta_iTL or Raw).")
    parser.add_argument("-r", "--ratio", required=True, help="S0.4 (e.g., S0.2 or S0.3 or S0.4 or S0.5 or S0.6).")
    args = parser.parse_args()

    target_study = args.target_study
    data_type=args.data_type
    ratio=args.ratio

    if data_type=="Meta_iTL":
        MNN_all_studys_function(ratio,target_study)
    if data_type=="Raw":
        raw_select_feature_function(target_study)
    print(f"{data_type} merge feature calculation completed.")