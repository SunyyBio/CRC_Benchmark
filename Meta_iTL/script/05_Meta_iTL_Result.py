import argparse
from pathlib import Path
import pandas as pd
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from Meta_iTL.function.MNN_function import plot_auc_from_file, split_data,train_model
import warnings
import sys
import os
warnings.filterwarnings("ignore")
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'
figure_path = sys.path[1] + '\\Result\\figures\\'
param_path=sys.path[1] + '\\Result\\param\\'
def prepare_dataframes(meta_feature, source_study, target_study, ratio):
    """
    Prepare training and testing dataframes.
    """
    source_df = meta_feature[meta_feature['Study'].str.contains("|".join(source_study))]
    target_df = meta_feature[meta_feature['Study'].str.contains(target_study)]

    train_path = f'{data_path}{"_".join(source_study)}_{target_study}/filtered_' \
                 f'{"_".join(source_study)}_{target_study}_{ratio}_train.csv'
    target_df_random = pd.read_csv(train_path)
    target_select_train = target_df[target_df.index.isin(target_df_random["Sample_ID"])]
    target_select_test = target_df[~target_df.index.isin(target_df_random["Sample_ID"])]

    return source_df, target_select_train, target_select_test
def raw_MNN_all_studys(source_study, target_study, ratio):
    """
    The original data is used as the training set, and the remaining target domain samples are used as the test set.
    """
    meta_feature= split_data("species", "CTR_ADA", "Raw_log")
    source_df, _, target_select_test = prepare_dataframes(meta_feature, source_study, target_study, ratio)
    feature_finally = pd.read_csv(f"{feature_path}raw/{'_'.join(source_study)}_optimal.csv")
    finally_frequency_feature=["Group"] + pd.Series(feature_finally[f"{'_'.join(source_study)}_synergistic"]).dropna().tolist()

    print("Raw data result")
    result_dir = f"{figure_path}Raw/{target_study}/{ratio}/AUC_data/"
    save_png = f"{figure_path}Raw/{target_study}/{ratio}/AUROC_figure/"
    result_file = f"{figure_path}Raw/{target_study}/{ratio}/All_result/"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    Path(save_png).mkdir(parents=True, exist_ok=True)
    Path(result_file).mkdir(parents=True, exist_ok=True)

    train_model(source_df, target_select_test, finally_frequency_feature, "raw_optimal", result_dir, result_file,
                "_".join(source_study) + "_" + target_study, target_study)
    plot_auc_from_file(result_dir + "rf" + ".pkl", save_png + "rf.pdf")
def raw_MNN_all_studys_main(target_study,ratio):
    """
    The original data is used as the training set, and the remaining target domain samples are used as the test set.
    """
    if target_study == "CHN_WF-CRC":
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-4",
                        "CHN_SH-CRC-2"]
    if target_study == "CHN_SH-CRC-4":
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "CHN_WF-CRC", "CHN_SH-CRC-2", "US-CRC-3"]

    raw_MNN_all_studys(source_study, target_study, ratio)

def MNN_all_studys(source_study,target_study,ratio):
    train_data = pd.read_csv( f'{data_path}{source_study}_{target_study}/filtered_' \
                 f'{source_study}_{target_study}_{ratio}_train.csv',index_col=0)
    test_data = pd.read_csv( f'{data_path}{source_study}_{target_study}/' \
                 f'{target_study}_{ratio}_test.csv',index_col=0)

    feature = pd.read_csv(f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/Meta_iTL_optimal.csv")

    feature_target = pd.read_csv(f"{feature_path}Meta_iTL/merge_feature/target_{target_study}/{ratio}/target_optimal.csv")
    frequency_feature_name2 = "Target_" + source_study + "_" + target_study + "_synergistic"
    finally_rf_optimal_feature2 = feature_target[frequency_feature_name2].dropna().tolist()

    frequency_feature_name = source_study + "_" + target_study + "_synergistic"
    finally_rf_optimal_feature = feature[frequency_feature_name].dropna().tolist()
    optimal_features_set = list(dict.fromkeys(finally_rf_optimal_feature2 + finally_rf_optimal_feature))
    finally_feature = ["Group"] + optimal_features_set

    print("Meta_iTL result")
    result_dir = f"{figure_path}Meta_iTL/{target_study}/{ratio}/AUC_data/"
    save_png = f"{figure_path}Meta_iTL/{target_study}/{ratio}/AUROC_figure/"
    result_file = f"{figure_path}Meta_iTL/{target_study}/{ratio}/All_result/"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    Path(save_png).mkdir(parents=True, exist_ok=True)
    Path(result_file).mkdir(parents=True, exist_ok=True)

    train_model(train_data, test_data,finally_feature, "MNN_all_optimal", result_dir, result_file,source_study + "_" + target_study, target_study)
    plot_auc_from_file(result_dir + "rf"+".pkl", save_png + "rf.pdf")
def MNN_all_studys_main(target_study,ratio):
    if target_study == "CHN_WF-CRC":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"
    if target_study == "CHN_SH-CRC-4":
        source_study = "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"
    MNN_all_studys(source_study,target_study,ratio)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions for the target domain across cohorts.")
    parser.add_argument("-t", "--target_study", required=True, help="Target study (e.g., CHN_WF-CRC or CHN_SH-CRC-4 ).")
    parser.add_argument("-d", "--data_type", required=True, help="Data Types (e.g., Meta_iTL or Raw).")
    parser.add_argument("-r", "--ratio", required=True, help="ratio (e.g., S0.2 or S0.3 or S0.4 ...).")

    args = parser.parse_args()
    target_study = args.target_study
    data_type=args.data_type
    ratio = args.ratio

    if data_type=="Meta_iTL":
        MNN_all_studys_main(target_study, ratio)
    if data_type=="Raw":
        raw_MNN_all_studys_main(target_study, ratio)
    print(f"{data_type} Synergistic select feature completed.")
