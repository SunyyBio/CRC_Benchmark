import copy
import sys
import time
import numpy as np
import pandas as pd
from gevent import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from Benchmark.Models.dataprocess import split_data,change_group

# Set up paths
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
figure_path = sys.path[1] + '\\Result\\figures\\Fig03\\'
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

def raw_select_feature_all(class_dir, study, feature_file, group, data_type):
    """
    Select features for all studies using Random Forest feature importance ranking.
    Normalize the features based on frequency and importance.
    """
    meta_feature = split_data(class_dir, group, data_type)
    meta_feature = meta_feature[meta_feature['Study'].str.contains("|".join(study))]

    meta_sig_feature = change_group(meta_feature, group)
    Y, X_train = meta_sig_feature["Group"], meta_sig_feature.iloc[:, :-2]
    model = RandomForestClassifier(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    selector = RFECV(model, step=10, cv=cv).fit(X_train, Y)
    selected_features_names = X_train.columns[selector.support_].tolist()

    directory = os.path.dirname(feature_file)
    os.makedirs(directory, exist_ok=True)
    if not os.path.exists(feature_file):
        rf_features = pd.DataFrame(["IFECV"], columns=['Column_Name'])
    else:
        rf_features = pd.read_csv(feature_file)

    finally_series = pd.Series(selected_features_names)
    rows_to_add = len(finally_series) - len(rf_features)
    if rows_to_add > 0:
        extra_df = pd.DataFrame(np.nan, index=range(rows_to_add), columns=rf_features.columns)
        rf_features = pd.concat([rf_features, extra_df], ignore_index=True)
    rf_features["_".join(study) + "_IFECV"] = finally_series
    rf_features.to_csv(feature_file, index=False)


def raw_select_feature_single(class_dir, study, feature_file, group, data_type):
    """
    Select features for each individual study using Random Forest feature importance ranking.
    Normalize the features based on frequency and importance.
    """
    for study_i in study:
        meta_feature = split_data(class_dir, group, data_type)
        meta_feature = meta_feature[meta_feature['Study'].str.contains(study_i)]
        meta_sig_feature = change_group(meta_feature, group)
        Y, X_train = meta_sig_feature["Group"], meta_sig_feature.iloc[:, :-2]
        model = RandomForestClassifier(random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        selector = RFECV(model, step=10, cv=cv).fit(X_train, Y)
        selected_features_names = X_train.columns[selector.support_].tolist()

        directory = os.path.dirname(feature_file)
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(feature_file):
            rf_features = pd.DataFrame(["IFECV"], columns=['Column_Name'])
        else:
            rf_features = pd.read_csv(feature_file)

        finally_series = pd.Series(selected_features_names)
        rows_to_add = len(finally_series) - len(rf_features)
        if rows_to_add > 0:
            extra_df = pd.DataFrame(np.nan, index=range(rows_to_add), columns=rf_features.columns)
            rf_features = pd.concat([rf_features, extra_df], ignore_index=True)
        rf_features[study_i + "_IFECV"] = finally_series
        rf_features.to_csv(feature_file, index=False)


# Main execution block
for class_dir in ["species"]:
    group = "CTR_CRC"
    data_type = "Raw"
    feature_file = feature_path + f"/{class_dir}/{data_type}/{group}/All_features_tools/RFECV.csv"
    study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "CHN_WF-CRC", "CHN_SH-CRC", "CHN_HK-CRC", "DE-CRC", "IND-CRC",
             "US-CRC"]
    start_time = time.time()
    raw_select_feature_all(class_dir, study, feature_file, group, data_type)
    raw_select_feature_single(class_dir, study, feature_file, group, data_type)
    elapsed_time = time.time() - start_time
    print(data_type, class_dir, "Selected IFECV features for all studies")
