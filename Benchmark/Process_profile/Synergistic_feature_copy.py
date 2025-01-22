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

"""
benchmark中的按照特征频率选特征
lefse的特征名称是”AT_lefse“
MaAsLin2的特征名称是”AT_MaAsLin“
all的特征是”AT_all“
AT_FR
AT_FR_ITA
AT_FR_ITA_US
AT_FR_ITA_US_JPN
"""

def optimal_features(sorted_features, data, analysis_level):
    """
    Select the optimal feature combination based on AUC.
    :param sorted_features: List of features sorted by importance.
    :param data: Feature abundance table.
    :param analysis_level: Taxonomic level for analysis.
    :return: Best feature combination, corresponding AUC, and feature ranking.
    """
    base_path = sys.path[1]+'/Result/param1/'
    features = data.iloc[:, 1:]
    features = features.astype(np.float32)
    target = data.iloc[:, 0]
    param_df = pd.read_csv(base_path+"/"+analysis_level+ "_best_params.csv")
    study_params = param_df.loc[param_df['model'] == "RandomForest"]
    best_param_str = study_params['best_params'].values[0]
    param_dict = ast.literal_eval(best_param_str)
    adjusted_param_dict = {
        param: value[0] if isinstance(value, list) and len(value) == 1 else value for param, value in
        param_dict.items()
    }
    model = RandomForestClassifier(random_state=42,class_weight='balanced').set_params(**adjusted_param_dict)
    sk = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
    best_auc = 0
    best_features = sorted_features
    feature_rank = []
    while len(sorted_features) >30:
        aucs = []
        for ni in sorted_features:
            temp = sorted_features.copy()
            temp.remove(ni)
            cv_scores = cross_val_score(model, features[temp], target, cv=sk, scoring='roc_auc')
            mean_cv_score = np.mean(cv_scores)
            aucs.append((temp, mean_cv_score))
        select, cv_scores = max(aucs, key=lambda x: x[1])
        if cv_scores > best_auc:
            best_auc = cv_scores
            best_features = select
        sorted_features = select
        feature_rank.append((best_features, best_auc))
    return best_features, best_auc, feature_rank
def feature_importance(data, num_repeats):
    """
    Rank features by importance using Random Forest.
    :param data: Feature abundance table with target labels.
    :param num_repeats: Number of repetitions for cross-validation.
    :return: Sorted feature list and feature importance dictionary.
    """
    feature_order = []
    X = data.iloc[:, 1:]
    y = data.iloc[:,0]
    importance_lists = []
    min_class_samples = min(np.bincount(y))
    n_splits = min(5, min_class_samples)

    for _ in range(num_repeats):
        importance_list = []
        try:
            skf = StratifiedKFold(n_splits=n_splits)
            splits = skf.split(X, y)
        except ValueError as e:
            n_splits = min(2, min_class_samples)
            skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = skf.split(X)
        for train_index, _ in splits:
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            model = RandomForestClassifier(random_state=42)
            norm_obj = MinMaxScaler()
            X_train = norm_obj.fit_transform(X_train)
            model.fit(X_train, y_train)
            feature_importances = model.feature_importances_
            importance_list.append(feature_importances)
        importance_lists.append(importance_list)

    average_importance = np.mean(np.mean(importance_lists, axis=0),axis=0)
    feature_names = X.columns
    feature_importances_dict = {feature_name: importance for feature_name, importance in zip(feature_names, average_importance)}
    sorted_features = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        feature_order.append(feature)
    return feature_order,feature_importances_dict
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
        for feature_name in lefse_single_feature + MaAsLin_feature +ancom_feature:
            if feature_name in feature_study_mapping:
                feature_study_mapping[feature_name].append(study)
            else:
                feature_study_mapping[feature_name] = [study]
    lefse_all_feature = select_all_study_feature(analysis_level, group_name,"lefse",data_type)
    MaAsLin_all_feature =select_all_study_feature(analysis_level, group_name, "maaslin2",data_type)
    ancom_all_feature=select_all_study_feature(analysis_level, group_name, "ancom",data_type)
    study_single_lefse_MaAsLin +=lefse_all_feature+MaAsLin_all_feature+ancom_all_feature
    feature_frequencies = Counter(study_single_lefse_MaAsLin)
    df_frequencies = pd.DataFrame(list(feature_frequencies.items()), columns=["Feature", "Frequency"])
    df_frequencies["Study Count"] = df_frequencies["Feature"].map(lambda x: len(set(feature_study_mapping.get(x, []))))
    df_frequencies["Frequency"] = df_frequencies["Feature"].map(feature_frequencies)
    base_directory = sys.path[1] + '/Result/feature/'+analysis_level+"/"+data_type+"/"+group_name
    print(base_directory)
    output_freq_file = base_directory+"/feature_frequencies.csv"
    directory = os.path.dirname(output_freq_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_frequencies.to_csv(output_freq_file, index=False)

def select_top_feature(filename,frequency_thre):
    feature=pd.read_csv(filename)
    feature = feature[feature["Frequency"] >frequency_thre]
    select_feature = feature["Feature"]
    return select_feature
def save_feature_importance(filename,feature_importances_dict,frequency_thre):
    feature = pd.read_csv(filename)
    feature = feature[feature["Frequency"] >frequency_thre]
    feature=feature[["Feature","Frequency","Study Count"]]
    importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient='index', columns=['Importance'])
    importances_df.index.name = 'Feature'
    merged_table = pd.merge(feature, importances_df, on='Feature', how='left')
    merged_table['total_value'] = merged_table['Frequency'] + merged_table['Importance']
    merged_table['Frequency_normalized'] = (merged_table['Frequency'] - merged_table['Frequency'].min()) / (
                merged_table['Frequency'].max() - merged_table['Frequency'].min())
    merged_table['Importance_normalized'] = (merged_table['Importance'] - merged_table['Importance'].min()) / (
                merged_table['Importance'].max() - merged_table['Importance'].min())

    merged_table['total_value_norm'] =merged_table['Frequency_normalized'] + merged_table['Importance_normalized']
    merged_table = merged_table.sort_values(by='total_value_norm', ascending=False)
    merged_table = merged_table.reset_index(drop=True)
    #merged_table.to_csv(filename)
    return merged_table

def mrdd(X_train,Y):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index)
    X_train = pd.concat([Y.rename('Label'), X_train_scaled], axis=1)

    min_class_samples = min(np.bincount(Y))
    n_splits = min(5, min_class_samples)
    try:
        kf = StratifiedKFold(n_splits=n_splits)
    except ValueError as e:
        print("StratifiedKFold has an error, use KFold instead and adjust the number of folds")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    feature_performance = []
    feature_num = len(X_train.columns)
    #num_to_add = max(1, feature_num // 5)
    for num_features in range(30, feature_num, 1):
        features =  pymrmr.mRMR(X_train,  'MIQ', num_features)
        X_train_selected = X_train[features]
        model = RandomForestClassifier()
        accuracies = cross_val_score(model, X_train_selected, Y, cv=kf, scoring='roc_auc')
        mean_accuracy = np.mean(accuracies)
        feature_performance.append((features, mean_accuracy))
    best_features, best_accuracy = max(feature_performance, key=lambda x: x[1])
    return best_features
def raw_select_feature(meta_feature, study, frequence_file, feature_file,class_dir,frequency_thre):
    """
    Perform feature optimization on raw data.
    :param study: Name of the study
    :param frequence_file: File to save the frequency of features
    :param feature_file: Final feature file
    :return: Optimal features identified by Random Forest (RF)
    """
    first_sig_feature = select_top_feature(frequence_file,frequency_thre).tolist()
    first_sig_feature = ['Group'] + list(first_sig_feature)
    meta_sig_feature = meta_feature.loc[:, first_sig_feature]

    feature_order, feature_importances_dict = feature_importance(meta_sig_feature, 10)
    merged_table = save_feature_importance(frequence_file, feature_importances_dict,frequency_thre)
    total_value_feature = merged_table["Feature"].dropna().tolist()

    if len(total_value_feature) < 30:
        finally_features = total_value_feature
    else:
        X = meta_sig_feature[total_value_feature]
        finally_features = mrdd(X, meta_sig_feature.iloc[:, 0])
        finally_features, _, _ = optimal_features(finally_features, meta_sig_feature, class_dir)

    directory = os.path.dirname(feature_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(feature_file):
        with open(feature_file, 'w') as file:
            file.write('rf')
    rf_features = pd.read_csv(feature_file)
    finally_series = pd.Series(finally_features)

    rows_to_add = len(finally_series) - len(rf_features)
    if rows_to_add > 0:
        extra_df = pd.DataFrame(np.nan, index=range(rows_to_add), columns=rf_features.columns)
        rf_features = pd.concat([rf_features, extra_df], ignore_index=True)
    rf_features["_".join(study) + "_rf_optimal"] = finally_series
    rf_features.to_csv(feature_file, index=False)
def raw_main():
    analysis_levels=["class","order","family","genus","species","t_sgb","ko_gene","uniref_family"]
    group_name="CTR_ADA"
    #studys = [["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "CHN_WF-CRC", "CHN_SH-CRC", "CHN_HK-CRC", "DE-CRC", "IND-CRC","US-CRC"]]
    studys = [["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "CHN_WF-CRC", "CHN_SH-CRC-2","US-CRC-2", "CHN_SH-CRC-3","US-CRC-3"]]
    for analysis_level in analysis_levels:
        for study in studys:
            data_type="Raw_log"
            base_directory =feature_path+analysis_level + "/" + data_type + "/" + group_name
            feature_file =base_directory+"/feature.csv"
            print(feature_file)
            frequence_file = base_directory + "/feature_frequencies.csv"
            meta_feature = split_data( analysis_level,  group_name, data_type)
            meta_feature = meta_feature[meta_feature['Study'].str.contains("|".join(study))]
            meta_feature = change_group(meta_feature,  group_name)
            raw_select_feature(meta_feature,study, frequence_file, feature_file,analysis_level)
if __name__ == "__main__":
    # Select optimal features
    raw_main()

