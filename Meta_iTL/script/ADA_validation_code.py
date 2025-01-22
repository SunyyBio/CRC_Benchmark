import copy
import os
import pickle
import sys

import pandas as pd
import xgboost as xgb
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from Meta_iTL.function.MNN_function import split_data, select_feature
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore")
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'
figure_path = sys.path[1] + '\\Result\\figures\\'
param_path=sys.path[1] + '\\Result\\param\\'
xgb.set_config(verbosity=0)
warnings.filterwarnings('ignore')
SEED=42
def change_group(meta_feature, group_name):
    if group_name == 'CTR_ADA':
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "ADA" else 0)
    elif group_name == "CTR_CRC":
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    else:
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    return meta_feature
def tools_feature(df, feature_name):
    feature_list = df[feature_name].dropna()
    return feature_list.tolist()
def get_kfold_CTR_ADA(data, meta_group, model_type, **params):
    """

    :param data:
    :param meta_group:
    :param model_type:
    :param params:
    :return:
    """
    aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    splitor = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for train_index, test_index in splitor.split(data, meta_group):
        X_train, X_test = data.iloc[train_index].values, data.iloc[test_index].values
        y_train, y_test = meta_group.iloc[train_index].values, meta_group.iloc[test_index].values

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_type == 'rf':
            clf = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
        elif model_type == 'xgb':
            clf = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss').set_params(**params)
        else:
            raise ValueError("Invalid model type")

        probas = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        aucs.append(roc_auc)
        tprs.append(interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    return mean_auc, mean_tpr, mean_fpr, std_auc, tprs
def get_kfold_single_study(data, meta_group, model_type,output_file,n_repeats=50, **params):
    """
    Repeat 5-fold cross-validation for n_repeats times, calculate the average results, and save them locally.
    :param data: Feature data (DataFrame)
    :param meta_group: Label data (Series)
    :param model_type: Model type ('rf' or 'xgb')
    :param n_repeats: Number of repetitions
    :param output_file: File name to save the results
    :param params: Model parameters
    :return: Average AUC, average TPR, average FPR, AUC standard deviation,
    """
    overall_aucs = []
    overall_tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    results_list = []  # 用于保存每次重复的结果

    for repeat in range(n_repeats):
        aucs = []
        tprs = []
        splitor = StratifiedKFold(n_splits=5)

        for train_index, test_index in splitor.split(data, meta_group):
            X_train, X_test = data.iloc[train_index].values, data.iloc[test_index].values
            y_train, y_test = meta_group.iloc[train_index].values, meta_group.iloc[test_index].values

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if model_type == 'rf':
                clf = RandomForestClassifier(random_state=repeat).set_params(**params)
            elif model_type == 'xgb':
                clf = XGBClassifier(random_state=repeat, use_label_encoder=False, eval_metric='logloss').set_params(**params)
            else:
                raise ValueError("Invalid model type")

            probas = clf.fit(X_train, y_train).predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(interp(mean_fpr, fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        overall_aucs.append(mean_auc)
        overall_tprs.append(mean_tpr)

        results_list.append({
            "Repeat": repeat + 1,
            "Mean AUC": mean_auc,
            "AUCs": aucs
        })
    final_mean_tpr = np.mean(overall_tprs, axis=0)
    final_mean_tpr[-1] = 1.0
    final_mean_auc = np.mean(overall_aucs)
    final_std_auc = np.std(overall_aucs)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False)
    return final_mean_auc, final_mean_tpr, mean_fpr, final_std_auc, overall_tprs
def cross_validation_result():
    """
    Calculate the results of five-fold cross validation for CTRvsCRC and CTRvsADA, and draw the AUROC curve
    """
    analysis_level = "species"
    group_names = ["CTR_CRC", "CTR_ADA"]
    colors = ["blue", "green"]
    feature_name = "new_method"
    plt.figure(figsize=(10, 8))
    for i, group_name in enumerate(group_names):
        meta_feature_all= split_data(analysis_level, group_name, Raw="Raw_log")
        sig_feature = select_feature(analysis_level, group_name, feature_name, Raw="Raw_log")
        if group_name == "CTR_CRC":
            studies_of_interest = ['DE-CRC', 'AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC', "CHN_HK-CRC", "CHN_SH-CRC", "IND-CRC", "US-CRC"]
        else:
            studies_of_interest = ['AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC', "CHN_SH-CRC-3", "US-CRC-2", "US-CRC-3", "CHN_SH-CRC-2"]
        meta_feature = meta_feature_all[meta_feature_all["Study"].isin(studies_of_interest)]
        sig_feature_columns = set(sig_feature)
        meta_feature_columns = set(meta_feature.columns)
        intersection_columns = sig_feature_columns.intersection(meta_feature_columns)
        data = meta_feature.loc[:, list(intersection_columns)]
        y_data = change_group(meta_feature[["Group"]], group_name)
        with open(f"{param_path}/rf{group_name}{analysis_level}_best_params.pkl", 'rb') as f:
            loaded_best_param_rf = pickle.load(f)
        mean_auc, mean_tpr, mean_fpr, std_auc, tprs = get_kfold_CTR_ADA(data, y_data, "rf", **loaded_best_param_rf)

        tprs_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
        tprs_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 0)

        plt.plot(mean_fpr, mean_tpr, label=f'{group_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})', color=colors[i])
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('AUROC Comparison: CTR_CRC vs CTR_ADA', fontsize=14)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_dir = f"{figure_path}CTR_CRC_ADA_cross/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir+f"/AUROC_cross_validation_CRC_ADA.pdf", dpi=300)
    plt.show()
def LODO(target_study):
    analysis_level = "species"
    group_names = ["CTR_CRC", "CTR_ADA"]
    colors = ["blue", "green"]
    feature_name = "new_method"
    plt.figure(figsize=(10, 8))
    for i, group_name in enumerate(group_names):
        if group_name == "CTR_CRC":
            train_study = ['DE-CRC', 'AT-CRC', 'JPN-CRC', 'FR-CRC', 'ITA-CRC', "CHN_HK-CRC", "CHN_SH-CRC", "IND-CRC",
                           "US-CRC"]
        if group_name == "CTR_ADA":
            train_study = ['AT-CRC', 'JPN-CRC', 'FR-CRC', 'ITA-CRC', "CHN_SH-CRC-3", "US-CRC-2", "US-CRC-3",
                           "CHN_SH-CRC-2"]

        with open(f"{param_path}/rf{group_name}{analysis_level}_best_params.pkl", 'rb') as f:
            loaded_best_param_rf = pickle.load(f)
        meta_feature = split_data("species", group_name, "Raw_log")
        train_df = meta_feature[meta_feature['Study'].isin(train_study)]
        target_df = meta_feature[meta_feature['Study'].str.contains(target_study)]
        sig_feature = select_feature(analysis_level, group_name, feature_name, Raw="Raw_log")
        train_x = train_df.loc[:, sig_feature]
        train_y = change_group(train_df[["Group"]], group_name)
        target_x = target_df.loc[:, sig_feature]
        target_y = change_group(target_df[["Group"]], group_name)

        smote = SMOTE()
        train_x, train_y = smote.fit_resample(train_x, train_y)

        scaler = MinMaxScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        target_x_scaled = scaler.transform(target_x)
        auc_scores = []
        all_fpr = np.linspace(0, 1, 100)
        tprs = []
        for SEED in np.arange(10):
            #rf_model = RandomForestClassifier(random_state  **loaded_best_param_rf)
            rf_model = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**loaded_best_param_rf)
            rf_model.fit(train_x_scaled, train_y)
            y_proba = rf_model.predict_proba(target_x_scaled)[:, 1]
            auc_score = roc_auc_score(target_y, y_proba)
            auc_scores.append(auc_score)
            fpr, tpr, _ = roc_curve(target_y, y_proba)
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
            tprs.append(np.clip(np.interp(all_fpr, fpr, tpr), 0, 1))
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        print(auc_scores)
        mean_tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        mean_tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.plot(all_fpr, mean_tpr, color=colors[i], label=f'{group_name} (AUC = {np.mean(auc_scores):.2f})')
        plt.fill_between(all_fpr, mean_tpr_lower, mean_tpr_upper, color=colors[i], alpha=0.2, label=f'{group_name} STD')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUROC Curve - {target_study}')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right')
    plt.grid(True)
    save_dir = f"{figure_path}CTR_CRC_ADA_cross/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + f"/AUROC_LODO_CRC_ADA.pdf", dpi=300)
    plt.show()
    auc_df = pd.DataFrame(auc_scores, columns=['AUC_Score'])
    save_dir = save_dir+"auc_rf_lodo.csv"
    auc_df.to_csv(save_dir, index=False)
def cross_validation_baseline():
    # 数据准备和处理
    analysis_level = "species"
    group_name = "CTR_ADA"
    feature_dir = (
        f"{feature_path}raw/All_cohorts_feature.csv")
    feature = pd.read_csv(feature_dir)

    meta_feature_all = split_data(analysis_level, group_name, Raw="Raw_log")
    for study in [["CHN_SH-CRC-2"],["CHN_WF-CRC"]]:
        meta_feature = meta_feature_all[meta_feature_all["Study"].isin(study)]
        sig_feature = feature[study[0]+'_rf_optimal'].dropna().tolist()
        data = meta_feature.loc[:, sig_feature]
        y_data = change_group(meta_feature[["Group"]], group_name)

        with open(f"{param_path}/rf" + group_name + analysis_level + '_best_params.pkl', 'rb') as f:
            loaded_best_param_rf = pickle.load(f)
        output_file=f"{figure_path}/baseline/{study}_baseline.csv"
        model_result_rf = get_kfold_single_study(data, y_data, "rf",output_file, n_repeats=50, **loaded_best_param_rf)

LODO("CHN_WF-CRC")
cross_validation_result()
cross_validation_baseline()
