import os
import sys

import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from finally_code.TL_pairs.function_dir.MNN_function import split_data, predict_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from Benchmark.Models.dataprocess import split_data,change_group
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
figure_path = sys.path[1] + '\\Result\\figures\\Fig03\\'
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

def tools_feature(df, tool):
    df.fillna(1, inplace=True)
    if tool=="lefse":
        feature_list = df[df[tool] < p_sig_lefse].index.tolist()
    else:
        feature_list = df[df[tool] < p_sig].index.tolist()
    return feature_list
def cross_val(train_data, features,group):
    meta_data=train_data.loc[:,features]
    np.random.seed(42)
    auc_all = []
    for _ in np.arange(30):
        train_data_copy = copy.deepcopy(meta_data)
        train_1 = change_group(train_data_copy, group)
        train_feature = train_1[features]
        X = train_feature.iloc[:, 1:].values
        y = train_feature.iloc[:, 0].values
        sss_outer = KFold(n_splits=5,shuffle=True,random_state=42)
        test_auc = []
        for train_index, test_index in sss_outer.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #model = RandomForestClassifier(random_state=42,n_estimators=400,max_depth=9,min_samples_leaf=2,min_samples_split=7)
            model = XGBClassifier( random_state=42)
            norm_obj = MinMaxScaler()
            X_train = norm_obj.fit_transform(X_train)
            X_test = norm_obj.transform(X_test)
            model.fit(X_train, y_train)
            y_pred_prob, recall_test, accuracy_test, precision_test, mcc_test, f1_test, roc_test = predict_score(model, X_test,y_test)
            test_auc.append(roc_test)
        auc_all.append(np.mean(test_auc))
    return np.mean(auc_all)
def cal_AUC_main(data_type):
    studys = [ "IND-CRC", "CHN_WF-CRC", "CHN_HK-CRC", "AT-CRC", "FR-CRC", "ITA-CRC", "US-CRC","CHN_SH-CRC","JPN-CRC","DE-CRC"]
    group = "CTR_CRC"
    tools = ["ALL","ancom","lefse","maaslin2", "RFECV","metagenomeSeq", "ttest", "wilcoxon"]
    meta_feature= split_data("species", group,data_type)
    results_df = pd.DataFrame(columns=tools)
    results_df["Study"] = studys
    results_df = results_df.set_index("Study")
    for study in studys:
        for tool in tools:
            source_df = meta_feature[(meta_feature['Study'].str.contains(study))]
            if tool=="ALL":
                feature=meta_feature.columns[:-2].tolist()
            else:
                df=pd.read_csv(feature_path + f'/species/{data_type}/{group}/{study}/adj_p_{group}.csv',index_col=0)
                feature = tools_feature(df, tool)
            if len(feature)!=0:
                all_feature = ["Group"] + feature
                all_features = pd.Series(all_feature).dropna().tolist()
                auc = cross_val(source_df, all_features,group)
            else:
                auc=0
            results_df.loc[study, tool] = auc
            print(f"Study: {study}, Tool: {tool}, AUC: {auc}")
    AUC_path = figure_path + f'/species/{data_type}/{group}/{p_sig}/single_AUC.csv'
    dir_path = os.path.dirname(AUC_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")
    results_df.to_csv(AUC_path)
def cal_AUC_all(data_type):
    study = ["CHN_HK-CRC", "AT-CRC", "IND-CRC", "CHN_SH-CRC", "FR-CRC","DE-CRC","ITA-CRC", "US-CRC","JPN-CRC", "CHN_WF-CRC"]
    group = "CTR_CRC"
    tools = ["ALL","ancom","lefse","maaslin2", "RFECV","metagenomeSeq", "ttest", "wilcoxon"]
    meta_feature = split_data("species", group,data_type)
    results_df = pd.DataFrame(columns=tools)
    for tool in tools:
        source_df = meta_feature[(meta_feature['Study'].str.contains("|".join(study)))]
        if tool=="ALL":
            feature=meta_feature.columns[:-2].tolist()
        else:
            df = pd.read_csv(feature_path + f'/species/{data_type}/{group}/adj_p_{group}.csv',index_col=0)
            feature = tools_feature(df, tool)
        sig_feature_columns = set(feature)
        meta_feature_columns = set(meta_feature.columns)
        intersection_columns = sig_feature_columns.intersection(meta_feature_columns)
        feature = list(intersection_columns)
        if len(feature)!=0:
            all_feature = ["Group"] + feature
            all_features = pd.Series(all_feature).dropna().tolist()
            auc = cross_val(source_df, all_features,group)
        else:
            auc=0
        results_df.loc["All_data", tool] = auc
        print(f"All, Tool: {tool}, AUC: {auc}")
    AUC_path = figure_path + f'/species/{data_type}/{group}/{p_sig}/All_AUC.csv'
    dir_path = os.path.dirname(AUC_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")
    results_df.to_csv(AUC_path)
def plot_auc(group, data_type):
    columns_order = ["ALL", "ancom","lefse", "maaslin2", "metagenomeSeq", "RFECV", "ttest", "wilcoxon"]
    rows_order = ["AT-CRC", "JPN-CRC", "FR-CRC", "CHN_WF-CRC", "ITA-CRC", "CHN_HK-CRC", "CHN_SH-CRC", "IND-CRC",
                  "US-CRC", "DE-CRC"]
    AUC_path = figure_path + f'/species/{data_type}/{group}/{p_sig}/single_AUC.csv'
    results_df = pd.read_csv(AUC_path,index_col='Study')
    results_df = results_df.loc[rows_order, columns_order]
    results_df.loc["Average"] = results_df.mean(axis=0)
    colors = ["#006DA3", "#007EBD", "#008FD6", "#0088CC", "#47C2FF", "#C7ECFF", "#FFEBE0", "#FFDAC7", "#FFA061"]
    cmap = LinearSegmentedColormap.from_list("custom_blue_green_red", colors, N=256)
    plt.figure(figsize=(5, 4))
    heatmap = sns.heatmap(results_df, annot=True, fmt=".2f", cmap=cmap, linewidths=.5,
                          cbar_kws={'label': 'AUC Value'}, annot_kws={'size': 7})
    heatmap.set_title('Heatmap of AUC Values for Different Studies and Tools', pad=20)
    figer_path=figure_path + f'/species/{data_type}/{group}/{p_sig}/AUC_single_study_heatmap.pdf'
    plt.savefig(figer_path,format='pdf')
    plt.show()
if __name__ == "__main__":
    p_sig=0.01
    p_sig_lefse=0.05
    #cal_AUC_main(data_type="Batch")
    #plot_auc("CTR_CRC",data_type="Batch")
    cal_AUC_all(data_type="Batch")