import os
import sys
import numpy as np
import pandas as pd
from pymrmr import mRMR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import argparse
import ast
import os
import sys
import numpy as np
import pandas as pd
from pymrmr import mRMR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'

"""
1.利用随机森林筛选出重要性的物种
输入：已经筛选的特征
输出随机森林认为最好的特征
"""
# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
# seed_everything(42)
def change_group(meta_feature,group_name):
    if group_name == 'CTR_ADA':
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "ADA" else 0)
    elif group_name == "CTR_CRC":
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    else:
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    return meta_feature
def feature_importance(data, num_repeats):
    """
    RF进行特种重要性排序
    :param data:数据
    :return:已排序的特征
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
            print("StratifiedKFold has an error, use KFold instead and adjust the number of folds")
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
def optimal_features(sorted_features, data):
    """
    选择最佳的特征组合
    :param sorted_features: 按照特征重要性排序后的特征列表
    :param data: 丰度表
    :param num_repeats: 进行交叉验证的重复次数
    :return: 最佳特征组合及对应的 AUC
    """
    features = data.iloc[:, 1:]
    features = features.astype(np.float32)
    target = data.iloc[:, 0]
    model = RandomForestClassifier(random_state=42,class_weight='balanced')
    sk = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

    best_auc = 0
    best_features = sorted_features
    feature_rank = []

    while len(sorted_features) > 30:
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
def mrdd(X_train,Y):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_train = pd.concat([Y.rename('Label'), X_train_scaled], axis=1)
    min_class_samples = min(np.bincount(Y))
    n_splits = min(5, min_class_samples)
    try:
        kf = StratifiedKFold(n_splits=n_splits)
    except ValueError as e:
        print("StratifiedKFold has an error, use KFold instead and adjust the number of folds")
        # If StratifiedKFold fails, adjust to KFold to ensure that the number of folds does not exceed the minimum number of class samples
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_performance = []
    feature_num=len(X_train.columns)
    for num_features in range(20, feature_num,1):
        features = mRMR(X_train, 'MIQ', num_features)
        X_train_selected = X_train[features]
        model = RandomForestClassifier(random_state=42)
        accuracies = cross_val_score(model, X_train_selected, Y, cv=kf, scoring='roc_auc')
        mean_accuracy = np.mean(accuracies)
        feature_performance.append((features, mean_accuracy))
    best_features, best_accuracy = max(feature_performance, key=lambda x: x[1])
    return best_features
def select_top_feature(filename,fre_quncy_the):
    feature = pd.read_csv(filename)
    # Flexible selection of frequency thresholds
    selected_feature = feature[feature["Frequency"] >fre_quncy_the].sort_values(
        by=["Frequency", "Feature"], ascending=[False, True]
    )["Feature"]
    return selected_feature
def save_feature_importance(filename, feature_importances_dict,fre_quncy_the):
    feature = pd.read_csv(filename)
    feature = feature[["Feature", "Frequency"]]
    #Flexible selection of frequency thresholds
    feature = feature[feature["Frequency"] >fre_quncy_the]
    importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient='index', columns=['Importance'])
    importances_df.index.name = 'Feature'
    merged_table = pd.merge(feature, importances_df, on='Feature', how='left')
    merged_table['total_value'] = merged_table['Frequency'] + merged_table['Importance']
    merged_table['Frequency_normalized'] = (merged_table['Frequency'] - merged_table['Frequency'].min()) / (
            merged_table['Frequency'].max() - merged_table['Frequency'].min())
    merged_table['Importance_normalized'] = (merged_table['Importance'] - merged_table['Importance'].min()) / (
            merged_table['Importance'].max() - merged_table['Importance'].min())
    merged_table['total_value_norm'] = merged_table['Frequency_normalized'] + merged_table['Importance_normalized']
    merged_table = merged_table.sort_values(by='total_value_norm', ascending=False)
    merged_table = merged_table.reset_index(drop=True)
    #merged_table.to_csv(filename, index=False)
    return merged_table
def Synergistic_select_feature(meta_feature,frequence_file,Synergistic_feature_file,frequency_thre,synergistic_feature_name):
    first_sig_feature = select_top_feature(frequence_file, frequency_thre).tolist()
    first_sig_feature = ['Group'] + first_sig_feature
    meta_sig_feature = meta_feature.loc[:, first_sig_feature]

    feature_order, feature_importances_dict = feature_importance(meta_sig_feature, 10)
    merged_table = save_feature_importance(frequence_file, feature_importances_dict, frequency_thre)
    total_value_feature = merged_table["Feature"].dropna().tolist()
    if len(total_value_feature) < 30:
        finally_features = total_value_feature
    else:
        X = meta_sig_feature[total_value_feature]
        finally_features = mrdd(X, meta_sig_feature.iloc[:, 0])
        finally_features, _, _ = optimal_features(finally_features, meta_sig_feature)
    finally_features_df = merged_table[merged_table['Feature'].isin(finally_features)]
    finally_features_df = finally_features_df.sort_values(by='total_value_norm', ascending=False)
    finally_features = finally_features_df['Feature'].dropna().tolist()

    directory = os.path.dirname(Synergistic_feature_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(Synergistic_feature_file):
        with open(Synergistic_feature_file, 'w') as file:
            file.write('rf')
    rf_features = pd.read_csv(Synergistic_feature_file)
    finally_series = pd.Series(finally_features)
    rows_to_add = len(finally_series) - len(rf_features)
    if rows_to_add > 0:
        extra_df = pd.DataFrame(np.nan, index=range(rows_to_add), columns=rf_features.columns)
        rf_features = pd.concat([rf_features, extra_df], ignore_index=True)
    rf_features[synergistic_feature_name] = finally_series
    rf_features.to_csv(Synergistic_feature_file, index=False)