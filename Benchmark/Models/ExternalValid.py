import os
import pickle
import sys
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from Benchmark.Models.dataprocess import split_data, change_group, select_feature
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
xgb.set_config(verbosity=0)
warnings.filterwarnings('ignore')
SEED=42
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'
figure_path=sys.path[1]+'\\Result\\figures\\Fig04\\'

def plot_roc(mean_fpr, mean_tpr, tprs_upper, tprs_lower,mean_auc,std_auc,save_dir):
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + "_AUC.pdf")
    plt.show()
def get_kfold(data, meta_group, model_type, **params):
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
        aucs.append(roc_auc)
        tprs.append(interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    return mean_auc, mean_tpr, mean_fpr, std_auc, tprs
def get_kfold_auc_op(data, meta_group,model_name, **params):
    mean_auc,mean_tpr,mean_fpr,std_auc,tprs=get_kfold(data, meta_group,model_name, **params)
    return mean_auc
def validation_auc(meta_data, valid_data, data, group_name, valid_studies_list, features, AUC_dir, model_name, **params):
    plt.figure(figsize=(10, 8))
    all_results = {}
    for valid_studies in valid_studies_list:
        meta_data_subset = meta_data[meta_data["Study"].isin(valid_studies)]
        common_columns = valid_data.columns.intersection(meta_data_subset["Sample_ID"].values)
        if common_columns.empty:
            raise ValueError(f"No overlapping columns between valid_data and Sample_IDs in {valid_studies}")

        valid_data_subset = valid_data[common_columns]
        valid_data_subset = valid_data_subset.transpose()

        merged_df = pd.merge(
            valid_data_subset,
            meta_data_subset[["Group", "Sample_ID", "Study"]],
            left_index=True,
            right_on='Sample_ID',
            how="inner"
        )

        train_data = data[~data["Study"].isin(["CHN_SH-CRC-3", "CHN_SH-CRC-4"])]
        train_X = train_data.loc[:, features]
        train_Y = change_group(train_data[["Group"]], group_name)
        valid_X = merged_df.loc[:, features]
        valid_Y = change_group(merged_df[["Group"]], group_name)

        # 数据标准化
        scaler = MinMaxScaler()
        train_X = scaler.fit_transform(train_X)
        valid_X = scaler.transform(valid_X)

        aucs = []
        std_tprs = []
        all_fpr = np.linspace(0, 1, 100)

        for SEED in range(20):
            if model_name == 'rf':
                clf = RandomForestClassifier(random_state=SEED).set_params(**params)
            elif model_name == 'xgb':
                clf = XGBClassifier(random_state=SEED).set_params(**params)
            else:
                raise ValueError("Invalid model type")

            probas = clf.fit(train_X, train_Y).predict_proba(valid_X)
            fpr, tpr, _ = roc_curve(valid_Y, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
            std_tprs.append(np.clip(np.interp(all_fpr, fpr, tpr), 0, 1))

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_tpr = np.mean(std_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(std_tprs, axis=0)

        lower_bound = np.maximum(mean_tpr - std_tpr, 0)
        upper_bound = np.minimum(mean_tpr + std_tpr, 1)

        all_results[','.join(valid_studies)] = {
            "AUC_mean": mean_auc,
            "AUC_std": std_auc,
            "AUC_scores": aucs
        }

        plt.plot(
            all_fpr,
            mean_tpr,
            label=f"Studies: {','.join(valid_studies)} (AUC = {mean_auc:.2f} ± {std_auc:.2f})"
        )

        plt.fill_between(
            all_fpr,
            lower_bound,
            upper_bound,
            alpha=0.3,
            label=f"Studies: {','.join(valid_studies)} ± STD"
        )

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Validation Studies')
    plt.legend(loc='lower right')
    plt.grid(True)

    if not os.path.exists(AUC_dir):
        os.makedirs(AUC_dir)
    output_path = os.path.join(AUC_dir, "ROC_combined.pdf")
    plt.savefig(output_path, dpi=1000)
    print(f"ROC curve saved to: {output_path}")
    plt.show()

    results_df = pd.DataFrame.from_dict(all_results, orient='index').reset_index()
    auc_scores_path = os.path.join(AUC_dir, "AUC_scores.csv")
    results_df.to_csv(auc_scores_path, index=False)
    print(f"AUC scores saved to: {auc_scores_path}")
    return all_results

def get_kfold_auc(data, meta_group, AUC_dir,model_name,**params):
    mean_auc,mean_tpr,mean_fpr,std_auc,tprs=get_kfold(data, meta_group, model_name,**params)
    tprs_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
    tprs_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 0)
    plot_roc( mean_fpr, mean_tpr, tprs_upper, tprs_lower, mean_auc,std_auc,AUC_dir)
    return mean_auc
def finally_all_model_result(analysis_level,group_name,feature_name,data_type):

    meta_feature_all= split_data(analysis_level, group_name, data_type)
    sig_feature = select_feature(analysis_level, group_name, feature_name, data_type)

    studies_of_interest = ['DE-CRC', 'AT-CRC', 'JPN-CRC', 'FR-CRC','CHN_WF-CRC', 'ITA-CRC', "AT-CRC", "CHN_HK-CRC","CHN_SH-CRC","IND-CRC", "US-CRC"]
    meta_feature=meta_feature_all[meta_feature_all["Study"].isin(studies_of_interest)]
    sig_feature_columns = set(sig_feature)
    meta_feature_columns = set(meta_feature.columns)
    intersection_columns = sig_feature_columns.intersection(meta_feature_columns)
    data = meta_feature.loc[:, list(intersection_columns)]
    y_data = change_group(meta_feature[["Group"]], group_name)

    with open(sys.path[1]+'/Result/param/'+"rf"+group_name+analysis_level+'_best_params.pkl', 'rb') as f:
        loaded_best_param_rf = pickle.load(f)
    with open(sys.path[1]+'/Result/param/'+"xgb"+group_name+analysis_level+'_best_params.pkl', 'rb') as f:
        loaded_best_param_xg = pickle.load(f)
    save_figure_dir = os.path.join(figure_path, data_type, analysis_level, "valid")

    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)

    model_result_rf = get_kfold_auc(data, y_data,save_figure_dir+analysis_level+"_rf_Kfold",model_name="rf",**loaded_best_param_rf)
    model_result_xg = get_kfold_auc(data, y_data,save_figure_dir+analysis_level+"_xg_Kfold", model_name="xgb",**loaded_best_param_xg)

    valid_studies = [["CHN_SH-CRC-2"], ["CHN_SH-CRC-3"]]
    meta = pd.read_csv(data_path+"/meta.csv")
    if data_type in ["Raw", "Raw_log"]:
        data = data_path + f"{analysis_level}/feature_rare_ext_{group_name}.csv"
        valid_data = pd.read_csv(data, sep='\t', index_col=0)
        if data_type == "Raw_log":
            valid_data = np.log(valid_data + 1)
    validation = validation_auc(meta, valid_data, meta_feature_all, group_name, valid_studies,
                                list(intersection_columns),
                                save_figure_dir+analysis_level+"rf_external_valid", model_name="rf", **loaded_best_param_rf)
    print("rf", validation)
    validation = validation_auc(meta, valid_data, meta_feature_all, group_name, valid_studies,
                                list(intersection_columns),
                                save_figure_dir+analysis_level + "xg_external_valid", model_name="xgb", **loaded_best_param_xg)
    print("xg", validation)
if __name__ == "__main__":
    analysis_level = "genus"
    group_name = "CTR_CRC"
    feature_name = "new_method"
    data_type = "Raw_log"
    finally_all_model_result(analysis_level,group_name,feature_name,data_type)
