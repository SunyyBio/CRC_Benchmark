import sys
import numpy as np
import copy
import os
import pickle
import random
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import  SMOTE
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, accuracy_score, precision_score, matthews_corrcoef, \
    f1_score, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']= str(SEED)
new_path = os.path.abspath(os.path.join(__file__, "../../../"))
sys.path[1] = new_path
feature_path = sys.path[1] + '\\Benchmark\\Result\\feature\\'
data_path = sys.path[1] + '\\Meta_iTL\\Result\\data\\'
param_path = sys.path[1] + '\\Meta_iTL\\Result\\param\\'

def select_feature(analysis_level, group_name, feature_name,Raw):
    if Raw == "Raw":
        if feature_name == "new_method":
            feature_dir = (
                    f"{feature_path}/{analysis_level}/Raw/{group_name}/feature.csv")
        else:
            feature_dir = (
                f"{feature_path}/{analysis_level}/Raw/{group_name}/adj_p_{group_name}.csv")
    elif Raw == "Raw_log":
        if feature_name == "new_method":
            feature_dir = (
                    f"{feature_path}/{analysis_level}/Raw_log/{group_name}/feature.csv")
        else:
            feature_dir = (
            f"{feature_path}/{analysis_level}/Raw_log/{group_name}/adj_p_{group_name}.csv")
    else:
        if feature_name == "new_method":
            feature_dir = (
                    f"{feature_path}/{analysis_level}/Batch/{group_name}/feature.csv")
        else:
            feature_dir = (
                f"{feature_path}/{analysis_level}/Batch/{group_name}/adj_p_{group_name}.csv")
    feature = pd.read_csv(feature_dir, index_col=0)

    if feature_name == "all":
        feature.fillna(1, inplace=True)
        feature_select = feature.index.tolist()
    elif feature_name == "union":
        feature.fillna(1, inplace=True)
        feature_select = feature[feature.lt(0.01).all(axis=1)].index.tolist()
    elif feature_name=="new_method":
        if group_name=="CTR_CRC":
            feature_select = feature['FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_CHN_WF-CRC_CHN_SH-CRC_CHN_HK-CRC_DE-CRC_IND-CRC_US-CRC_rf_optimal'].dropna().tolist()
        if group_name == "CTR_ADA":
            feature_select=feature['AT-CRC_JPN-CRC_FR-CRC_CHN_WF-CRC_ITA-CRC_CHN_SH-CRC-3_US-CRC-3_CHN_SH-CRC-2_US-CRC-2_rf_optimal'].dropna().tolist()
    elif feature_name=="lefse":
        feature.fillna(1, inplace=True)
        feature_select = feature[feature[feature_name] < 0.05].index.tolist()
    else:
        feature_select = feature[feature[feature_name] < 0.01].index.tolist()
    return feature_select
def split_data(analysis_level, groups, Raw):
    new_path = os.path.abspath(os.path.join(__file__, "../../../"))
    sys.path[1] = new_path
    feature_path = sys.path[1] + '\\Benchmark\\Result\\feature\\'
    data_path = sys.path[1] + '\\Meta_iTL\\Result\\data\\'
    param_path = sys.path[1] + '\\Meta_iTL\\Result\\param\\'
    if Raw in ["Raw", "Raw_log"]:
        data = data_path + f"feature_rare_{groups}.csv"
        feature = pd.read_csv(data, sep='\t', index_col=0)
        if Raw == "Raw_log":
            feature = np.log(feature+1)
    if groups=="CTR_CRC":
        meta = pd.read_csv(sys.path[1] + '/Benchmark/Result/data/meta.csv')
    else:
        meta = pd.read_csv(data_path + "/meta.csv")
    merged_df = pd.merge(feature, meta[["Group", "Sample_ID", "Study"]], left_index=True, right_on='Sample_ID',how="inner")
    merged_df.set_index('Sample_ID', inplace=True)
    return merged_df
def balance_samples(meta_feature, group1, group2):
    grouped = meta_feature.groupby("Study")
    selected_samples = pd.DataFrame()
    for name, group in grouped:
        ada_samples = group[group["Group"] == group1]
        ctr_samples = group[group["Group"] == group2]
        min_count = min(len(ada_samples), len(ctr_samples))
        if min_count > 0:
            ada_samples = ada_samples.sample(n=min_count, replace=False)
            ctr_samples = ctr_samples.sample(n=min_count, replace=False)
            selected_samples = selected_samples.append(ada_samples)
            selected_samples = selected_samples.append(ctr_samples)
    selected_samples.drop("Study",axis=1,inplace=True)
    return selected_samples
def change_group(meta_feature,group_name):
    if group_name == 'CTR_ADA':
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "ADA" else 0)
    elif group_name == "CTR_CRC":
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    else:
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    return meta_feature
def plot_auc_from_file(dir_,save_png):
    # Load all results from the file
    all_results = {}
    with open(dir_, 'rb') as f:
        try:
            while True:
                result = pickle.load(f)
                all_results.update(result)
        except EOFError:
            pass

    plt.figure(figsize=(10, 6))
    for result_name, (mean_tpr, mean_fpr, mean_auc,std_auc) in all_results.items():
        print(result_name)

        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{result_name} (AUC = {mean_auc:.3f} ± {std_auc:.2f})',color=color)

        # tprs_upper = np.minimum(mean_tpr + std_auc, 1)
        # tprs_lower = np.maximum(mean_tpr - std_auc, 0)
        # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2)

    # Plotting diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Setting plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_png, dpi=300)
    plt.show()
def bayesian_optimise_rf(X, y, clf_kfold, n_iter, init_points=5):
    def rf_crossval(n_estimators, max_features, max_depth, max_samples):
        return clf_kfold(
            data=X,
            meta_group=y,
            model_name="rf",
            n_estimators=int(n_estimators),
            max_samples=max(min(max_samples, 0.999), 0.1),
            max_features=max(min(max_features, 0.999), 0.1),
            max_depth=int(max_depth),
            bootstrap=True
        )
    optimizer = BayesianOptimization(
        random_state=SEED,
        f=rf_crossval,
        pbounds={
            "n_estimators": (10, 500),
            "max_features": (0.1, 0.999),
            "max_samples": (0.1, 0.999),
            "max_depth": (1, 10)
        }
    )
    optimizer.maximize(n_iter=n_iter, init_points=init_points)
    print("Final result:", optimizer.max)
    return optimizer.max
def bayesian_optimise_xgb(X, y, clf_kfold, n_iter, init_points=5):
    def xgb_crossval(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
        # 仅传递XGBoost支持的参数
        return clf_kfold(
            data=X,
            meta_group=y,
            model_name="xgb",
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    optimizer = BayesianOptimization(
        random_state=SEED,
        f=xgb_crossval,
        pbounds={
            "n_estimators": (10, 500),
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.2, 1.0),
            "colsample_bytree": (0.2, 1.0)
        }
    )
    optimizer.maximize(n_iter=n_iter, init_points=init_points)
    print("Final result:", optimizer.max)
    return optimizer.max
def get_kfold(data, meta_group, model_type, **params):
    aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    splitor = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if model_type == 'rf':
        clf = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
    elif model_type == 'xgb':
        clf = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss').set_params(**params)
    else:
        raise ValueError("Invalid model type")
    meta_group = meta_group['Group']
    meta_group = meta_group.astype(int)  # 转换为整数类型

    for train_index, test_index in splitor.split(data, meta_group):
        X_train, X_test = data.iloc[train_index].values, data.iloc[test_index].values
        y_train, y_test = meta_group.iloc[train_index].values, meta_group.iloc[test_index].values

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
def change_group(meta_feature,group_name):
    if group_name == 'CTR_ADA':
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "ADA" else 0)
    elif group_name == "CTR_CRC":
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    else:
        meta_feature['Group'] = meta_feature['Group'].apply(lambda x: 1 if x == "CRC" else 0)
    return meta_feature
def test_TL_target_30(train_data, test_data, features, feature_name,outfile,result_list_file,study_name,target_study,model_type,**params):
    result_list = []
    auc_all = []
    fpr_all = []
    tpr_all = []
    score_f1 = []
    score_mcc = []
    score_accuracy = []
    score_precision = []
    score_recall = []
    train_data_copy = copy.deepcopy(train_data)
    test_data_copy = copy.deepcopy(test_data)
    meta_train = change_group(train_data_copy, "CTR_ADA")
    meta_test = change_group(test_data_copy, "CTR_ADA")
    source_train = meta_train[~meta_train["Study"].str.contains(target_study)].reindex(columns=meta_train.columns)
    target_train = meta_train[meta_train["Study"].str.contains(target_study)].reindex(columns=meta_train.columns)
    source_train=source_train[features]
    target_train=target_train[features]
    all_auc_results = []
    for _ in np.arange(50):
        test_feature = meta_test[features]
        if len(test_feature)>25:
            if target_study == "CHN_WF-CRC":
                test_size_30 = 25
            elif target_study == "CHN_SH-CRC-2":
                test_size_30 = 26
            _,selected_test_feature = train_test_split(test_feature, test_size=test_size_30, stratify=test_feature['Group'])
        else:
            selected_test_feature=test_feature
        smote = SMOTE(random_state=SEED)
        try:
            X1_train, y1_train = smote.fit_resample(source_train.iloc[:, 1:], source_train.iloc[:, 0].values)
        except Exception as e:
            X1_train, y1_train = source_train.iloc[:, 1:], source_train.iloc[:, 0].values
        try:
            X2_train, y2_train = smote.fit_resample(target_train.iloc[:, 1:], target_train.iloc[:, 0].values)
        except Exception as e:
            X2_train, y2_train=target_train.iloc[:, 1:], target_train.iloc[:, 0].values
        norm = MinMaxScaler()
        X1_train_norm = norm.fit_transform(X1_train)
        X2_train_norm = norm.transform(X2_train)
        X_test_norm = norm.transform(selected_test_feature.iloc[:, 1:])
        if model_type == 'rf':
            # Step 1: Train an initial model on a larger dataset
            model = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
            model.fit(X1_train_norm, y1_train)
            # # Step 2: Generate pseudo labels and pseudo label probabilities
            pseudo_labels = model.predict_proba(X2_train_norm)[:,1]
            enhanced_data = pd.DataFrame(X2_train_norm.copy())
            enhanced_data.columns = enhanced_data.columns.astype(str)
            enhanced_data['pseudo_labels'] = pseudo_labels
            # Step 4: Create an augmented training set (using pseudo labels)
            X_enhanced = enhanced_data
            # Step 5: Retrain a new random forest model using augmented data
            final_rf_model = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
            final_rf_model.fit(X_enhanced, y2_train)
            # Step 6: Predict pseudo labels for the test set
            pseudo_labels_test = model.predict_proba(X_test_norm)[:,1]
            if isinstance(X_test_norm, np.ndarray):
                X_test_norm = pd.DataFrame(X_test_norm)
            # Step 7: Add pseudo labels to the test set DataFrame
            X_test_norm.columns = X_test_norm.columns.astype(str)
            X_test_norm['pseudo_labels'] = pseudo_labels_test
            # Step 8: Use the final model to make predictions
            y_pred1 = final_rf_model.predict_proba(X_test_norm)[:, 1]
            # Step 9: Calculate AUC score
            auc_score = roc_auc_score(selected_test_feature.iloc[:, 0].values, y_pred1)
            fpr, tpr, _ = roc_curve(selected_test_feature.iloc[:, 0].values, y_pred1)

            auc_all.append(auc_score)
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            final_pred=final_rf_model.predict(X_test_norm)
            def calculate_metrics(y_true, y_pred):
                recall = recall_score(y_true, y_pred)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                return recall, accuracy, precision, mcc, f1
            recall, accuracy, precision, mcc, f1 = calculate_metrics(selected_test_feature.iloc[:, 0].values, final_pred)
            score_recall.append(recall)
            score_accuracy.append(accuracy)
            score_precision.append(precision)
            score_mcc.append(mcc)
            score_f1.append(f1)
        else:
            raise ValueError("Invalid model type")
        all_auc_results.append(auc_score)
    result_AUC = {
        "study_name": study_name,
        "target_name": target_study,
        "AUC": np.mean(auc_all),
        "f1": np.mean(score_f1),
        "mcc": np.mean(score_mcc),
        "accuracy": np.mean(score_accuracy),
        "precision": np.mean(score_precision),
        "recall": np.mean(score_recall)}
    result_list.append(result_AUC)
    result_AUC_list_df = pd.DataFrame(result_list)

    directory = os.path.dirname(result_list_file)
    os.makedirs(directory, exist_ok=True)
    result_path=result_list_file+"/"+model_type + ".csv"
    result_AUC_list_df.to_csv(result_path, mode="a")

    auc_results_df = pd.DataFrame({"AUC": all_auc_results})
    auc_results_file = result_list_file + "/"+model_type + "_50_AUC_results.csv"
    auc_results_df.to_csv(auc_results_file, index=False)
    print(f"AUC results saved to {auc_results_file}")
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_all, tpr_all)], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_all)
    all_results = {}
    all_results[feature_name] = (mean_tpr, mean_fpr, mean_auc, std_auc)
    directory_pkl = os.path.dirname(outfile + model_type + ".pkl")
    os.makedirs(directory_pkl, exist_ok=True)
    with open(outfile + model_type + ".pkl", 'ab') as f:
        pickle.dump(all_results, f)
    print(f"test AUC = {round(np.mean(auc_all), 3)}")
def test_Raw_target_30(train_data, test_data, features, feature_name, outfile, result_list_file, study_name,target_study, model_type, **params):
    result_list = []
    auc_all = []
    fpr_all = []
    tpr_all = []
    score_f1 = []
    score_mcc = []
    score_accuracy = []
    score_precision = []
    score_recall = []

    train_data_copy = copy.deepcopy(train_data)
    test_data_copy = copy.deepcopy(test_data)
    meta_train = change_group(train_data_copy, "CTR_ADA")
    meta_test = change_group(test_data_copy, "CTR_ADA")

    train_feature = meta_train[features]
    all_auc_results = []
    for SEED in np.arange(50):
        test_feature = meta_test[features]
        if len(test_feature) > 25:
            if target_study == "CHN_WF-CRC":
                test_size_30 = 24
            elif target_study == "CHN_SH-CRC-2":
                test_size_30 = 26
            _, test_feature = train_test_split(test_feature, test_size=25, stratify=test_feature['Group'])
        smote = SMOTE(random_state=SEED)
        try:
            X_train, y_train = smote.fit_resample(train_feature.iloc[:, 1:], train_feature.iloc[:, 0].values)
        except Exception as e:
            X_train, y_train = train_feature.iloc[:, 1:], train_feature.iloc[:, 0].values

        norm = MinMaxScaler()
        X_train_norm = norm.fit_transform(X_train)
        X_test_norm = norm.transform(test_feature.iloc[:, 1:])
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
        elif model_type == 'xgb':
            model = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss').set_params(
                **params)
        else:
            raise ValueError("Invalid model type")
        model.fit(X_train_norm, y_train)
        y_pred_B_valid = model.predict_proba(X_test_norm)[:, 1]
        auc_score = roc_auc_score(test_feature.iloc[:, 0].values, y_pred_B_valid)
        auc_all.append(auc_score)
        fpr, tpr, _ = roc_curve(test_feature.iloc[:, 0].values, y_pred_B_valid)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        y_pred = model.predict(X_test_norm)
        recall = recall_score(test_feature.iloc[:, 0].values, y_pred)
        accuracy = accuracy_score(test_feature.iloc[:, 0].values, y_pred)
        precision = precision_score(test_feature.iloc[:, 0].values, y_pred)
        mcc = matthews_corrcoef(test_feature.iloc[:, 0].values, y_pred)
        f1 = f1_score(test_feature.iloc[:, 0].values, y_pred)

        score_f1.append(f1)
        score_mcc.append(mcc)
        score_accuracy.append(accuracy)
        score_precision.append(precision)
        score_recall.append(recall)
        all_auc_results.append(auc_score)

    result_AUC = {
        "study_name": study_name,
        "target_name": target_study,
        "AUC": np.mean(auc_all),
        "f1": np.mean(score_f1),
        "mcc": np.mean(score_mcc),
        "accuracy": np.mean(score_accuracy),
        "precision": np.mean(score_precision),
        "recall": np.mean(score_recall)}
    result_list.append(result_AUC)

    result_AUC_list_df = pd.DataFrame(result_list)

    directory = os.path.dirname(result_list_file)
    os.makedirs(directory, exist_ok=True)
    result_path=directory+"/"+model_type + ".csv"
    result_AUC_list_df.to_csv(result_path, mode="a")

    auc_results_df = pd.DataFrame({"AUC": all_auc_results})
    auc_results_file = result_list_file + model_type + "_50_AUC_results.csv"
    auc_results_df.to_csv(auc_results_file, index=False)
    print(f"50 AUC results saved to {auc_results_file}")

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_all, tpr_all)], axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_all)
    all_results = {}
    all_results[feature_name] = (mean_tpr, mean_fpr, mean_auc, std_auc)
    directory_pkl = os.path.dirname(outfile + model_type + ".pkl")
    os.makedirs(directory_pkl, exist_ok=True)
    with open(outfile + model_type + ".pkl", 'ab') as f:
        pickle.dump(all_results, f)
    print(f" test AUC = {round(np.mean(auc_all), 3)}")
def train_model(train_data, test_data, features, feature_name,outfile,result_list_file,study_name,target_study):
    with open(param_path+target_study+"rfall_studys_CTR_ADA_species_best_params.pkl", 'rb') as f:
        loaded_best_param_rf = pickle.load(f)
    if feature_name in ["MNN_study_optimal", "MNN_all_optimal", "raw_target_optimal"]:
        test_TL_target_30(train_data, test_data, features, feature_name, outfile, result_list_file, study_name,
                          target_study, "rf", **loaded_best_param_rf)
    else:
        test_Raw_target_30(train_data, test_data, features, feature_name, outfile, result_list_file, study_name,
                           target_study, "rf", **loaded_best_param_rf)
