import os
import pandas as pd
from misc import get_metrics
from train import train
import copy
import tqdm
from dataprocess import change_group

"""
comb_dir: path to store output results
result_AUC: store results
analysis_level:species,t_sgb,ASV
feature_method: method to select features
"""
def  train_module(meta_feature,comb_dir,group_name,feature_method,param_df):
    result_AUC = pd.DataFrame(columns=[group_name, 'mlp', 'xgb', 'rf', 'svm', 'knn'])  # 输出结果到txt文件
    result_AUC.set_index(group_name, inplace=True)
    for model_name in ['mlp','xgb', 'rf', 'svm', 'knn']:
        comb_dir = comb_dir
        mode_dir = model_name
        save_dir = '{}{}/{}'.format(comb_dir, group_name, mode_dir)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + '/checkpoints', exist_ok=True)
        os.makedirs(save_dir + '/results', exist_ok=True)
        kwargs = {}
        num_runs = 100
        for iter_id in tqdm.tqdm(range(num_runs)):
            meta_feature_copy = copy.deepcopy(meta_feature)
            meta_feature_ = change_group(meta_feature_copy, group_name)
            test_size = 0.3
            kwargs['seed'] = iter_id
            X = meta_feature_.iloc[:, 2:].values
            y = meta_feature_.iloc[:, 0]
            args = (model_name, X, y, test_size, save_dir, param_df)
            train(*args, **kwargs)
        list_csv = os.listdir(save_dir + '/results')
        list_csv = ['{}/results/{}'.format(save_dir, fn) for fn in list_csv]
        y_true_all, y_pred_all, scores_all = [], [], []
        for fn in list_csv:
            df = pd.read_csv(fn)
            y_true_all.append(df['Y Label'].to_numpy())
            y_pred_all.append(df['Y Predicted'].to_numpy())
            scores_all.append(df['Predicted score'].to_numpy())
        met_all = get_metrics(y_true_all, y_pred_all, scores_all)
        for k, v in met_all.items():
            if k not in ['Confusion Matrix']:
                result_AUC.at[k,model_name] = v[0]
        for k, v in met_all.items():
            if k not in ['Confusion Matrix']:
                print('{}:\t{:.4f} \u00B1 {:.4f}'.format(k, v[0], v[1]).expandtabs(20))
    os.makedirs(comb_dir+'/Result/'+feature_method+'/',exist_ok=True)
    result_AUC.to_csv(comb_dir+'/Result/'+feature_method+'/'+group_name+'_indices.txt', sep='\t')

