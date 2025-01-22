import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from xgboost import XGBClassifier
from Benchmark.Models.dataprocess import change_group

def Lodo_study_study_fig(data_,studies,studies_num,output_dir,group_name,SEED,model_type,**params):
    """
    study-to-study
    :param data_:
    :param studies:
    :param studies_num:
    :param output_dir:
    :param group_name:
    :return:
    """
    def get_kfold_auc(data, id1, id2):
        """
        study_to_study
        :param data:
        :param id1:
        :param id2:
        :return:
        """
        aucs = []
        for _ in range(2):
            #clf = RandomForestClassifier(random_state=seed).set_params(**param)
            if model_type=='xgb':
                clf = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss').set_params(**params)
            else:
                clf = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
            ### Train
            data_train = data[data['Study'] == id1]
            X_train = data_train.iloc[:, 2:]
            y_train = data_train.iloc[:, 0]
            ### Test
            data_test = data[data['Study'] == id2]
            X_test = data_test.iloc[:, 2:]
            y_test = data_test.iloc[:, 0]

            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)

            nor = preprocessing.MinMaxScaler()
            X_train = nor.fit_transform(X_train)
            X_test = nor.transform(X_test)
            for _ in range(2):
                clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        return np.mean(aucs)
    plot_data = []
    for id1 in studies:
        temp = []
        for id2 in studies:
            score=[]
            for _ in range(2):
                data_copy= data_.copy(deep=True)
                meta_feature = change_group(data_copy, group_name)
                roc_auc = get_kfold_auc(meta_feature, id1, id2)
                score.append(round(roc_auc, 2))
            temp.append(round(np.mean(score), 2))
        plot_data.append(temp)
    def get_kfold_auc3(data, id1):
        """
         # inter validation
        :param data:
        :param id1:
        :return:
        """
        data_train = data[data['Study'] == id1].reset_index(drop=True, inplace=False)
        X = data_train.iloc[:, 2:]
        y = data_train.iloc[:, 0]
        nor = preprocessing.MinMaxScaler()
        X = nor.fit_transform(X)
        aucs = []
        for _ in range(2):
            ss = StratifiedKFold(n_splits=5,shuffle=True)
            for train_index, test_index in ss.split(X, y):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                X_train, X_test = X[train_index], X[test_index]
                if model_type == 'xgb':
                    clf = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss').set_params(
                        **params)
                else:
                    clf = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
                nor = preprocessing.MinMaxScaler()
                X_train = nor.fit_transform(X_train)
                X_test = nor.transform(X_test)
                probas = clf.fit(X_train, y_train).predict_proba(X_test)
                fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
        return np.mean(aucs)

    plot_diagonal = []
    for id1 in studies:
        roc_auc=[]
        for _ in range(2):
            data_copy= data_.copy(deep=True)
            data = change_group(data_copy, group_name)
            k = get_kfold_auc3(data, id1)
            roc_auc.append(k)
        plot_diagonal.append(round(np.mean(roc_auc), 2))

    def get_kfold_auc2(data, id1):
        """
        LODO
        :param data:
        :param id1:
        :return:
        """
        aucs = []
        data_train = data[data['Study'] != id1]
        data_test = data[data['Study'] == id1]
        for _ in range(2):
            if model_type=='xgb':
                clf = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss').set_params(**params)
            else:
                clf = RandomForestClassifier(random_state=SEED, class_weight='balanced').set_params(**params)
            X_train = data_train.iloc[:, 2:]
            y_train = data_train.iloc[:, 0]

            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            X_test = data_test.iloc[:,2:]
            y_test = data_test.iloc[:, 0]
            nor = preprocessing.MinMaxScaler()
            X_train = nor.fit_transform(X_train)
            X_test = nor.transform(X_test)
            for _ in range(2):
                clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        return np.mean(aucs)
    LODO = []
    for id1 in studies:
        try:
            roc_auc=[]
            for _ in range(2):
                data_copy= data_.copy(deep=True)
                data = change_group(data_copy, group_name)
                n = get_kfold_auc2(data, id1)
                roc_auc.append(n)
        except:
            pass
        LODO.append(round(np.mean(roc_auc), 2))

    LODO.append(np.mean(LODO))
    LODO = np.array(LODO).reshape(1, studies_num+1)

    ######prepare plot_heatmap_data
    matrix = np.array(plot_data)
    main = np.array(plot_diagonal)
    np.fill_diagonal(matrix, main)

    matrix = np.c_[matrix, np.mean(matrix, axis=1)]
    a = np.mean(matrix, axis=0).reshape(1, studies_num+1)
    plot_matrix = np.around(np.r_[matrix, a], 2)

    plot_matrix[studies_num, studies_num] = np.mean(plot_matrix[0:studies_num, 0:studies_num])
    plot_matrix = np.around(np.r_[plot_matrix, LODO], 2)
    np.savetxt(output_dir+'plot_matrix.txt', plot_matrix, delimiter=',', fmt='%.2f')

    xLabel = studies+['Average']
    yLabel =  studies+['Average','LODO']
    fig = plt.figure(1, (studies_num, studies_num), dpi=300)
    font2 = {'family': 'Arial', 'weight': 'normal', 'size': 16}

    colors = ["#006DA3", "#0096E0", "#E6E600", "#FFFF3D", "#E04B00", "#A33600"]
    cmap = LinearSegmentedColormap.from_list("custom_blue_green_red", colors, N=256)

    ax = sns.heatmap(plot_matrix, vmin=0.5, vmax=1, cmap=cmap, linewidths=0.8, annot=True, annot_kws={'size': 10},
                     center=0.8, xticklabels=xLabel, yticklabels=yLabel)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=-30, horizontalalignment='right')
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=0)
    plt.title(group_name, font2)
    plt.ylabel('Training Cohorts', font2)
    plt.xlabel('Testing Cohorts', font2)
    plt.yticks(fontproperties='Arial', size=13)
    plt.xticks(fontproperties='Arial', size=13)
    fig.tight_layout()
    plt.show()
    fig.savefig(output_dir,dpi=300, format='pdf')
