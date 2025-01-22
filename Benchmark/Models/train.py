import ast
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from modelfunction import MLPClassifier as MLP
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import joblib
def select_best_parmam(model,params_df, model_name):
    study_params = params_df.loc[params_df['model'] == model_name]
    best_param_str = study_params['best_params'].values[0]
    param_dict = ast.literal_eval(best_param_str)
    adjusted_param_dict = {
        param: value[0] if isinstance(value, list) and len(value) == 1 else value for param, value in
        param_dict.items()
    }
    model = model.__class__(**adjusted_param_dict)
    return model
def train(
    model_name,
    X,y,test_size, save_dir,params_df,
    seed,
):
    assert model_name in ['mlp', 'xgb', 'rf', 'svm', 'knn']
    if model_name == 'mlp':
        model = MLP(device='cpu',)
        model = select_best_parmam(model, params_df, "MLP")
    elif model_name == 'xgb':
        model = XGBClassifier()
        model=select_best_parmam(model,params_df,"XGBoost")
    elif model_name == 'rf':
        model = RandomForestClassifier()
        model = select_best_parmam(model, params_df, "RandomForest")
    elif model_name == 'svm':
        model = SVC()
        model = select_best_parmam(model, params_df, "SVM")
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        model = select_best_parmam(model, params_df,"KNN")

    X_train, x_test, Y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=True,stratify=y,random_state=seed)
    smote = SMOTE(random_state=seed)
    x_train, y_train = smote.fit_resample(X_train, Y_train)
    norm_obj = MinMaxScaler()
    x_train = norm_obj.fit_transform(x_train)
    x_test = norm_obj.transform(x_test)

    model.fit(x_train, y_train)
    joblib.dump(model, '{}/checkpoints/test_{}.ckpt'.format(save_dir, seed))
    s_test = model.predict_proba(x_test)[:, 1] if model_name != 'svm' else model.decision_function(x_test)
    p_test = model.predict(x_test)
    _save_results(y_test, p_test, s_test, save_dir, seed)
def _save_results(labels, predictions, scores, save_dir, seed):
    df_results = pd.DataFrame()
    df_results['Y Label'] = labels
    df_results['Y Predicted'] = predictions
    df_results['Predicted score'] = scores
    os.makedirs(save_dir+'/results/',exist_ok=True)
    df_results.to_csv('{}/results/test_{}.csv'.format(save_dir, seed), index=False)

