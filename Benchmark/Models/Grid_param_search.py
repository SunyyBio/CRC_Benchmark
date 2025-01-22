import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from dataprocess import split_data
from modelfunction import MLPClassifier as MLP
from sklearn.metrics import make_scorer, roc_auc_score

# Add project root directory to sys.path for module imports
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

def change_group(meta_feature: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """
    Convert group labels into binary format.

    Parameters
    ----------
    meta_feature : pd.DataFrame
        Metadata with group labels.
    group_name : str
        Name of the group for binary classification.

    Returns
    -------
    pd.DataFrame
        Updated metadata with binary labels.
    """
    if group_name == 'CTR_ADA':
        meta_feature['Group'] = meta_feature['Group'].apply(
            lambda x: 1 if x == "ADA" else 0
        )
    elif group_name == "CTR_CRC":
        meta_feature['Group'] = meta_feature['Group'].apply(
            lambda x: 1 if x == "CRC" else 0
        )
    else:
        meta_feature['Group'] = meta_feature['Group'].apply(
            lambda x: 1 if x == "CRC" else 0
        )
    return meta_feature

def select_param():
    """
    Perform parameter selection for multiple models (RandomForest, SVM, XGBoost, KNN, MLP)
    across different taxonomic (and other) levels. Each model uses a predefined grid of
    hyperparameters, and GridSearchCV is performed using RepeatedStratifiedKFold (5-fold).
    Best parameters are saved as CSV files for each level (analysis).
    """
    param_grids = [
        # {
        #     'name': 'RandomForest',
        #     'model': RandomForestClassifier(random_state=42),
        #     'params': {
        #         'n_estimators': np.arange(100, 1000, 10),
        #         'max_depth': np.arange(2, 10, 1),
        #         'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
        #         'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8],
        #     },
        # },
        # {
        #     'name': 'SVM',
        #     'model': SVC(probability=True, random_state=42),
        #     'params': {
        #         'C': [0.01, 0.1, 0.5, 1, 5, 10],
        #         'kernel': ['linear', 'rbf'],
        #     },
        # },
        # {
        #     'name': 'XGBoost',
        #     'model': XGBClassifier(
        #         use_label_encoder=False,
        #         eval_metric='logloss',
        #         random_state=42
        #     ),
        #     'params': {
        #         'n_estimators': np.arange(100, 1000, 10),
        #         'max_depth': np.arange(2, 10, 1),
        #         'learning_rate': np.arange(0.01, 0.3, 0.01),
        #         'subsample': [0.25, 0.5, 0.75, 1.0],
        #     },
        # },
        # {
        #     'name': 'KNN',
        #     'model': KNeighborsClassifier(),
        #     'params': {
        #         'n_neighbors': [3, 5, 7, 9, 10],
        #         'weights': ['uniform', 'distance'],
        #         'metric': ['euclidean', 'manhattan'],
        #     },
        # },
        {
            'name': 'MLP',
            'model': MLP(),
            'params': {
                'hidden_dims': [(32,), (64,), (32, 32), (64, 32)],
                'num_epochs': [50, 100, 200],
                'batch_size': [16, 32, 64],
                'lambda_l1': [0.0001, 0.001, 0.01],
                'lambda_l2': [0.0001, 0.001, 0.01],
            },
        },
    ]

    for analysis in [
        "class", "order", "family", "genus", "species",
        "t_sgb", "ko_gene", "uniref_family"
    ]:
        groups = ['CTR_CRC']
        source_study = [
            "CHN_HK-CRC", "AT-CRC", "IND-CRC", "CHN_SH-CRC",
            "FR-CRC", "DE-CRC", "ITA-CRC", "US-CRC",
            "JPN-CRC", "CHN_WF-CRC"
        ]

        save_dir = os.path.join(new_path, "Result/param1/")
        os.makedirs(save_dir, exist_ok=True)

        # Load and preprocess data
        meta_feature= split_data("species", "CTR_CRC", "Raw")
        meta_feature = change_group(meta_feature, "CTR_CRC")

        # Filter data based on study sources
        source_df = meta_feature[
            meta_feature['Study'].str.contains("|".join(source_study))
        ]

        y = source_df["Group"]
        X = source_df.drop(columns=["Group", "Study"])

        # Perform grid search for each model
        results = []
        for grid in param_grids:
            model_name = grid['name']
            model = grid['model']
            params = grid['params']

            cv_outer = RepeatedStratifiedKFold(n_splits=5)
            optimized_model = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring=auc_scorer,
                cv=cv_outer,
                verbose=1
            )
            optimized_model.fit(X, y)

            results.append(
                {
                    'study': "_".join(source_study),
                    'model': model_name,
                    'best_param': optimized_model.best_params_,
                }
            )

        result_df = pd.DataFrame(results)
        result_df.to_csv(
            os.path.join(save_dir, f"{analysis}_best_params.csv"),
            index=False
        )

        print(f"Analysis for {analysis} completed and results saved.")

def finally_select_param():
    """
    Perform parameter selection specifically for RandomForest and XGBoost models.
    Uses pre-selected features from a CSV file, then performs GridSearchCV for
    each taxonomic (and other) level. Best parameters are saved as CSV.
    """
    param_grids1 = [
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': np.arange(100, 1000, 10),
                'max_depth': np.arange(2, 10, 1),
            },
        },
        {
            'name': 'XGBoost',
            'model': XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ),
            'params': {
                'n_estimators': np.arange(100, 1000, 10),
                'max_depth': np.arange(2, 10, 1),
                'learning_rate': np.arange(0.01, 0.3, 0.01),
            },
        },
    ]

    for analysis in ["genus", "species","t_sgb"]:
        meta_feature = split_data(analysis, "CTR_CRC", "Raw_log")
        meta_feature = change_group(meta_feature, "CTR_CRC")

        source_study = [
            "CHN_HK-CRC", "AT-CRC", "IND-CRC", "CHN_SH-CRC",
            "FR-CRC", "DE-CRC", "ITA-CRC", "US-CRC",
            "JPN-CRC", "CHN_WF-CRC"
        ]
        source_df = meta_feature[
            meta_feature['Study'].str.contains("|".join(source_study))
        ]

        feature_path = (
            sys.path[1]
            + f'/Result/feature/{analysis}/Raw_log/CTR_CRC/feature.csv'
        )
        df = pd.read_csv(feature_path, index_col=0)
        feature = df[
            "FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_CHN_WF-CRC_"
            "CHN_SH-CRC_CHN_HK-CRC_DE-CRC_IND-CRC_US-CRC_rf_optimal"
        ]

        y = source_df["Group"]
        X = source_df[feature]

        results = []
        for grid in param_grids1:
            model_name = grid['name']
            model = grid['model']
            params = grid['params']

            cv_outer = RepeatedStratifiedKFold(n_splits=5)
            optimized_model = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring="roc_auc",
                cv=cv_outer,
                verbose=1
            )
            optimized_model.fit(X, y)

            results.append(
                {
                    'study': "_".join(source_study),
                    'model': model_name,
                    'best_param': optimized_model.best_params_,
                }
            )

        result_df = pd.DataFrame(results)
        save_dir = os.path.join(new_path, "Result/param/")
        os.makedirs(save_dir, exist_ok=True)
        result_df.to_csv(
            os.path.join(save_dir, f"{analysis}_best_params.csv"),
            index=False
        )

        print(f"Analysis for {analysis} completed and results saved.")

def main():
    """
    Main function to execute the parameter selection tasks.
    """
    print("Starting parameter selection...")
    select_param()
    print("Completed first parameter selection.")

    # print("Starting final parameter selection...")
    # finally_select_param()
    # print("Completed final parameter selection.")

if __name__ == "__main__":
    main()
