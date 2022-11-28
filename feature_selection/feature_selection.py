import pandas as pd
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from utils.find_class_weights import find_class_weights
from sklearn.utils import class_weight
from logger.logger import MongoLogger


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    - Feature selector selects n_features by training RandomForestClassifier and XGBClassifier.
    - It uses hyperopt hyperparameter tuning to choose the best model from the two.
    - It uses feature importance of the best model to select features.
    """
    def __init__(self, n_features: int, n_trials: int = 10, cv_splits: int = 2):
        self.n_features = int(n_features)
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.selected_features = []
        self.feature_importance = None

    def __find_best_model(self, x: pd.DataFrame, y: pd.Series):
        x = x.reset_index(drop=True).copy()
        y = y.reset_index(drop=True).copy()
        classifier_params = ['random_forest', 'xgboost']
        max_depth_params = [3, 4, 5, 6, 7, 8, 9, 10]
        n_estimators_params = [50, 100, 200, 300, 500]
        eta_params = [0.1, 0.3, 0.01, 0.001, 0.0001, 1]
        search_space = {'classifier': hp.choice('classifier', [
            {
                'type': classifier_params[0],
                'max_depth': hp.choice('rf_max_depth', max_depth_params),
                'n_estimators': hp.choice('rf_n_estimators', n_estimators_params)
            },
            {
                'type': classifier_params[1],
                'max_depth': hp.choice('xgb_max_depth', max_depth_params),
                'n_estimators': hp.choice('xgb_n_estimators', n_estimators_params),
                'eta': hp.choice('xgb_eta', eta_params)
            }
        ])}

        def objective(params):
            classifier = None
            max_depth = params['classifier']['max_depth']
            n_estimators = params['classifier']['n_estimators']
            train_scores = []
            val_scores = []
            skfold = StratifiedKFold(n_splits=self.cv_splits)
            for train_idx, test_idx in skfold.split(x, y):
                if params['classifier']['type'] == classifier_params[0]:  # rf
                    class_weight_params = find_class_weights(y=y[train_idx])
                    classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42,
                                                        class_weight=class_weight_params)
                    classifier.fit(x.iloc[train_idx], y[train_idx])

                if params['classifier']['type'] == classifier_params[1]:  # xgboost
                    eta = params['classifier']['eta']
                    sample_weight_params = class_weight.compute_sample_weight(class_weight="balanced", y=y[train_idx])
                    classifier = XGBClassifier(n_estimators=n_estimators, eta=eta,
                                               max_depth=max_depth, random_state=42, verbosity=0)
                    classifier.fit(x.iloc[train_idx], y[train_idx], sample_weight=sample_weight_params)

                train_scores.append(roc_auc_score(y[train_idx],
                                                  classifier.predict_proba(x.iloc[train_idx])[:, 1]))
                val_scores.append(roc_auc_score(y[test_idx],
                                                classifier.predict_proba(x.iloc[test_idx])[:, 1]))

            avg_train_score = np.mean(train_scores)
            avg_val_score = np.mean(val_scores)
            return {'loss': -avg_val_score, 'train_score': avg_train_score, 'val_score': avg_val_score,
                    'status': STATUS_OK}

        model_trials = Trials()
        model_best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            catch_eval_exceptions=False,
            verbose=False,
            trials=model_trials
        )

        best_classifier_name = classifier_params[model_best['classifier']]
        best_classifier_model = None
        if best_classifier_name == classifier_params[0]:  # rf
            class_weight_params = find_class_weights(y=y)
            best_max_depth = max_depth_params[model_best['rf_max_depth']]
            best_n_estimators = n_estimators_params[model_best['rf_n_estimators']]
            best_classifier_model = RandomForestClassifier(max_depth=best_max_depth,
                                                           n_estimators=best_n_estimators,
                                                           random_state=42,
                                                           class_weight=class_weight_params)

        if best_classifier_name == classifier_params[1]:  # xgboost
            best_n_estimators = n_estimators_params[model_best['xgb_n_estimators']]
            best_max_depth = max_depth_params[model_best['xgb_max_depth']]
            best_eta = eta_params[model_best['xgb_eta']]
            best_classifier_model = XGBClassifier(n_estimators=best_n_estimators, eta=best_eta,
                                                  max_depth=best_max_depth, random_state=42, verbosity=0)

        return best_classifier_model, model_trials.best_trial

    def fit(self, x: pd.DataFrame, y: pd.Series):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering feature_selection.fit")
            x = x.reset_index(drop=True).copy()
            y = y.reset_index(drop=True).copy()
            best_classifier, _ = self.__find_best_model(x, y)
            if isinstance(best_classifier, XGBClassifier):
                sample_weight_params = class_weight.compute_sample_weight(class_weight="balanced", y=y)
                _ = best_classifier.fit(x, y, sample_weight=sample_weight_params)
            else:
                _ = best_classifier.fit(x, y)
            self.feature_importance = pd.Series(best_classifier.feature_importances_,
                                                index=x.columns).sort_values(ascending=False)
            self.selected_features = [*self.feature_importance[: self.n_features].index]
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected feature_selection.fit error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting feature_selection.fit")
        return self

    def transform(self, x: pd.DataFrame, y=None):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering feature_selection.transform")
            x = x.reset_index(drop=True).copy()
        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected feature_selection.transform error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting feature_selection.transform")
        return x[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features
