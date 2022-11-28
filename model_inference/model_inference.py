from utils.load_model import load_model
from feature_construction.feature_construction import FeatureConstructor
import pandas as pd
from logger.logger import MongoLogger


def predict(x: pd.DataFrame, predict_proba: bool = True,
            use_deployed_model: bool = True, model_file_name: str = None, predict_label: bool = False):
    prediction = None
    logger = MongoLogger()
    try:
        logger.log_to_db(level="INFO", message="entering model_inference.predict")

        loaded_model = load_model(load_deployed_model=use_deployed_model, model_file_name=model_file_name)

        if loaded_model is not None:
            (label_encoder, cat_var_id_transform, cat_var_threshold_transform,
             num_var_threshold_transform, rare_cat_transform, outlier_transform,
             cat_missing_imputer, num_missing_imputer, one_hot_encoder,
             ordinal_encoder, minmax_scaler, clusterer, multicoll_transform,
             feature_selector, class_weight_flag, smote_transform, classifier, best_class_threshold) = loaded_model

            feature_constructor = FeatureConstructor()
            x = feature_constructor.add_features(x).copy()

            if cat_var_id_transform is not None:
                x = cat_var_id_transform.transform(x).copy()

            if cat_var_threshold_transform is not None:
                x = cat_var_threshold_transform.transform(x).copy()

            if num_var_threshold_transform is not None:
                x = num_var_threshold_transform.transform(x).copy()

            if rare_cat_transform is not None:
                x = rare_cat_transform.transform(x).copy()

            if outlier_transform is not None:
                x = outlier_transform.transform(x).copy()

            if cat_missing_imputer is not None:
                x = cat_missing_imputer.transform(x).copy()

            if num_missing_imputer is not None:
                x = num_missing_imputer.transform(x).copy()

            if one_hot_encoder is not None:
                x = one_hot_encoder.transform(x).copy()

            if ordinal_encoder is not None:
                x = ordinal_encoder.transform(x).copy()

            if minmax_scaler is not None:
                x = minmax_scaler.transform(x).copy()

            if clusterer is not None:
                cluster_ohe, _ = clusterer.predict(x)
                x = pd.concat([x, cluster_ohe], axis=1)

            if multicoll_transform is not None:
                x = multicoll_transform.transform(x).copy()

            if feature_selector is not None:
                x = feature_selector.transform(x).copy()

            if predict_proba:
                prediction = classifier.predict_proba(x)[:, 1]
            else:
                prediction = (classifier.predict_proba(x)[:, 1] >= best_class_threshold).astype('int')
                if predict_label:
                    prediction = label_encoder.inverse_transform(prediction)

    except Exception as e:
        logger.log_to_db(level="CRITICAL", message=f"unexpected model_inference.predict error: {e}")
        raise
    logger.log_to_db(level="INFO", message="exiting model_inference.predict")

    return prediction
