import pandas as pd
from logger.logger import MongoLogger


class FeatureConstructor:
    """
    Adds new engineered features to input data
    """
    @staticmethod
    def add_features(x: pd.DataFrame):
        logger = MongoLogger()
        try:
            logger.log_to_db(level="INFO", message="entering add_features")
            x = x.copy()

            education_group = {' 5th-6th': 'middle_school',
                               ' 7th-8th': 'middle_school',
                               ' 9th': 'middle_school',
                               ' 10th': 'high_school',
                               ' 11th': 'high_school',
                               ' 12th': 'high_school',
                               ' HS-grad': 'hs_grad',
                               ' Prof-school': 'high_school',
                               ' Some-college': 'college',
                               ' Masters': 'college',
                               ' Bachelors': 'college',
                               ' 1st-4th': 'primary_school',
                               ' Preschool': 'primary_school',
                               ' Assoc-voc': 'college',
                               ' Assoc-acdm': 'college',
                               ' Doctorate': 'doctorate'}
            x['education_group'] = x['education'].map(education_group)

            # workclass group has ver low mutual info score so removing it

            # workclass_group = {' Federal-gov': 'government',
            #                    ' Local-gov': 'government',
            #                    ' State-gov': 'government',
            #                    ' Private': 'private',
            #                    ' Self-emp-inc': 'self_emp',
            #                    ' Self-emp-not-inc': 'self_emp',
            #                    ' Never-worked': 'no_work',
            #                    ' Without-pay': 'no_work'}
            # x['workclass_group'] = x['workclass'].map(workclass_group)

            is_single = {' Divorced': 1,
                               ' Married-spouse-absent': 1,
                               ' Never-married': 1,
                               ' Separated': 1,
                               ' Widowed': 1,
                               ' Married-AF-spouse': 0,
                               ' Married-civ-spouse': 0}
            x['is_single'] = x['marital_status'].map(is_single)

            x['relationship_marital'] = x['relationship'] + x['marital_status']

            # x.loc[(x['capital_gain'] > 5000), 'capital_gain'] = 5000

            x['has_capital_gain'] = (x['capital_gain'] > 0).astype("int")

            # x['has_capital_loss'] = (x['capital_loss'] > 0).astype("int")

            x['age_of_first_edu'] = x['age'] - x['education_num']

        except Exception as e:
            logger.log_to_db(level="CRITICAL", message=f"unexpected add_features error: {e}")
            raise
        logger.log_to_db(level="INFO", message="exiting add_features")

        return x
