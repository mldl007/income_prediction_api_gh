import pandas as pd


def update_var_list(var_list, x):
    """
    This function updates the variable list after each transform in the pipeline during
    fitting the final model after finding the best hyperparameters. This is similar to
    update_transform except it doesn't apply a transformer to data. It just updates
    the variable list during final model training.
    """
    var_list_series = pd.Series(var_list)
    return [*var_list_series[var_list_series.isin(x.columns)]].copy()
