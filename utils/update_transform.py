from utils.column_transformer import ColumnTransformer
import pandas as pd


def update_transform(transformer, x_train, x_val, columns,
                     cat_vars, num_vars, one_hot_vars, ordinal_vars,
                     y=None, update_cols=True):
    """
    - Function to fit and transform column transformers and also update which columns are removed.
    - Few transformers remove columns from a dataset. The removed columns
      must be updated so that further transformers in the pipeline don't consider them.
    - If we try to apply a transform to a deleted/removed column, this throws an error.
    """
    def update_var_list(var_list, x_train, update_cols=True):
        """
        Function that removes deleted columns by removing columns from variable list which are
        not present in the dataset after applying a transform.
        """
        if update_cols:
            var_list_series = pd.Series(var_list)
            return [*var_list_series[var_list_series.isin(x_train.columns)]].copy()
        else:
            return var_list.copy()

    x_train = x_train.copy()
    x_val = x_val.copy()
    transformer_ct = ColumnTransformer(transformer, columns=columns)
    if y is None:
        transformer_ct.fit(x_train)
    else:
        transformer_ct.fit(x_train, y)
    x_train_out = transformer_ct.transform(x_train).copy()
    x_val_out = transformer_ct.transform(x_val).copy()
    updated_cat_vars = update_var_list(cat_vars, x_train_out, update_cols)
    updated_num_vars = update_var_list(num_vars, x_train_out, update_cols)
    updated_ohe_vars = update_var_list(one_hot_vars, x_train_out, update_cols)
    updated_ord_vars = update_var_list(ordinal_vars, x_train_out, update_cols)
    return x_train_out, x_val_out, updated_cat_vars, updated_num_vars, updated_ohe_vars, updated_ord_vars
