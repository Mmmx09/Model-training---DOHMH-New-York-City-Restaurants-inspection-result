from typing import Callable
import copy
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype


def add_date_columns(df: pd.DataFrame, fields: list[str], drop: bool = True) -> pd.DataFrame:
    """
    Convert a column of df from a datetime64 to many columns containing
    the information from the date.
    """
    df = copy.deepcopy(df)
        
    for field in fields:
        fld = df[field]
        fld_dtype = fld.dtype
        
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[field] = fld = pd.to_datetime(fld, infer_datetime_format = True)
        targ_pre = re.sub('[Dd]ate$', '', field)
        attr = [
            'Year', 'Month', 'Day', 
            'Dayofweek', 'Dayofyear',
            'Is_month_start', 'Is_month_end',
            'Is_quarter_start', 'Is_quarter_end',
            'Is_year_start', 'Is_year_end',
        ]
        for n in attr: 
            df[targ_pre + n] = getattr(fld.dt, n.lower())
            
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        
    if drop: 
        df.drop(columns = fields, axis = 1, inplace = True)
    return df

    
def is_date(x: pd.Series) -> bool: 
    """
    Assert whether a pandas Series is of dtype np.datetime64.
    """
    return np.issubdtype(x.dtype, np.datetime64)


def change_columns_from_str_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change any columns of strings in a panda's dataframe to a column of
    categorical values.
    """
    df = copy.deepcopy(df)
    
    for n, c in df.items():
        if is_string_dtype(c): 
            df[n] = c.astype('category').cat.as_ordered()
    return df

    
def apply_cats(df, trn):
    """
    Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.
    """
    for n, c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name == 'category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(trn[n].cat.categories, ordered = True, inplace = True)
    return


def fix_missing(df, col, name, na_dict):
    """
    Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

    
def numericalize(df: pd.DataFrame, col: str, name: str, max_n_cat: int | None) -> pd.DataFrame:
    """
    Changes the column col from a categorical type to it's integer codes.
    """
    df = copy.deepcopy(df)
    if (not is_numeric_dtype(col) 
        and (max_n_cat is None or len(col.cat.categories) > max_n_cat)):
        df[name] = pd.Categorical(col).codes + 1
    return df


def process_df(
    df: pd.DataFrame, 
    y_field: str | None = None, 
    skip_flds: list = [],
    ignore_flds: list = [], 
    na_dict: dict = {},
    preproc_fn: Callable = None, 
    max_n_cat = None, 
    ):
    """
    Take a dataframe df and splits off the response variable, and
    changes the df into an entirely numeric dataframe. For each column of df 
    which is not in skip_flds nor in ignore_flds, na values are replaced by the
    median value of the column.
    """
    df = copy.deepcopy(df)
    
    df_ignored = df.loc[:, ignore_flds]
    df = df.drop(columns = ignore_flds)
    
    if preproc_fn: 
        preproc_fn(df)
        
    if y_field is None: 
        y = None
    else:
        if not is_numeric_dtype(df[y_field]): 
            df[y_field] = pd.Categorical(df[y_field]).codes
        y = df[y_field].values
        skip_flds += [y_field]
    
    df = df.drop(columns = skip_flds)

    na_dict_initial = na_dict.copy()
    for n, c in df.items(): 
        na_dict = fix_missing(df, c, n, na_dict)
    
    if len(na_dict_initial) > 0:
        df = df.drop(
            [a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], 
            axis = 1,
        )
    for n,c in df.items(): 
        df = numericalize(df, c, n, max_n_cat)
        
    df = pd.get_dummies(df, dummy_na = True)
    df = pd.concat([df_ignored, df], axis = 1)
    return (df, y, na_dict)


def split_vals(df: pd.DataFrame, n: int) -> pd.DataFrame: 
    return df[:n].copy(), df[n:].copy()