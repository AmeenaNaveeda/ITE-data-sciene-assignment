import pandas as pd
import os
import re

BASE_PATH = os.path.join(os.path.dirname(__file__), '../../source')

def load_exhibitors():
    """
    load exhibitors.csv
    """
    return pd.read_csv(os.path.join(BASE_PATH, 'exhibitors.csv'))

def load_exhibitor_categories():
    """
    exhibitor_categories.csv
    """
    return pd.read_csv(os.path.join(BASE_PATH, 'exhibitor_categories.csv'))

def preprocess_exhibitors(exhibitors_df):
    """
    convert MainCategories from string separated by pipe to list
    """
    exhibitors_df['categoryId'] = exhibitors_df['MainCategories'].str.split('|')
    return exhibitors_df

def explode_exhibitors(exhibitors_df):
    """
    explode exhibitors based on category id
    """
    exhibitors_df['categoryId'] = exhibitors_df['MainCategories'].str.split('|')
    exploded_exhibitors_df = exhibitors_df.explode('categoryId')
    exploded_exhibitors_df['categoryId'] = exploded_exhibitors_df['categoryId'].astype(int)
    return exploded_exhibitors_df

def merged_exhibitors_categories(exploded_exhibitors_df, exhibitors_categories_df):
    """
    merge exhibitors and categories based on categoryId
    """
    return exploded_exhibitors_df.merge(exhibitors_categories_df, on='categoryId', how='left')

def preprocess_exhibitor_categories(exhibitor_categories_df: pd.DataFrame, column_name: str = "categoryName") -> pd.DataFrame:
    """
    - Removes leading numbers
    - Replaces '/' with 'or', '&' with 'and'
    - Converts to lowercase and strips whitespace
    """
    exhibitor_categories_df[f"{column_name}_cleaned"] = exhibitor_categories_df[column_name].apply(
        lambda text: (
            re.sub(r'\s+', ' ',
                re.sub(r'\s*/\s*', ' or ',
                re.sub(r'&', 'and',
                re.sub(r'^\d+(\.\d+)?\.?\s*', '', str(text)))
            )).strip().lower()
        ) if isinstance(text, str) else ""
    )
    return exhibitor_categories_df