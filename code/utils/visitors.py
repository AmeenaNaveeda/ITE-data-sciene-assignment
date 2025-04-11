import pandas as pd
import json
import re
import os

BASE_PATH = os.path.join(os.path.dirname(__file__), '../../source')

def load_visitors():
    """
    Load visitors.csv
    """
    return pd.read_csv(os.path.join(BASE_PATH, 'visitors.csv'))

def load_visitors_answers():
    """
    Load visitors_answers.csv
    """
    return pd.read_csv(os.path.join(BASE_PATH, 'visitors_answers.csv'))

def load_visitors_questions():
    """
    Load visitors_questions.csv
    """
    return pd.read_csv(os.path.join(BASE_PATH, 'visitors_questions.csv'))

def preprocess_visitor_answers(df: pd.DataFrame, column_name: str = "answer") -> pd.DataFrame:
    """
    Cleans a category column in a DataFrame:
    - Replaces '/' with 'or', '&' with 'and'
    - Converts to lowercase and strips whitespace
    """
    df[f"{column_name}_cleaned"] = df[column_name].apply(
        lambda text: (
            re.sub(r'\s+', ' ',
                re.sub(r'\s*/\s*', ' or ',
                re.sub(r'&', 'and',
                str(text)))
            ).strip().lower()
        ) if isinstance(text, str) else ""
    )
    return df

def merge_answers_questions():
    """
    - merge visitor_answers and questions
    - remove duplicate column - i.e. questionid
    """
    answers_df = load_visitors_answers()
    questions_df = load_visitors_questions()
    merged_df = answers_df.merge(
        questions_df,
        left_on="questionId",
        right_on="id",
        suffixes=("_answer", "_question")
    )
    merged_df = merged_df[["id_question", "questionTypeId", "stepId", "id_answer", "question", "answer",]]
    return merged_df

def preprocess_visitors(visitors_df):
    """
    - convert data column into json from json string
    - explode visitors with respect to their conversation
    - flatten visitors rows per question and answer
    """
    visitors_df['parsed_data'] = visitors_df['data'].apply(json.loads)

    exploded_df = visitors_df.explode('parsed_data')

    expanded_answers = pd.json_normalize(exploded_df['parsed_data'])

    visitor_expanded_answer_df = pd.concat([
        exploded_df[['id', 'email', 'gender']].reset_index(drop=True),
        expanded_answers
    ], axis=1)
    return visitor_expanded_answer_df

def filter_valid_visitor_answers(visitor_expanded_answer_df, merged_answers_questions_df):
    """
    - join visitor answer with question
    - drop entries without valid answerid
    """
    visitor_merge_answer_df = visitor_expanded_answer_df.merge(
        merged_answers_questions_df[['id_answer', 'answer', 'id_question', 'question']],
        left_on='answerId',
        right_on='id_answer',
        how='left',
        suffixes=('', '_merged_answer')
    )
    valid_visitor_answer_df = visitor_merge_answer_df.dropna(subset=['id_answer'])
    return valid_visitor_answer_df

