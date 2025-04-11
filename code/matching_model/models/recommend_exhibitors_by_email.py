import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
from utils.visitors import (
    load_visitors,
    merge_answers_questions,
    preprocess_visitors,
    filter_valid_visitor_answers
    
)
from utils.exhibitors import (
    load_exhibitors
)

def recommend_exhibitors_for_visitor_email(email, top_k=7, penalize=True):
    """
    - return top_k exhibitors for an answer_text
    """
    answer_exhibitor_matching_df = pd.read_csv("../results/answer_to_exhibitor_mapping.csv")
    
    visitors_df = load_visitors()
    exhibitors_df = load_exhibitors()

    # preprocess data
    exhibitors_df["category_list"] = exhibitors_df["MainCategories"].astype(str).apply(lambda x: x.split("|"))
    exhibitor_category_count = exhibitors_df.set_index("exhibitorid")["category_list"].apply(len).to_dict()
    matched_categories_set = set(answer_exhibitor_matching_df["matched_category_ids"]
                                .dropna()
                                .str.split("|")
                                .explode())
    total_categories = len(matched_categories_set)
    visitor_expanded = preprocess_visitors(visitors_df)
    merged_answers_questions_df = merge_answers_questions()
    valid_visitor_answers_df = filter_valid_visitor_answers(visitor_expanded, merged_answers_questions_df)
    
    # get answer_ids based on email
    answer_ids = valid_visitor_answers_df[valid_visitor_answers_df["email"] == email]["answerId"].dropna().astype(str).unique().tolist()
    
    if not answer_ids:
        print("No answers found for this email.")
        return pd.DataFrame()

    # Filter the mapping dataframe for matching answers
    df = answer_exhibitor_matching_df[answer_exhibitor_matching_df["answer_id"].isin(answer_ids)]

    if df.empty:
        print("No matching exhibitor categories found.")
        return pd.DataFrame()

    # Aggregate scores per exhibitor
    grouped = df.groupby(["exhibitor_id", "exhibitor_name"]).agg({
        "similarity_score_sum": "sum",
        "matched_category_count": "sum"
    }).reset_index()

    # Add penalty for exhibitors who select too many categories
    if penalize:
        grouped["category_count"] = grouped["exhibitor_id"].map(exhibitor_category_count)
        grouped["penalty_factor"] = 1 - (grouped["category_count"] / total_categories)
        grouped["final_score"] = grouped["similarity_score_sum"] * grouped["penalty_factor"]
    else:
        grouped["final_score"] = grouped["similarity_score_sum"]

    # Return top-k ranked exhibitors with core details
    return grouped.sort_values(by="final_score", ascending=False)[
        ["exhibitor_id", "exhibitor_name", "similarity_score_sum", "matched_category_count", "final_score"]
    ].head(top_k)