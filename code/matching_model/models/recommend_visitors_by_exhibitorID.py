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

def recommend_visitors_for_exhibitor(exhibitor_id, top_k=7):
    """
    return top_k visitors for a given exhibitor id
    """

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    RESULTS_PATH = os.path.join(BASE_DIR, "results")
    answer_exhibitor_matching_df = pd.read_csv(os.path.join(RESULTS_PATH, "answer_to_exhibitor_mapping.csv"))
    visitors_df = load_visitors()
    answer_exhibitor_matching_df["answer_id"] = answer_exhibitor_matching_df["answer_id"].astype(str)

    visitor_expanded = preprocess_visitors(visitors_df)
    merged_answers_questions_df = merge_answers_questions()
    valid_visitor_answers_df = filter_valid_visitor_answers(visitor_expanded, merged_answers_questions_df)
    valid_visitor_answers_df["answerId"] = valid_visitor_answers_df["answerId"].astype(str)

    visitor_lookup_df = valid_visitor_answers_df[["answerId", "id", "email"]].rename(columns={
        "answerId": "answer_id", "id": "visitor_id"
    }).dropna()

    # Merge answer_exhibitor_matching_df and visitor info
    map_exhibitor_visitor_df = answer_exhibitor_matching_df.merge(visitor_lookup_df, on="answer_id", how="left")
    map_exhibitor_visitor_df["exhibitor_id"] = map_exhibitor_visitor_df["exhibitor_id"].astype(str)
    exhibitor_id = str(exhibitor_id)

    # Filter only rows matching given exhibitor
    filtered = map_exhibitor_visitor_df[map_exhibitor_visitor_df["exhibitor_id"] == exhibitor_id]

    if filtered.empty:
        print(f"No visitor matches found for exhibitor ID: {exhibitor_id}")
        return pd.DataFrame()

    # Aggregate scores by visitor
    grouped = filtered.groupby(["visitor_id", "email"]).agg({
        "similarity_score_sum": "sum",
        "matched_category_count": "sum"
    }).reset_index()

    grouped["final_score"] = grouped["similarity_score_sum"]

    # Return top-k visitors
    return grouped.sort_values(by="final_score", ascending=False).head(top_k)
