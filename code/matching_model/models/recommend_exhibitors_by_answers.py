import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
import pandas as pd
from utils.exhibitors import (
    load_exhibitors
)

def recommend_exhibitors_for_answer(answer_text, penalize=True, top_k=7):
    """
    - return top_k exhibitors for an answer_text
    """
    answer_exhibitor_matching_df = pd.read_csv("../results/answer_to_exhibitor_mapping.csv")
    exhibitors_df = load_exhibitors()

    exhibitors_df["category_list"] = exhibitors_df["MainCategories"].astype(str).apply(lambda x: x.split("|"))
    exhibitor_category_count = exhibitors_df.set_index("exhibitorid")["category_list"].apply(len).to_dict()

    # Compute total unique categories
    all_cats = answer_exhibitor_matching_df["matched_category_ids"].dropna().str.split("|").explode().unique()
    total_categories = len(all_cats)

    answer_text = answer_text.strip().lower()
    filtered = answer_exhibitor_matching_df[answer_exhibitor_matching_df["answer_text"].str.strip().str.lower() == answer_text]

    if filtered.empty:
        print(f"No matches found for: '{answer_text}'")
        return pd.DataFrame()

    grouped = filtered.groupby(["exhibitor_id", "exhibitor_name"]).agg({
        "similarity_score_sum": "sum",
        "matched_category_count": "sum",
        "matched_category_names": lambda x: "|".join(set(cat for cats in x for cat in cats.split("|")))
    }).reset_index()

    if penalize:
        grouped["category_count"] = grouped["exhibitor_id"].map(exhibitor_category_count)
        grouped["penalty_factor"] = 1 - (grouped["category_count"] / total_categories)
        grouped["final_score"] = grouped["similarity_score_sum"] * grouped["penalty_factor"]
    else:
        grouped["final_score"] = grouped["similarity_score_sum"]

    return grouped.sort_values(by="final_score", ascending=False).head(top_k)
