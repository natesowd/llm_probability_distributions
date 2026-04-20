import pandas as pd


def find_low_prob_sibling_groups(file_path, threshold=0.5):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)

    # Ensure Node_ID is string
    df["Node_ID"] = df["Node_ID"].astype(str)

    # Extract Parent_ID for each node
    def get_parent_id(node_id):
        if "." not in node_id:
            return "ROOT"  # Or some indicator for top level nodes
        return ".".join(node_id.split(".")[:-1])

    df["Parent_ID"] = df["Node_ID"].apply(get_parent_id)

    # Group by Parent_ID and sum probabilities
    sibling_sums = df.groupby("Parent_ID")["Probability"].sum().reset_index()
    sibling_sums.columns = ["Parent_ID", "Total_Probability"]

    # Filter groups where sum < threshold
    low_prob_groups = sibling_sums[sibling_sums["Total_Probability"] < threshold]

    # Exclude ROOT if it's not a real parent in this context (optional, depending on user intent)
    # But usually, root siblings (1, 2, 3...) also form a distribution.

    # Get parent details for flagged groups
    # We want to flag the parent node itself.
    # Note: Parent_ID might not exist as a row if it's the absolute root or header,
    # but in this file, 1.1.1 is the parent of 1.1.1.1 etc.

    parents_mapping = df.set_index("Node_ID")[
        ["Branch_Context", "Candidate_Token"]
    ].to_dict("index")

    print(f"\nSibling groups with total probability < {threshold}:")
    print("-" * 60)

    flagged_count = 0
    for _, row in low_prob_groups.iterrows():
        p_id = row["Parent_ID"]
        total_p = row["Total_Probability"]

        parent_info = parents_mapping.get(
            p_id, {"Branch_Context": "N/A", "Candidate_Token": "N/A"}
        )

        print(f"Parent ID: {p_id}")
        print(f"  Total Sibling Prob: {total_p:.4f}")
        print(f"  Parent Context: {parent_info['Branch_Context']}")
        print(f"  Parent Token: {parent_info['Candidate_Token']}")
        print("-" * 30)
        flagged_count += 1

    print(f"\nTotal distributions flagged: {flagged_count}")
    return low_prob_groups


if __name__ == "__main__":
    # csv_path = "/Users/nwsowder/Documents/GitHub/llm_probability_distributions/4_wide_7_deep_clean.csv"
    csv_path = "/Users/nwsowder/Documents/GitHub/llm_probability_distributions/4_wide_7_deep_clean_v2.csv"
    find_low_prob_sibling_groups(csv_path)
