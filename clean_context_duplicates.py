import pandas as pd
import os


def clean_context_duplicates(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    initial_row_count = len(df)

    # Ensure string types and handle NaNs for calculation
    df["Node_ID"] = df["Node_ID"].astype(str)
    # Using a temporary copy for comparison to avoid modifying original whitespace if it matters
    # but the user said "duplicate string", so we'll compare stripped versions to be safe
    # or should we compare exactly? Let's compare exactly first.

    # Map Node_ID to Branch_Context
    context_map = dict(zip(df["Node_ID"], df["Branch_Context"].fillna("").astype(str)))

    nodes_to_remove = set()

    for index, row in df.iterrows():
        node_id = row["Node_ID"]
        if "." in node_id:
            parent_id = ".".join(node_id.split(".")[:-1])
            if parent_id in context_map:
                parent_context = context_map[parent_id]
                child_context = (
                    str(row["Branch_Context"])
                    if pd.notna(row["Branch_Context"])
                    else ""
                )

                if child_context == parent_context:
                    print(
                        f"Found match: Node {node_id} has same context as Parent {parent_id}"
                    )
                    nodes_to_remove.add(node_id)

    if not nodes_to_remove:
        print("No parent-child context duplicates found.")
        return False

    # Also need to remove all descendants of flagged nodes
    # A descendant of 1.1.2 is any node starting with 1.1.2.
    all_to_remove = set()
    for node_id in nodes_to_remove:
        all_to_remove.add(node_id)
        # Find all nodes that start with this ID followed by a dot
        prefix = node_id + "."
        for potential_descendant in df["Node_ID"]:
            if potential_descendant.startswith(prefix):
                all_to_remove.add(potential_descendant)

    print(f"Removing {len(all_to_remove)} nodes (including descendants)...")
    df_cleaned = df[~df["Node_ID"].isin(all_to_remove)]

    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned file saved to {output_file}")
    print(f"Total rows removed: {initial_row_count - len(df_cleaned)}")
    return True


if __name__ == "__main__":
    input_csv = "/Users/nwsowder/Documents/GitHub/llm_probability_distributions/4_wide_7_deep_clean.csv"
    output_csv = "/Users/nwsowder/Documents/GitHub/llm_probability_distributions/4_wide_7_deep_clean.csv"  # Overwrite as requested or suggested by context

    clean_context_duplicates(input_csv, output_csv)
