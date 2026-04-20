import csv


def convert_csv_to_ts(input_csv, output_ts):
    # Data structures
    nodes_by_id = {}

    # Root node
    root_word = "European Union"
    nodes_by_id["root"] = {
        "word": root_word,
        "prob": 1.0,
        "children": [],
        "endProb": 0.0,
        "rank": 0,
    }

    # Read CSV and build basic node objects
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["Node_ID"]
            token = row["Candidate_Token"].strip()
            prob = float(row["Probability"])
            rank = int(row["Rank"])

            nodes_by_id[node_id] = {
                "word": token,
                "prob": prob,
                "children": [],
                "endProb": 0.0,
                "rank": rank,
                "node_id": node_id,
            }

    # Second pass: Associate children with parents and handle endProb
    # Reload CSV to handle parent-child relationships in the correct order or just iterate through nodes_by_id
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["Node_ID"]
            token = nodes_by_id[node_id]["word"]
            prob = float(row["Probability"])

            # Determine parent ID
            if "." in node_id:
                parent_id = ".".join(node_id.split(".")[:-1])
            else:
                parent_id = "root"

            if parent_id in nodes_by_id:
                if token in ["<|eot_id|>", "."]:
                    nodes_by_id[parent_id]["endProb"] += prob
                else:
                    nodes_by_id[parent_id]["children"].append(nodes_by_id[node_id])

    # Sort children of each node by rank
    for node in nodes_by_id.values():
        node["children"].sort(key=lambda x: x["rank"])

    def format_node(node, indent=0):
        tab = "  " * (indent + 1)

        args = [f'"{node["word"]}"', f"{node['prob']:g}"]

        if node["children"] or node["endProb"] > 0:
            child_list_str = "[]"
            if node["children"]:
                child_lines = [format_node(c, indent + 1) for c in node["children"]]
                child_list_str = "[\n" + ",\n".join(child_lines) + "\n" + tab + "]"

            args.append(child_list_str)
            if node["endProb"] > 0:
                args.append(f"{node['endProb']:g}")

        # In current example, they use n(word, prob, [children], endProb)
        # If no children and no endProb, it's just n(word, prob)
        # If no children but has endProb, it's n(word, prob, [], endProb)

        # Clean up empty arrays if they are the last argument and no endProb
        if len(args) == 3 and args[2] == "[]":
            args.pop()

        return f"{tab}n({', '.join(args)})"

    # Build the final content
    ts_header = (
        """export interface PredictionNode {
  word: string;
  prob: number;
  children: PredictionNode[];
  endProb: number;
}

const n = (word: string, prob: number, children: PredictionNode[] = [], endProb = 0): PredictionNode =>
  ({ word, prob, children, endProb });

export const predictionTree: PredictionNode = n(\""""
        + root_word
        + """\", 1, [
"""
    )

    root_children = nodes_by_id["root"]["children"]
    branch_outputs = []
    for i, branch_node in enumerate(root_children):
        # branch_outputs.append(f"  // BRANCH {i+1}: {branch_node['word']} ({branch_node['prob']:g})")
        branch_content = []
        branch_content.append("  // " + "═" * 60)
        branch_content.append(
            f"  // BRANCH {i + 1}: {branch_node['word']} ({branch_node['prob']:g})"
        )
        branch_content.append("  // " + "═" * 60)
        branch_content.append(format_node(branch_node, 0))
        branch_outputs.append("\n".join(branch_content))

    ts_content = ts_header + ",\n".join(branch_outputs) + "\n]);\n"

    with open(output_ts, "w", encoding="utf-8") as f:
        f.write(ts_content)


if __name__ == "__main__":
    convert_csv_to_ts(
        "/Users/nwsowder/Documents/GitHub/llm_probability_distributions/4_wide_7_deep_clean_v5.csv",
        "/Users/nwsowder/Documents/GitHub/llm_probability_distributions/predictionTreeData.ts",
    )
    print("Success: Generated predictionTreeData.ts")
