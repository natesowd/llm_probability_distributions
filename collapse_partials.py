"""
collapse_partials.py

Post-processes a token probability tree CSV to collapse partial-word tokens
into whole-word tokens.

A token is considered "partial" if its children in the tree do NOT start with
a space (meaning they are sub-word continuations, e.g. "Cons" -> "ensus").

For each partial token, this script:
  1. Greedily selects the rank-1 (most likely) continuation token.
  2. Concatenates the partial + continuation into a single token
     (e.g. " Cons" + "ensus" = " Consensus").
  3. Chains the probabilities: P(merged) = P(partial) * P(continuation).
     LogProbs are summed accordingly.
  4. Drops all other continuation siblings (rank 2, 3, 4) and recursively
     removes all of their descendants.
  5. The merged node adopts the children of the completed word node.
  6. This is applied recursively — if the merged result is still partial,
     it continues collapsing deeper.

Usage:
    python collapse_partials.py 4_wide_7_deep.csv -o 4_wide_7_deep_words.csv

The output CSV has the same columns as the input, with updated Node_IDs,
Depths, and recomputed cumulative probabilities.
"""

import argparse
import csv
import io
import math
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Parse CSV into tree structure
# ---------------------------------------------------------------------------


def parse_csv(filepath):
    """Read the CSV and return a list of row dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def build_tree(rows):
    """
    Build a dict mapping node_id -> row_dict, and a children mapping
    parent_id -> [child_row_dicts] (ordered by rank).
    """
    nodes = {}
    children = defaultdict(list)

    for row in rows:
        nid = row["Node_ID"]
        nodes[nid] = row
        # Parent is everything up to the last dot
        parts = nid.rsplit(".", 1)
        if len(parts) == 2:
            parent_id = parts[0]
            children[parent_id].append(row)

    return nodes, children


# ---------------------------------------------------------------------------
# Detect whether a node is partial
# ---------------------------------------------------------------------------


def is_partial_word_node(node_id, children_map):
    """
    A node is a partial-word node if ALL of its children's Candidate_Token
    values do NOT start with a space (meaning they are sub-word continuations).

    Also returns False if the node has no children (leaf nodes are not partial).
    Special tokens like <|eot_id|>, ".", etc. are excluded from the
    continuation check.
    """
    kids = children_map.get(node_id, [])
    if not kids:
        return False

    # Check the rank-1 child: if it doesn't start with a space and doesn't
    # look like a special token or punctuation, this is a partial word.
    rank1 = None
    for kid in kids:
        if int(kid["Rank"]) == 1:
            rank1 = kid
            break

    if rank1 is None:
        return False

    token = rank1["Candidate_Token"]

    # Special tokens and punctuation are not continuations
    if token.startswith("<|") or token in (
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        '"',
        "'",
        '."',
    ):
        return False

    # If the rank-1 continuation does NOT start with a space, it's a sub-word
    # continuation (partial word)
    if not token.startswith(" "):
        return True

    return False


# ---------------------------------------------------------------------------
# Collapse partial word nodes
# ---------------------------------------------------------------------------


def collapse_tree(nodes, children_map):
    """
    Walk the tree top-down. When a partial-word node is found:
    1. Find the rank-1 child (greedy continuation).
    2. Merge the continuation token into the parent's Candidate_Token.
    3. Chain probabilities.
    4. Drop all sibling continuations (and their descendants).
    5. Adopt the children of the merged continuation node.
    6. Repeat recursively if the result is still partial.

    Returns a new list of output rows.
    """
    # Find root nodes (depth 1)
    root_ids = [nid for nid, node in nodes.items() if int(node["Depth"]) == 1]
    root_ids.sort(key=lambda x: [int(p) for p in x.split(".")])

    output_rows = []

    def process_node(node_id):
        """Process a node, collapsing partials, and return the (possibly
        modified) node plus recursively processed children."""
        node = dict(nodes[node_id])  # shallow copy

        # Keep collapsing while this node is partial
        current_id = node_id
        while is_partial_word_node(current_id, children_map):
            kids = children_map[current_id]

            # Find rank-1 child
            rank1_child = None
            for kid in kids:
                if int(kid["Rank"]) == 1:
                    rank1_child = kid
                    break

            if rank1_child is None:
                break

            rank1_child_id = rank1_child["Node_ID"]

            # Merge token
            continuation_token = rank1_child["Candidate_Token"]
            node["Candidate_Token"] = node["Candidate_Token"] + continuation_token

            # Chain probability: P(merged) = P(parent) * P(continuation)
            parent_prob = float(node["Probability"])
            child_prob = float(rank1_child["Probability"])
            merged_prob = parent_prob * child_prob

            parent_logprob = float(node["LogProb"])
            child_logprob = float(rank1_child["LogProb"])
            merged_logprob = parent_logprob + child_logprob

            node["Probability"] = merged_prob
            node["LogProb"] = merged_logprob

            # NOTE: Branch_Context stays the same — it represents the
            # context BEFORE this token was generated. The token itself
            # just got longer via concatenation.

            # Update cumulative values from the rank-1 child
            node["Cumulative_LogProb"] = float(rank1_child["Cumulative_LogProb"])
            node["Cumulative_Probability"] = float(
                rank1_child["Cumulative_Probability"]
            )

            # Remove the intermediate depth: the merged node stays at the
            # original depth, but now adopts the rank-1 child's children.
            # Drop ALL siblings of the rank-1 child (ranks 2, 3, 4) and
            # their descendants — they are implicitly removed by not being
            # visited.

            # Replace children_map for current_id with the rank-1 child's
            # children.
            children_map[current_id] = children_map.get(rank1_child_id, [])

            # NOTE: We don't change current_id — we keep checking if the
            # merged node is STILL partial (recursive collapsing).

        # Add this (possibly merged) node to output
        output_rows.append(node)

        # Now process children of the (possibly merged) node
        kids = children_map.get(current_id, [])
        for kid in kids:
            process_node(kid["Node_ID"])

    for root_id in root_ids:
        process_node(root_id)

    return output_rows


# ---------------------------------------------------------------------------
# Reassign Node_IDs, Depths, and recompute cumulative probabilities
# ---------------------------------------------------------------------------


def reassign_ids_and_depths(rows):
    """
    After collapsing, the Node_IDs and Depths may be inconsistent.
    Reassign them based on tree structure (parent-child relationships
    from the original traversal order).

    We rebuild the tree by tracking the "path" — each node at depth d
    gets a sequential branch number under its parent.
    """
    if not rows:
        return rows

    # We need to reconstruct the tree from the flat list.
    # The rows are in DFS order from the collapse step.
    # We'll assign new IDs based on original depth relationships.

    # First, figure out the original depth for each row.
    # After collapsing, we need to recompute depths from scratch.
    # Use the Branch_Context to determine depth: count words after
    # the initial prompt.

    # Actually, the simplest approach: rebuild the tree from the DFS order.
    # The original Depth field tells us the nesting. But after collapsing,
    # a node that was at depth 5 might now logically be at depth 4 if its
    # parent was collapsed.

    # Let's use a stack-based approach:
    # - Track the current path (stack of node references)
    # - When we encounter a node whose original depth is > current depth,
    #   it's a child
    # - When depth is <= current depth, pop back up

    # However, since we collapsed intermediate nodes, the original Depth
    # field is unreliable. Instead, we use the DFS output order and the
    # parent-child relationship we maintained.

    # Simplest approach: use the existing Node_IDs from the original tree
    # and just renumber them sequentially.

    # Actually, let's take a simpler approach: rebuild by tracking which
    # node is the parent of which. During collapse, we output nodes in
    # DFS order. We can use the Branch_Context to determine parentage:
    # a node B is a child of node A if B's Branch_Context starts with
    # A's Branch_Context + A's Candidate_Token.

    # Even simpler: just re-derive depth from the original depth field
    # minus the number of collapsed levels.

    # Let me think about this differently. After collapse, the flat list
    # is in DFS order. Let me just reassign depths and IDs properly by
    # tracking the context prefix.

    # The prompt prefix (before any generated tokens)
    first_row = rows[0]
    # The Branch_Context of depth-1 nodes is the original prompt
    prompt = first_row["Branch_Context"]

    # Build a mapping: for each row, figure out its "parent context"
    # (the Branch_Context, which is the context BEFORE this token was added)
    # and its "full context" (Branch_Context + Candidate_Token).

    # Use a stack to assign depths and IDs
    # Stack entry: (full_context, new_node_id, new_depth)
    stack = []
    branch_counters = defaultdict(int)  # parent_id -> next child number

    new_rows = []
    for row in rows:
        row_context = row["Branch_Context"]
        row_token = row["Candidate_Token"]
        row_full = row_context + row_token

        # Pop the stack until we find the parent whose full_context == row_context
        while stack and stack[-1][0] != row_context:
            stack.pop()

        if not stack:
            # Root level
            parent_id = ""
            new_depth = 1
        else:
            parent_id = stack[-1][1]
            new_depth = stack[-1][2] + 1

        # Assign branch number
        branch_counters[parent_id] += 1
        branch_num = branch_counters[parent_id]

        if parent_id:
            new_id = f"{parent_id}.{branch_num}"
        else:
            new_id = str(branch_num)

        # Update row
        new_row = dict(row)
        new_row["Node_ID"] = new_id
        new_row["Depth"] = new_depth

        # Determine rank: among siblings with the same parent
        # Rank = branch_num up to k
        new_row["Rank"] = branch_num

        # Recompute Is_Greedy: rank 1 = greedy
        new_row["Is_Greedy"] = branch_num == 1

        # Push onto stack
        stack.append((row_full, new_id, new_depth))

        new_rows.append(new_row)

    # Recompute cumulative log-prob and cumulative probability
    # by walking the tree: cumulative = sum of logprobs along the path
    node_map = {r["Node_ID"]: r for r in new_rows}
    for row in new_rows:
        nid = row["Node_ID"]
        # Walk up to root summing logprobs
        cum_logprob = 0.0
        current = nid
        while current:
            if current in node_map:
                cum_logprob += float(node_map[current]["LogProb"])
            parts = current.rsplit(".", 1)
            current = parts[0] if len(parts) == 2 else ""

        cum_prob = math.exp(cum_logprob) if cum_logprob > -700 else 0.0
        row["Cumulative_LogProb"] = cum_logprob
        row["Cumulative_Probability"] = cum_prob

    return new_rows


# ---------------------------------------------------------------------------
# Format and write output
# ---------------------------------------------------------------------------


def format_value(key, value):
    """Format a value for CSV output."""
    if key in ("Depth", "Rank"):
        return str(int(value))
    elif key == "Is_Greedy":
        if isinstance(value, bool):
            return str(value)
        return str(value)
    elif key in ("Probability", "Cumulative_Probability"):
        v = float(value)
        if v == 0.0:
            return "0.0"
        elif v < 0.001:
            return f"{v:.1e}"
        else:
            return f"{v:.6f}"
    elif key in ("LogProb", "Cumulative_LogProb"):
        v = float(value)
        return f"{v:.6f}"
    else:
        return value


def write_csv(rows, filepath):
    """Write processed rows to CSV."""
    if not rows:
        print("No rows to write!", file=sys.stderr)
        return

    fieldnames = [
        "Node_ID",
        "Depth",
        "Branch_Context",
        "Candidate_Token",
        "Rank",
        "Probability",
        "LogProb",
        "Cumulative_LogProb",
        "Cumulative_Probability",
        "Is_Greedy",
    ]

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        for row in rows:
            csv_row = []
            for key in fieldnames:
                val = row.get(key, "")
                csv_row.append(format_value(key, val))
            writer.writerow(csv_row)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_stats(original_rows, collapsed_rows):
    """Print before/after stats."""
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Collapse Partials — Summary", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Original rows:   {len(original_rows):>8}", file=sys.stderr)
    print(f"  Collapsed rows:  {len(collapsed_rows):>8}", file=sys.stderr)
    print(
        f"  Rows removed:    {len(original_rows) - len(collapsed_rows):>8}",
        file=sys.stderr,
    )
    reduction = (1 - len(collapsed_rows) / len(original_rows)) * 100
    print(f"  Reduction:       {reduction:>7.1f}%", file=sys.stderr)

    # Count unique tokens at each depth
    from collections import Counter

    orig_depths = Counter(int(r["Depth"]) for r in original_rows)
    new_depths = Counter(int(r["Depth"]) for r in collapsed_rows)
    max_orig = max(orig_depths.keys()) if orig_depths else 0
    max_new = max(new_depths.keys()) if new_depths else 0

    print(f"\n  Depth distribution:", file=sys.stderr)
    print(f"  {'Depth':<8} {'Original':<12} {'Collapsed':<12}", file=sys.stderr)
    for d in range(1, max(max_orig, max_new) + 1):
        print(
            f"  {d:<8} {orig_depths.get(d, 0):<12} {new_depths.get(d, 0):<12}",
            file=sys.stderr,
        )

    # Show some example merges
    print(
        f"\n  Sample merged tokens (first 10 collapsed partial words):", file=sys.stderr
    )
    count = 0
    for row in collapsed_rows:
        token = row["Candidate_Token"]
        # A merged token will be one that looks like it was combined
        # (has no leading space for middle part, but leading space at start,
        # and the combined result is longer than typical single tokens)
        if count >= 10:
            break
        # Heuristic: if Branch_Context shows more content than just prompt +
        # single word addition, it might be a merged token
        prob = float(row["Probability"])
        logprob = float(row["LogProb"])
        # Merged tokens will have logprob that's the sum of multiple tokens
        # We can check if the token contains what looks like a sub-word join
        # Actually just show tokens with their context for verification
        if " " not in token.strip() and len(token.strip()) > 5:
            # Might be a merged multi-subword token
            print(
                f"    {row['Branch_Context']} + [{token}] (P={prob:.6f})",
                file=sys.stderr,
            )
            count += 1

    print(f"{'=' * 60}\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Collapse partial-word tokens in a tree CSV into "
        "whole-word tokens by greedy concatenation."
    )
    parser.add_argument("input", help="Input CSV file (e.g. 4_wide_7_deep.csv)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV file. Defaults to <input>_words.csv",
    )
    args = parser.parse_args()

    if args.output is None:
        base = args.input.rsplit(".", 1)
        args.output = (
            f"{base[0]}_words.{base[1]}" if len(base) == 2 else f"{args.input}_words"
        )

    print(f"Reading {args.input}...", file=sys.stderr)
    original_rows = parse_csv(args.input)

    print(f"Building tree ({len(original_rows)} nodes)...", file=sys.stderr)
    nodes, children_map = build_tree(original_rows)

    print(f"Collapsing partial words...", file=sys.stderr)
    collapsed_rows = collapse_tree(nodes, children_map)

    print(f"Reassigning IDs and recomputing depths...", file=sys.stderr)
    final_rows = reassign_ids_and_depths(collapsed_rows)

    print(f"Writing {args.output}...", file=sys.stderr)
    write_csv(final_rows, args.output)

    print_stats(original_rows, final_rows)
    print(f"Done! Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
