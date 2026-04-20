"""
clean_tree.py

Post-processes a collapsed token probability tree CSV to remove noise:

  1. Remove trailing periods from Branch_Context strings.
  2. Remove rows with <|eom_id|> as a Candidate_Token.
  3. If <|eot_id|> appears in a Branch_Context, prune all children of that
     node — the <|eot_id|> candidate token is the terminal node.
  4. If a sibling group contains <|eot_id|>, replace tokens that signal the
     end of generation (punctuation, newlines) with <|eot_id|>.
  5. Merge identical candidate tokens within the same sibling group (case-
     insensitive), keeping the token with the highest probability and summing
     probabilities onto it.
  6. Flag branches whose candidate token is a partial word (using dictionary
     checks and metadata) in a new 'Is_Partial' column.
  7. Prune single-child <|eot_id|> branches.
  8. Reassign Node_IDs, ranks, and recompute cumulative probabilities.

Usage:
    python3 clean_tree.py 4_wide_7_deep_words.csv -o 4_wide_7_deep_clean_v#.csv
"""

import argparse
import csv
import math
import os
import string
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Attempt to load NLTK words if available
_NLTK_WORDS = None
try:
    import nltk

    # Note: This might trigger a download if not already present,
    # but we will catch potential failures.
    from nltk.corpus import words as nltk_words_corpus

    _NLTK_WORDS = set(w.lower() for w in nltk_words_corpus.words())
except (ImportError, LookupError):
    pass


def is_special_token(tok):
    """True for tokens like <|eot_id|>, <|eom_id|>, <|start_header_id|>, etc."""
    return tok.strip().startswith("<|") and tok.strip().endswith("|>")


def is_real_word_token(tok):
    """
    True if the token represents a real word (starts with a space followed
    by at least one alphabetic character). Partial-word continuations,
    punctuation, and special tokens are NOT real words.
    """
    if is_special_token(tok):
        return False
    if not tok:
        return False
    # Must start with a space (word boundary in tokenizer output)
    if not tok.startswith(" "):
        return False
    stripped = tok.strip()
    if not stripped:
        return False
    # Must start with a letter
    if not stripped[0].isalpha():
        return False
    return True


def is_terminator_token(tok):
    """
    True if the token looks like punctuation or whitespace that
    could signal the end of a generation.
    """
    if is_special_token(tok):
        return False
    s = tok.strip()
    if not s:
        return True  # Whitespace/newlines
    if s == "s":
        return True
    if s == "'<|eot_id|>":
        return True
    # If it consists only of punctuation
    return all(c in string.punctuation or c.isspace() for c in s)


def is_partial_word(tok, dictionary):
    """
    Returns True if the token appears to be a partial word fragment.
    """
    if is_special_token(tok):
        return False

    tok_str = tok.strip()
    if not tok_str:
        return False

    # Heuristic 1: No space prefix and starts with a letter (fragment/continuation)
    if not tok.startswith(" ") and tok_str[0].isalpha():
        return True

    # Heuristic 2: Known partial fragments from blocklist
    blocklist = {
        "eth",
        "glob",
        "intellig",
        "achie",
        "acc",
        "p",
        "un",
        "nation",
        "con",
        "cons",
        "intel",
    }
    if tok_str.lower() in blocklist:
        return True

    # Heuristic 3: Check against dictionary/NLTK
    if tok_str[0].isalpha():
        return not is_dictionary_word(tok_str, dictionary)

    return False


def prune_node_descendants(nid, nodes, children):
    """Recursively remove all descendant nodes from the tree structures."""
    if nid not in children:
        return
    # Use list() to avoid mutation issues during iteration
    for kid in list(children[nid]):
        k_id = kid["Node_ID"]
        prune_node_descendants(k_id, nodes, children)
        if k_id in nodes:
            del nodes[k_id]
    del children[nid]


def load_dictionary():
    """
    Load /usr/share/dict/words and expand with common morphological forms
    to handle plurals, verb forms, adjective forms, etc.
    """
    base_words = set()
    try:
        with open("/usr/share/dict/words", "r") as f:
            for line in f:
                base_words.add(line.strip().lower())
    except FileNotFoundError:
        print(
            "WARNING: /usr/share/dict/words not found, skipping dictionary check",
            file=sys.stderr,
        )
        return None

    # Expand with morphological variants
    expanded = set(base_words)
    for w in base_words:
        if len(w) < 2:
            continue
        # Plurals
        expanded.add(w + "s")
        expanded.add(w + "es")
        if w.endswith("y"):
            expanded.add(w[:-1] + "ies")  # policy -> policies
        # Verb forms
        expanded.add(w + "ed")
        expanded.add(w + "ing")
        expanded.add(w + "er")
        expanded.add(w + "tion")
        expanded.add(w + "ment")
        if w.endswith("e"):
            expanded.add(w[:-1] + "ing")  # achieve -> achieving
            expanded.add(w[:-1] + "ed")  # achieve -> achieved
            expanded.add(w[:-1] + "er")  # achieve -> achiever
            expanded.add(w[:-1] + "ation")  # regulate -> regulation
        if w.endswith("t"):
            expanded.add(w + "ting")  # commit -> committing
        # Adjective/adverb forms
        expanded.add(w + "ly")
        expanded.add(w + "al")
        expanded.add(w + "ic")
        expanded.add(w + "ical")
        expanded.add(w + "ive")
        expanded.add(w + "ous")
        # -ness, -ity
        expanded.add(w + "ness")
        expanded.add(w + "ity")
        # Un- prefix
        expanded.add("un" + w)
        # -ize / -ise
        expanded.add(w + "ize")
        expanded.add(w + "ise")
        # Comparative/superlative
        expanded.add(w + "er")
        expanded.add(w + "est")
        # -ful
        expanded.add(w + "ful")
        # -able / -ible
        expanded.add(w + "able")
        expanded.add(w + "ible")

    return expanded


def is_dictionary_word(word, dictionary):
    """
    Check if a word (stripped of spaces) is in the expanded dictionary.
    Also handles abbreviations, contractions with periods, etc.
    """
    w = word.lower().strip()
    if not w:
        return True

    # Whitelist of real words that might be missing from some dictionaries
    whitelist = {
        "groundbreaking",
        "robotic",
        "hugging",
        "landmark",
        "achieves",
        "unites",
        "reaches",
        "agrees",
        "historic",
        "guidelines",
        "standards",
        "regulations",
        "frameworks",
        "ethical",
        "morality",
        "artificial",
        "governance",
        "practises",
        "mentions",
        "agreement",
        "settlement",
        "accord",
        "pact",
        "consensus",
        "landmark",
    }
    if w in whitelist:
        return True

    # Check NLTK if was successfully loaded
    if _NLTK_WORDS and w in _NLTK_WORDS:
        return True

    if dictionary is None:
        return True  # If no dictionary, accept everything

    # Accept words with periods (e.g., A.I, U.S.)
    if "." in w:
        return True
    # Direct lookup
    if w in dictionary:
        return True
    # Try removing trailing 's' for possessives/plurals
    if w.endswith("'s") and w[:-2] in dictionary:
        return True
    return False


# ---------------------------------------------------------------------------
# Parse & build tree
# ---------------------------------------------------------------------------


def parse_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_tree(rows):
    """Returns nodes dict and children dict (parent_id -> [child rows])."""
    nodes = {}
    children = defaultdict(list)
    for row in rows:
        nid = row["Node_ID"]
        nodes[nid] = row
        parts = nid.rsplit(".", 1)
        if len(parts) == 2:
            children[parts[0]].append(row)
    return nodes, children


def get_root_ids(nodes):
    ids = [nid for nid, n in nodes.items() if "." not in nid]
    ids.sort(key=lambda x: int(x))
    return ids


# ---------------------------------------------------------------------------
# Step 1: Remove trailing periods from Branch_Context
# ---------------------------------------------------------------------------


def strip_trailing_periods(rows):
    """Remove trailing period(s) from Branch_Context."""
    for row in rows:
        ctx = row["Branch_Context"]
        # Only strip trailing periods (not valid punctuation inside context)
        while ctx.endswith("."):
            ctx = ctx[:-1]
        row["Branch_Context"] = ctx
    return rows


# ---------------------------------------------------------------------------
# Step 2: Remove <|eom_id|> rows
# ---------------------------------------------------------------------------


def remove_eom_rows(rows):
    """Remove rows where Candidate_Token is <|eom_id|>."""
    return [r for r in rows if r["Candidate_Token"].strip() != "<|eom_id|>"]


# ---------------------------------------------------------------------------
# Step 3: Prune children of nodes with <|eot_id|> in Branch_Context
# ---------------------------------------------------------------------------


def prune_eot_children(nodes, children):
    """
    If a node's Branch_Context contains <|eot_id|>, it means the parent
    already generated an <|eot_id|> token. Any children after that are
    meaningless continuations — prune them all.

    This means: any node whose Branch_Context contains <|eot_id|> should
    be removed, along with all its descendants.
    """
    to_remove = set()

    for nid, node in nodes.items():
        if "<|eot_id|>" in node["Branch_Context"]:
            to_remove.add(nid)

    # Also remove descendants of removed nodes
    changed = True
    while changed:
        changed = False
        for nid in list(nodes.keys()):
            if nid in to_remove:
                continue
            parts = nid.rsplit(".", 1)
            if len(parts) == 2 and parts[0] in to_remove:
                to_remove.add(nid)
                changed = True

    # Remove
    for nid in to_remove:
        if nid in nodes:
            del nodes[nid]

    # Rebuild children
    new_children = defaultdict(list)
    for nid, node in nodes.items():
        parts = nid.rsplit(".", 1)
        if len(parts) == 2 and parts[0] in nodes:
            new_children[parts[0]].append(node)

    return nodes, new_children


# ---------------------------------------------------------------------------
# Step 4: Prune sibling groups with only punct/special/partial + <|eot_id|>
# ---------------------------------------------------------------------------


def resolve_terminators_and_merge(nodes, children):
    """
    Step 4: Resolve terminators -> <|eot_id|>
    If a sibling group has <|eot_id|>, replace other end-of-generation-like
    tokens (punctuation, newlines) with <|eot_id|>.

    Step 5: Merge identical candidate tokens (theoretically only <|eot_id|>).
    Sum their probabilities and update logprobs.
    """
    # Pass 1: Replace terminators with <|eot_id|>
    for parent_id, kids in list(children.items()):
        has_eot = any(k["Candidate_Token"].strip() == "<|eot_id|>" for k in kids)
        if has_eot:
            for k in kids:
                tok = k["Candidate_Token"]
                if tok.strip() == "<|eot_id|>":
                    continue
                if is_terminator_token(tok):
                    k["Candidate_Token"] = "<|eot_id|>"
                    # Once converted to EOT, it shouldn't have children
                    prune_node_descendants(k["Node_ID"], nodes, children)

    # Pass 2: Merge identical tokens (case-insensitive)
    for parent_id in list(children.keys()):
        kids = children.get(parent_id)
        if not kids:
            continue

        # Group by lowercase token to handle case-insensitive identical-ness
        groups = defaultdict(list)
        for k in kids:
            groups[k["Candidate_Token"].lower()].append(k)

        new_kids = []
        to_del = []

        for lower_tok, group in groups.items():
            if len(group) == 1:
                new_kids.append(group[0])
                continue

            # Multiple capitalizations or identical tokens found
            # Keep the token with the highest probability as the representative
            def get_prob(node):
                p = float(node.get("Probability", 0))
                # Fallback to exponentiated LogProb if Probability is 0.0 or missing
                if p == 0:
                    val = node.get("LogProb", -999)
                    try:
                        p = math.exp(float(val))
                    except (ValueError, OverflowError):
                        p = 0.0
                return p

            # Sort group by probability descending to pick the best master
            group.sort(key=get_prob, reverse=True)
            master = group[0]
            others = group[1:]

            # Sum all other probabilities onto the master token
            master_p = get_prob(master)
            total_p = master_p
            for other in others:
                other_p = get_prob(other)
                total_p += other_p
                # Mark redundant branches for removal
                to_del.append(other["Node_ID"])
                prune_node_descendants(other["Node_ID"], nodes, children)

            master["Probability"] = total_p
            master["LogProb"] = math.log(total_p) if total_p > 0 else -999.0
            new_kids.append(master)

        # Update the children map for this parent and cleanup nodes dict
        if to_del:
            children[parent_id] = new_kids
            for nid in to_del:
                if nid in nodes:
                    del nodes[nid]

    return nodes, children


# ---------------------------------------------------------------------------
# Step 5: Prune non-dictionary word tokens
# ---------------------------------------------------------------------------


def flag_partial_words(nodes, children, dictionary):
    """
    Step 6: Flag partial-word tokens.
    Iterate through all nodes and mark them as 'Is_Partial' based on
    dictionary checks and heuristics.
    """
    for nid, node in nodes.items():
        # Root nodes are usually safe, but check anyway
        tok = node["Candidate_Token"]

        if is_partial_word(tok, dictionary):
            node["Is_Partial"] = True
        else:
            node["Is_Partial"] = False

    return nodes, children


# ---------------------------------------------------------------------------
# Step 6: Prune single-child <|eot_id|> branches
# ---------------------------------------------------------------------------


def prune_singleton_eot(nodes, children):
    """
    If a node has exactly one child and that child is <|eot_id|>,
    remove that child. An EOT token should only exist if it's part
    of a probability distribution with other alternatives.
    """
    changed = True
    while changed:
        changed = False
        to_remove = set()

        # Rebuild children map
        children = defaultdict(list)
        for nid, node in nodes.items():
            parts = nid.rsplit(".", 1)
            if len(parts) == 2 and parts[0] in nodes:
                children[parts[0]].append(node)

        for parent_id, kids in children.items():
            if len(kids) == 1:
                k = kids[0]
                if k["Candidate_Token"].strip() == "<|eot_id|>":
                    to_remove.add(k["Node_ID"])

        if to_remove:
            for nid in to_remove:
                if nid in nodes:
                    del nodes[nid]
            changed = True

    # Final children rebuild
    final_children = defaultdict(list)
    for nid, node in nodes.items():
        parts = nid.rsplit(".", 1)
        if len(parts) == 2 and parts[0] in nodes:
            final_children[parts[0]].append(node)

    return nodes, final_children


# ---------------------------------------------------------------------------
# Step 7: Reassign IDs, ranks, depths, and recompute cumulatives
# ---------------------------------------------------------------------------


def rebuild_tree(nodes, children):
    """
    Walk the tree in DFS order, reassigning Node_IDs, depths, ranks,
    and recomputing cumulative log-probs.
    """
    root_ids = [nid for nid in nodes if "." not in nid]
    root_ids.sort(key=lambda x: int(x))

    output_rows = []
    new_id_map = {}  # old_id -> new_id

    def walk(old_id, new_id, depth, cum_logprob):
        node = dict(nodes[old_id])
        logprob = float(node["LogProb"])
        new_cum = cum_logprob + logprob

        node["Node_ID"] = new_id
        node["Depth"] = depth
        node["Cumulative_LogProb"] = new_cum
        node["Cumulative_Probability"] = math.exp(new_cum) if new_cum > -700 else 0.0
        node["Is_Partial"] = node.get("Is_Partial", False)

        new_id_map[old_id] = new_id
        output_rows.append(node)

        # Process children, sorted by rank
        kids = children.get(old_id, [])
        kids.sort(key=lambda k: int(k["Rank"]))
        for i, kid in enumerate(kids, 1):
            kid_new_id = f"{new_id}.{i}"
            kid_node = dict(nodes[kid["Node_ID"]])
            kid_node["Rank"] = i
            kid_node["Is_Greedy"] = i == 1
            nodes[kid["Node_ID"]] = kid_node
            walk(kid["Node_ID"], kid_new_id, depth + 1, new_cum)

    for i, root_id in enumerate(root_ids, 1):
        root_node = dict(nodes[root_id])
        root_node["Rank"] = i
        root_node["Is_Greedy"] = i == 1
        nodes[root_id] = root_node
        walk(root_id, str(i), 1, 0.0)

    return output_rows


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------


def format_value(key, value):
    if key in ("Depth", "Rank"):
        return str(int(value))
    elif key == "Is_Greedy":
        return str(bool(value))
    elif key in ("Probability", "Cumulative_Probability"):
        v = float(value)
        if v == 0.0:
            return "0.0"
        elif v < 0.001:
            return f"{v:.1e}"
        else:
            return f"{v:.6f}"
    elif key in ("LogProb", "Cumulative_LogProb"):
        return f"{float(value):.6f}"
    elif key in ("Is_Greedy", "Is_Partial"):
        return str(bool(value))
    else:
        return value


def write_csv(rows, filepath):
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
        "Is_Partial",
    ]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([format_value(k, row.get(k, "")) for k in fieldnames])


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_stats(original, cleaned):
    from collections import Counter

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("  Clean Tree — Summary", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Original rows:   {len(original):>8}", file=sys.stderr)
    print(f"  Cleaned rows:    {len(cleaned):>8}", file=sys.stderr)
    print(f"  Rows removed:    {len(original) - len(cleaned):>8}", file=sys.stderr)
    reduction = (1 - len(cleaned) / len(original)) * 100 if original else 0
    print(f"  Reduction:       {reduction:>7.1f}%", file=sys.stderr)

    new_depths = Counter(int(r["Depth"]) for r in cleaned)
    print("\n  Depth distribution (cleaned):", file=sys.stderr)
    for d in sorted(new_depths.keys()):
        print(f"    depth {d}: {new_depths[d]} nodes", file=sys.stderr)

    # Count remaining special tokens
    eot_count = sum(1 for r in cleaned if r["Candidate_Token"].strip() == "<|eot_id|>")
    eom_count = sum(1 for r in cleaned if r["Candidate_Token"].strip() == "<|eom_id|>")
    special = sum(1 for r in cleaned if is_special_token(r["Candidate_Token"]))
    period_ctx = sum(1 for r in cleaned if r["Branch_Context"].endswith("."))
    eot_in_ctx = sum(1 for r in cleaned if "<|eot_id|>" in r["Branch_Context"])

    print(f"\n  Remaining special tokens:", file=sys.stderr)
    print(f"    <|eot_id|> candidate tokens: {eot_count}", file=sys.stderr)
    print(f"    <|eom_id|> candidate tokens: {eom_count}", file=sys.stderr)
    print(f"    Total special tokens:        {special}", file=sys.stderr)
    print(f"    Trailing '.' in context:     {period_ctx}", file=sys.stderr)
    print(f"    <|eot_id|> in context:       {eot_in_ctx}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Clean a collapsed token tree CSV by removing noise."
    )
    parser.add_argument("input", help="Input CSV (e.g. 4_wide_7_deep_words.csv)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV file. Defaults to <input>_clean.csv",
    )
    args = parser.parse_args()

    if args.output is None:
        name, ext = os.path.splitext(args.input)

        # If input is already like *_clean.csv, try to find *_clean_v2.csv etc.
        if "_clean" in name:
            # Check if it ends in _vN
            import re

            m = re.search(r"_v(\d+)$", name)
            if m:
                v = int(m.group(1)) + 1
                base_name = re.sub(r"_v\d+$", "", name)
                args.output = f"{base_name}_v{v}{ext}"
            else:
                args.output = f"{name}_v2{ext}"
        else:
            final_name = f"{name}_clean{ext}"
            # Check if it exists, if so increment
            if os.path.exists(final_name):
                args.output = f"{name}_clean_v2{ext}"
            else:
                args.output = final_name

    print(f"Reading {args.input}...", file=sys.stderr)
    original_rows = parse_csv(args.input)

    # Step 1: Strip trailing periods from Branch_Context
    print("Step 1: Stripping trailing periods from Branch_Context...", file=sys.stderr)
    rows = strip_trailing_periods(list(original_rows))

    # Step 2: Remove <|eom_id|> rows
    print("Step 2: Removing <|eom_id|> rows...", file=sys.stderr)
    rows = remove_eom_rows(rows)

    # Build tree for structural operations
    print("Building tree...", file=sys.stderr)
    nodes, children = build_tree(rows)

    # Step 3: Prune children of <|eot_id|> context nodes
    print("Step 3: Pruning children of <|eot_id|> context nodes...", file=sys.stderr)
    nodes, children = prune_eot_children(nodes, children)

    # Step 4: Resolve terminators and merge identical tokens
    print(
        "Step 4: Resolving terminators and merging identical tokens...", file=sys.stderr
    )
    nodes, children = resolve_terminators_and_merge(nodes, children)

    # Step 5: Loading dictionary
    print("Step 5: Loading dictionary...", file=sys.stderr)
    dictionary = load_dictionary()

    # Step 6: Flag partial-word tokens
    print("Step 6: Flagging partial-word tokens...", file=sys.stderr)
    nodes, children = flag_partial_words(nodes, children, dictionary)

    # Step 7: Prune single-child <|eot_id|> branches
    print("Step 7: Pruning single-child <|eot_id|> branches...", file=sys.stderr)
    nodes, children = prune_singleton_eot(nodes, children)

    # Step 8: Rebuild tree with clean IDs
    print("Step 8: Rebuilding tree with clean IDs...", file=sys.stderr)
    cleaned_rows = rebuild_tree(nodes, children)

    # Write output
    print(f"Writing {args.output}...", file=sys.stderr)
    write_csv(cleaned_rows, args.output)

    print_stats(original_rows, cleaned_rows)
    print(f"Done! Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
