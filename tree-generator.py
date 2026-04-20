"""
tree-generator.py

Standalone script to exhaustively explore the token probability tree for a
given prompt (and optional prior answer) using the HuggingFace Chat Completion
endpoint (OpenAI-compatible SDK).

The branching factor at each node is controlled by two parameters that can be
used independently or together:

  top_logprobs (1–5)  Server-side: the API returns this many candidate tokens
                       per position.  Acts as a hard top-k cap.  The HuggingFace
                       Chat Completion spec allows values 1–5.

  top_p (0.0–1.0)     Client-side nucleus filter applied *after* the API
                       returns candidates.  Keeps only the smallest set of
                       tokens whose cumulative probability >= top_p.  Use 1.0
                       to disable (keep everything the API returned).

Outputs a CSV with columns:
    Node_ID, Depth, Branch_Context, Candidate_Token, Rank,
    Probability, LogProb, Cumulative_LogProb, Cumulative_Probability, Is_Greedy

Usage (CLI):
    python tree-generator.py \
        --api-key "$HF_TOKEN" \
        --prompt "Write a haiku about the ocean." \
        --depth 3 --top-logprobs 5

    python tree-generator.py \
        --api-key "$HF_TOKEN" \
        --prompt "Summarize this concept:" \
        --answer "The key idea is" \
        --depth 2 --top-p 0.9 \
        --output tree.csv

Usage (as a library):
    from tree_generator import generate_tree
    csv_string = generate_tree(
        api_key="hf_...",
        prompt="Write a haiku about the ocean.",
        depth=3,
        top_logprobs=5,
    )
"""

import argparse
import math
import os
import sys
import time
from collections import namedtuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cheapest"
API_BASE_URL = "https://router.huggingface.co/v1"

TokenProb = namedtuple("TokenProb", ["token", "logprob"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(api_key):
    """Return an OpenAI-compatible client pointed at HuggingFace."""
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def _build_messages(system_prompt, user_prompt, assistant_text=""):
    """Build a chat-completion messages list."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _apply_top_p_filter(candidates, top_p):
    """
    Apply top-p (nucleus) filter to a descending-sorted list of TokenProb
    objects.  Keeps the smallest set whose cumulative probability >= top_p.
    Returns the filtered list.  Pass top_p=1.0 to disable.
    """
    if top_p and top_p < 1.0:
        filtered = []
        cumulative = 0.0
        for c in candidates:
            if cumulative >= top_p:
                break
            filtered.append(c)
            cumulative += math.exp(c.logprob)
        return filtered
    return candidates


_INITIAL_BACKOFF = 2      # seconds
_MAX_BACKOFF = 120        # cap so we don't wait forever
_BACKOFF_MULTIPLIER = 2


def _get_candidates(client, model, messages, temperature, top_logprobs, top_p):
    """
    Call the chat-completion endpoint for a single next token and return a
    sorted, filtered list of TokenProb candidates.

    top_logprobs (1–5) controls how many candidates the API returns (server-
    side top-k).  top_p is applied client-side afterward as a nucleus filter.

    When the last message has role "assistant", we pass
    ``continue_final_message=True`` so the API treats it as an incomplete
    prefix to continue from (rather than a finished turn that triggers a
    brand-new response).

    Retries indefinitely on transient / rate-limit errors with exponential
    backoff (2 s → 4 s → … → 120 s cap) so long-running generations are
    never lost to a temporary API hiccup.
    """
    # If the conversation ends with an assistant message, tell the backend
    # to continue it rather than start a new turn.
    extra = {}
    if messages and messages[-1]["role"] == "assistant":
        extra["continue_final_message"] = True

    backoff = _INITIAL_BACKOFF
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=temperature,
                extra_body=extra,
            )
            choice = response.choices[0]
            candidates = [
                TokenProb(tl.token, tl.logprob)
                for tl in choice.logprobs.content[0].top_logprobs
            ]
            candidates.sort(key=lambda x: x.logprob, reverse=True)
            return _apply_top_p_filter(candidates, top_p)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(
                f"  [retry] API error: {exc}  — retrying in {backoff}s …",
                file=sys.stderr,
            )
            time.sleep(backoff)
            backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)


def _rows_to_csv(rows):
    """Convert a list of row dicts to a CSV string."""
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for col in columns:
            val = row[col]
            # Quote strings that may contain commas/newlines
            if isinstance(val, str):
                val = '"' + val.replace('"', '""') + '"'
            else:
                val = str(val)
            values.append(val)
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Core: tree generator
# ---------------------------------------------------------------------------

def generate_tree(
    api_key,
    prompt,
    answer="",
    system_prompt="You are a helpful assistant.",
    model=DEFAULT_MODEL,
    depth=3,
    top_logprobs=5,
    top_p=1.0,
    temperature=1.0,
):
    """
    Exhaustively explore the token probability tree up to *depth* levels.

    The branching factor is controlled by two complementary parameters:

      * **top_logprobs** — sent to the API, which returns exactly this many
        candidate tokens per position (server-side top-k).  Valid range: 1–5.
      * **top_p** — applied client-side after the API response; keeps only
        the smallest set of tokens whose cumulative probability >= top_p.
        Set to 1.0 to disable.

    Parameters
    ----------
    api_key : str
        HuggingFace API token (must have Inference Providers permission).
    prompt : str
        The user prompt to send to the model.
    answer : str, optional
        Existing assistant text on which to continue building the tree.
        Defaults to "" (start from scratch).
    system_prompt : str, optional
        System prompt.  Defaults to "You are a helpful assistant."
    model : str, optional
        HuggingFace model identifier.
    depth : int, optional
        How many token positions deep to explore (default 3).
    top_logprobs : int, optional
        Number of most-probable tokens the API should return at each
        position (1–5).  This is the server-side branching factor.
        Default 5.
    top_p : float, optional
        Nucleus-sampling cutoff (0.0–1.0).  Applied client-side after
        receiving candidates.  1.0 means disabled (keep all).
    temperature : float, optional
        Sampling temperature (0.0–2.0).

    Returns
    -------
    str
        CSV-formatted string of the full tree.  Columns:
        Node_ID, Depth, Branch_Context, Candidate_Token, Rank,
        Probability, LogProb, Cumulative_LogProb, Cumulative_Probability,
        Is_Greedy
    """
    client = _get_client(api_key)
    rows = []
    api_calls = [0]

    # Pre-compute total API calls for the progress display.
    # Each internal node triggers one API call.  With branching factor k
    # and depth d the total is k^0 + k^1 + … + k^(d-1) = (k^d - 1)/(k-1).
    k = top_logprobs  # worst-case branching (top_p may trim some)
    if k <= 1:
        total_calls_est = depth
    else:
        total_calls_est = (k ** depth - 1) // (k - 1)

    def _progress(node_id, context_snippet):
        """Print a single-line progress update to stderr."""
        # Truncate context to keep the line readable.
        ctx = context_snippet
        if len(ctx) > 60:
            ctx = "…" + ctx[-57:]
        print(
            f"\r  [{api_calls[0]}/{total_calls_est} calls]  "
            f"node {node_id}  ctx={ctx!r}",
            end="",
            file=sys.stderr,
        )

    def _recurse(current_text, depth_remaining, path, cumulative_logprob):
        if depth_remaining == 0:
            return

        messages = _build_messages(system_prompt, prompt, current_text)

        api_calls[0] += 1
        node_label = path if path else "(root)"
        _progress(node_label, current_text or "(empty)")
        candidates = _get_candidates(
            client, model, messages, temperature, top_logprobs, top_p
        )

        for rank, cand in enumerate(candidates):
            node_id = f"{path}.{rank + 1}" if path else str(rank + 1)
            prob = math.exp(cand.logprob)
            node_cumulative_lp = cumulative_logprob + cand.logprob

            rows.append({
                "Node_ID": node_id,
                "Depth": len(node_id.split(".")),
                "Branch_Context": current_text,
                "Candidate_Token": cand.token,
                "Rank": rank + 1,
                "Probability": round(prob, 6),
                "LogProb": round(cand.logprob, 6),
                "Cumulative_LogProb": round(node_cumulative_lp, 6),
                "Cumulative_Probability": round(
                    math.exp(node_cumulative_lp), 6
                ),
                "Is_Greedy": rank == 0,
            })

            _recurse(
                current_text + cand.token,
                depth_remaining - 1,
                node_id,
                node_cumulative_lp,
            )

    _recurse(answer, depth, "", 0.0)

    # Clear the progress line, then print the final summary.
    print("", file=sys.stderr)
    print(
        f"Tree complete: {len(rows)} nodes across {depth} depth(s), "
        f"{api_calls[0]} API call(s).",
        file=sys.stderr,
    )

    return _rows_to_csv(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively explore the top-k / top-p token probability tree "
            "for a prompt using the HuggingFace Chat Completion endpoint."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace API token (default: $HF_TOKEN env var).",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="The user prompt to send to the model.",
    )
    parser.add_argument(
        "--answer", default="",
        help="Optional existing assistant answer to build on.",
    )
    parser.add_argument(
        "--system-prompt", default="You are a helpful assistant.",
        help="System prompt (default: 'You are a helpful assistant.').",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Tree depth — token positions to explore (default: 3).",
    )
    parser.add_argument(
        "--top-logprobs", type=int, default=5,
        help="Candidates per position (1–5, sent to API as top_logprobs). Default: 5.",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-P nucleus filter (1.0 = disabled, default: 1.0).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path.  Omit to print CSV to stdout.",
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "No API key provided.  Use --api-key or set $HF_TOKEN."
        )

    csv_output = generate_tree(
        api_key=args.api_key,
        prompt=args.prompt,
        answer=args.answer,
        system_prompt=args.system_prompt,
        model=args.model,
        depth=args.depth,
        top_logprobs=args.top_logprobs,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(csv_output)
        print(f"CSV written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(csv_output)


if __name__ == "__main__":
    main()
