import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import math

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Probability Explorer v2")

# --- Constants & Defaults ---
DEFAULT_MODEL = "llama3.1-8b"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "context_locked" not in st.session_state:
    st.session_state.context_locked = False
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = "Write a headline for a long form journalistic article about AI ethics agreement reached across the EU."
if "word_mode" not in st.session_state:
    st.session_state.word_mode = True
if "custom_token_key" not in st.session_state:
    st.session_state.custom_token_key = 0
if "tree_csv" not in st.session_state:
    st.session_state.tree_csv = None
if "tree_running" not in st.session_state:
    st.session_state.tree_running = False
if "temp_val" not in st.session_state:
    st.session_state.temp_val = 1.0
if "top_p_val" not in st.session_state:
    st.session_state.top_p_val = 1.0
if "top_k_val" not in st.session_state:
    st.session_state.top_k_val = 0

# --- Defaults ---
DEFAULT_TEMP = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0

# --- Helper Functions ---


def is_continuation(token):
    """Checks if a token is a continuation of a word (no leading space/newline)."""
    if not token:
        return False
    # Llama 3 tokens typically start with a space ' ' for new words.
    # Other boundaries include newlines and tabs.
    return not (
        token.startswith(" ") or token.startswith("\n") or token.startswith("\t")
    )


def apply_sampling_filters(candidates, top_p, top_k):
    """
    Apply top-p (nucleus) and top-k filters to a sorted list of TokenProb objects.
    Candidates must already be sorted descending by logprob.
    Returns filtered list.
    """
    # Apply top-k first
    if top_k and top_k > 0:
        candidates = candidates[:top_k]

    # Apply top-p (nucleus sampling): keep tokens whose cumulative probability <= top_p
    if top_p and top_p < 1.0:
        filtered = []
        cumulative = 0.0
        for c in candidates:
            prob = math.exp(c.logprob)
            if cumulative >= top_p:
                break
            filtered.append(c)
            cumulative += prob
        candidates = filtered

    return candidates


def get_client(api_key):
    return OpenAI(base_url=CEREBRAS_BASE_URL, api_key=api_key)


def build_llama_prompt():
    """Constructs the raw Llama 3 prompt for text completion."""
    sys = st.session_state.system_prompt
    user = st.session_state.user_prompt

    # Construct history string
    assistant_content = ""
    if st.session_state.history:
        assistant_content = "".join(
            [item["token"] for item in st.session_state.history]
        )

    # Llama 3 Format
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_content}"
    return prompt


def build_full_prompt_debug():
    return build_llama_prompt()


def analyze_next_step(api_key, model, temp, top_p, top_k):
    """
    Runs inference to 'peek' at probabilities.
    If word_mode is enabled, it looksahead to complete the word.
    """
    client = get_client(api_key)
    full_prompt = build_llama_prompt()

    try:
        # If in word mode, we peek ahead multiple tokens to find the word boundary
        max_tokens = 10 if st.session_state.word_mode else 1

        response = client.completions.create(
            model=model,
            prompt=full_prompt,
            max_tokens=max_tokens,
            logprobs=20,
            temperature=temp,
            top_p=top_p,
        )

        choice = response.choices[0]

        # Helper class shim
        class TokenProb:
            def __init__(self, token, logprob):
                self.token = token
                self.logprob = logprob

        if st.session_state.word_mode and choice.logprobs:
            tokens = choice.logprobs.tokens
            top_logprobs_list = choice.logprobs.top_logprobs

            # Build predicted_token by following the greedy path until a word boundary
            word_tokens = []
            for i, t in enumerate(tokens):
                if i > 0 and not is_continuation(t):
                    break
                word_tokens.append(t)
            predicted_token = "".join(word_tokens)

            # For each top candidate at position 0, complete it to a full word
            # by appending continuation tokens from the greedy sequence.
            # The greedy suffix (tokens[1:]) is the best available approximation
            # of what follows each candidate first-token.
            greedy_suffix = []
            for t in tokens[1:]:
                if not is_continuation(t):
                    break
                greedy_suffix.append(t)

            top_dict = top_logprobs_list[0]
            candidates = []
            for t, lp in top_dict.items():
                if is_continuation(t):
                    # Candidate is a continuation fragment — complete it with the greedy suffix
                    completed = t + "".join(greedy_suffix)
                else:
                    # Candidate starts a new word (has leading space/newline) — use as-is
                    completed = t
                candidates.append(TokenProb(completed, lp))
        else:
            predicted_token = choice.text
            top_dict = choice.logprobs.top_logprobs[0]
            candidates = []
            for t, lp in top_dict.items():
                candidates.append(TokenProb(t, lp))

        candidates.sort(key=lambda x: x.logprob, reverse=True)
        candidates = apply_sampling_filters(candidates, top_p, top_k)

        st.session_state.current_analysis = {
            "predicted_token": predicted_token,
            "candidates": candidates,
            "top_p": top_p,
            "top_k": top_k,
        }

    except Exception as e:
        st.error(f"Error calling Cerebras API: {e}")


def fast_forward(api_key, model, temp, top_p, top_k, num_tokens):
    full_prompt = build_llama_prompt()
    client = get_client(api_key)

    # In word mode, we might need more actual tokens to satisfy the 'num_tokens' as words
    # but for now, we'll keep the request limit simple.
    actual_max_tokens = num_tokens if not st.session_state.word_mode else num_tokens * 3

    try:
        response = client.completions.create(
            model=model,
            prompt=full_prompt,
            max_tokens=actual_max_tokens,
            logprobs=5,  # Minimal logprobs for history
            temperature=temp,
            top_p=top_p,
            # extra_body={"top_k": top_k},
        )

        if response.choices[0].logprobs:
            c = response.choices[0]
            tokens = c.logprobs.tokens
            top_logprobs_list = c.logprobs.top_logprobs

            # Helper class shim
            class TokenProb:
                def __init__(self, token, logprob):
                    self.token = token
                    self.logprob = logprob

            current_word_tokens = []
            current_word_cands = []

            for i, token_str in enumerate(tokens):
                # top_logprobs_list[i] is a dict {token: lp}
                # We need to convert it to a list of TokenProb objects
                current_cands = []
                if top_logprobs_list and i < len(top_logprobs_list):
                    for t, lp in top_logprobs_list[i].items():
                        current_cands.append(TokenProb(t, lp))
                    current_cands.sort(key=lambda x: x.logprob, reverse=True)

                if st.session_state.word_mode:
                    # Grouping logic
                    if i > 0 and not is_continuation(token_str) and current_word_tokens:
                        # Commit the previous word
                        st.session_state.history.append(
                            {
                                "token": "".join(current_word_tokens),
                                "candidates": current_word_cands,
                            }
                        )
                        current_word_tokens = []
                        current_word_cands = []

                    current_word_tokens.append(token_str)
                    if not current_word_cands:
                        current_word_cands = current_cands
                else:
                    # Token mode
                    st.session_state.history.append(
                        {"token": token_str, "candidates": current_cands}
                    )

            # Final word commit if in word mode
            if st.session_state.word_mode and current_word_tokens:
                st.session_state.history.append(
                    {
                        "token": "".join(current_word_tokens),
                        "candidates": current_word_cands,
                    }
                )

        st.session_state.current_analysis = None
        st.rerun()

    except Exception as e:
        st.error(f"Error in Fast Forward: {e}")


def commit_step(token_to_commit):
    """Commits a token (from any source) to history."""
    # If we have analysis data, we save it. If we are force-injecting a custom token
    # without prior analysis (rare in this flow), we save empty candidates.
    candidates = []
    if st.session_state.current_analysis:
        candidates = st.session_state.current_analysis["candidates"]

    # Force a new list assignment to ensure Streamlit detects the state change robustly
    new_entry = {"token": token_to_commit, "candidates": candidates}
    st.session_state.history = st.session_state.history + [new_entry]

    st.session_state.current_analysis = None
    st.rerun()


def undo_last_step():
    st.session_state.history.pop()
    st.session_state.current_analysis = None
    st.rerun()


def convert_history_to_csv():
    rows = []
    for step_idx, step_data in enumerate(st.session_state.history):
        chosen = step_data["token"]
        # If the user injected a custom token, it might not be in the candidate list.
        # We still list the candidates that WERE there, but mark none as chosen if match fails.

        for rank, cand in enumerate(step_data["candidates"]):
            prob = math.exp(cand.logprob)
            rows.append(
                {
                    "Step_Index": step_idx + 1,
                    "Chosen_Token_At_Step": chosen,
                    "Candidate_Token": cand.token,
                    "Rank": rank + 1,
                    "Probability": round(prob, 2),
                    "Is_Actual_Choice": (cand.token == chosen),
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def get_candidates_for_prompt(client, model, prompt, temp, top_p, top_k, word_mode):
    """
    Call the API for a given raw prompt string and return a list of (token, logprob) tuples,
    filtered by top-p/top-k. Also returns the greedy predicted token.
    """

    class TokenProb:
        def __init__(self, token, logprob):
            self.token = token
            self.logprob = logprob

    max_tokens = 10 if word_mode else 1
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        logprobs=20,
        temperature=temp,
        top_p=top_p,
    )
    choice = response.choices[0]

    if word_mode and choice.logprobs:
        tokens = choice.logprobs.tokens
        top_logprobs_list = choice.logprobs.top_logprobs

        # Greedy predicted token (full word)
        word_tokens = []
        for i, t in enumerate(tokens):
            if i > 0 and not is_continuation(t):
                break
            word_tokens.append(t)
        predicted_token = "".join(word_tokens)

        # Greedy suffix for completing candidates
        greedy_suffix = []
        for t in tokens[1:]:
            if not is_continuation(t):
                break
            greedy_suffix.append(t)

        top_dict = top_logprobs_list[0]
        candidates = []
        for t, lp in top_dict.items():
            completed = (t + "".join(greedy_suffix)) if is_continuation(t) else t
            candidates.append(TokenProb(completed, lp))
    else:
        predicted_token = choice.text
        top_dict = choice.logprobs.top_logprobs[0]
        candidates = [TokenProb(t, lp) for t, lp in top_dict.items()]

    candidates.sort(key=lambda x: x.logprob, reverse=True)
    candidates = apply_sampling_filters(candidates, top_p, top_k)
    return predicted_token, candidates


def build_llama_prompt_from_text(sys_prompt, user_prompt, assistant_so_far):
    """Build a raw Llama 3 prompt from explicit text (not session state)."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_so_far}"
    )


def explore_tree(
    api_key, model, temp, top_p, top_k, depth, word_mode, status_placeholder
):
    """
    Recursively explore the probability tree up to `depth` levels deep.
    Returns a flat list of row dicts for CSV export.

    Each row represents one candidate at one node, with columns:
      - Depth: how deep in the tree (1 = first generation)
      - Node_ID: dot-separated path, e.g. "1.2.3"
      - Context_So_Far: the text generated by the current branch to reach this node
      - Candidate_Token: this candidate token
      - Rank: rank among siblings
      - Probability: probability of this candidate
      - LogProb: log probability
      - Cumulative_LogProb: sum of logprobs along the path to this node
      - Is_Greedy: whether this candidate was the greedy (top-1) pick
    """
    client = get_client(api_key)
    sys_prompt = st.session_state.system_prompt
    user_prompt = st.session_state.user_prompt
    history_text = "".join([item["token"] for item in st.session_state.history])

    rows = []
    total_calls = [0]

    def recurse(current_text, depth_remaining, path, cumulative_logprob):
        if depth_remaining == 0:
            return

        prompt = build_llama_prompt_from_text(sys_prompt, user_prompt, current_text)

        try:
            total_calls[0] += 1
            status_placeholder.info(
                f"🌳 Exploring tree... {total_calls[0]} API call(s) made. Current path: {path or 'root'}"
            )
            predicted_token, candidates = get_candidates_for_prompt(
                client, model, prompt, temp, top_p, top_k, word_mode
            )
        except Exception as e:
            status_placeholder.error(f"API error at path {path}: {e}")
            return

        for rank, cand in enumerate(candidates):
            node_id = f"{path}.{rank + 1}" if path else str(rank + 1)
            prob = math.exp(cand.logprob)
            node_cumulative_lp = cumulative_logprob + cand.logprob

            rows.append(
                {
                    "Depth": len(node_id.split(".")),
                    "Node_ID": node_id,
                    "Context_So_Far": current_text,
                    "Candidate_Token": cand.token,
                    "Rank": rank + 1,
                    "Probability": round(prob, 6),
                    "LogProb": round(cand.logprob, 6),
                    "Cumulative_LogProb": round(node_cumulative_lp, 6),
                    "Cumulative_Probability": round(math.exp(node_cumulative_lp), 6),
                    "Is_Greedy": (rank == 0),
                }
            )

            # Recurse: append the candidate token to context and go deeper
            recurse(
                current_text + cand.token,
                depth_remaining - 1,
                node_id,
                node_cumulative_lp,
            )

    recurse(history_text, depth, "", 0.0)
    return rows


def tree_rows_to_csv(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


with st.sidebar:
    st.header("⚙️ Configuration")

    # api_key = st.text_input("Cerebras API Key", type="password")
    # Using secrets instead for security/convenience
    if "CEREBRAS_API_KEY" in st.secrets:
        api_key = st.secrets["CEREBRAS_API_KEY"]
    else:
        st.error("CEREBRAS_API_KEY not found in secrets.")
        st.stop()

    model_name = st.text_input("Model Name", value=DEFAULT_MODEL)

    st.divider()
    st.subheader("Sampling Parameters")

    t_col, t_reset = st.columns([4, 1])
    with t_col:
        temp = st.slider(
            "Temperature",
            0.0,
            1.5,
            st.session_state.temp_val,
            0.1,
            help="Higher = More Creative/Random",
            key="temp_slider",
        )
    with t_reset:
        st.write("")  # spacer
        if st.button("↺", key="reset_temp", help="Reset to default (1.0)"):
            st.session_state.temp_val = DEFAULT_TEMP
            st.rerun()
    st.session_state.temp_val = temp

    p_col, p_reset = st.columns([4, 1])
    with p_col:
        top_p = st.slider(
            "Top-P",
            0.0,
            1.0,
            st.session_state.top_p_val,
            0.05,
            help="Nucleus Sampling cutoff — filters probability table and CSV",
            key="top_p_slider",
        )
    with p_reset:
        st.write("")
        if st.button("↺", key="reset_top_p", help="Reset to default (1.0)"):
            st.session_state.top_p_val = DEFAULT_TOP_P
            st.session_state.current_analysis = None
            st.rerun()
    if top_p != st.session_state.top_p_val:
        st.session_state.current_analysis = None
    st.session_state.top_p_val = top_p

    k_col, k_reset = st.columns([4, 1])
    with k_col:
        top_k = st.slider(
            "Top-K",
            0,
            100,
            st.session_state.top_k_val,
            5,
            help="Limit to top K tokens — filters probability table and CSV (0 = disabled)",
            key="top_k_slider",
        )
    with k_reset:
        st.write("")
        if st.button("↺", key="reset_top_k", help="Reset to default (0)"):
            st.session_state.top_k_val = DEFAULT_TOP_K
            st.session_state.current_analysis = None
            st.rerun()
    if top_k != st.session_state.top_k_val:
        st.session_state.current_analysis = None
    st.session_state.top_k_val = top_k

    st.divider()
    st.subheader("Display & Mode")
    st.session_state.word_mode = st.toggle(
        "Word Mode",
        value=st.session_state.word_mode,
        help="Group sub-tokens into full words. Prevents fragments like 'Ag' + 'rees'.",
    )

    st.divider()

    if st.session_state.history:
        csv_data = convert_history_to_csv()
        st.download_button(
            "⬇️ Download Data (CSV)", csv_data, "cerebras_probabilities.csv", "text/csv"
        )
    else:
        st.button("⬇️ Download Data (CSV)", disabled=True)

    st.divider()
    st.subheader("🌳 Tree Explorer")
    tree_depth = st.slider(
        "Tree Depth",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="How many generations deep to explore. Warning: grows exponentially with candidates × depth.",
    )

    tree_status = st.empty()

    if st.session_state.tree_csv is not None:
        st.download_button(
            "⬇️ Download Tree CSV",
            st.session_state.tree_csv,
            "probability_tree.csv",
            "text/csv",
            key="tree_dl",
        )
        if st.button("🗑️ Clear Tree", key="clear_tree"):
            st.session_state.tree_csv = None
            st.rerun()
    else:
        if st.button(
            "🌳 Explore Tree",
            key="run_tree",
            disabled=not st.session_state.context_locked,
        ):
            with st.spinner("Building probability tree..."):
                rows = explore_tree(
                    api_key,
                    model_name,
                    temp,
                    top_p,
                    top_k,
                    tree_depth,
                    st.session_state.word_mode,
                    tree_status,
                )
            if rows:
                st.session_state.tree_csv = tree_rows_to_csv(rows)
                tree_status.success(
                    f"✅ Tree complete! {len(rows)} rows across {tree_depth} depth(s)."
                )
                st.rerun()
            else:
                tree_status.warning(
                    "No rows generated. Check your filters or API connection."
                )

    st.divider()
    if st.button("🔄 Reset / Start Over"):
        st.session_state.history = []
        st.session_state.current_analysis = None
        st.session_state.context_locked = False
        st.rerun()

    st.divider()
    st.markdown("### 🔧 Debug Tools")

    with st.expander("View Raw History (repr)"):
        st.code(str([t["token"] for t in st.session_state.history]))

# --- Main UI ---

st.title("Probability Explorer v2")

# 1. Setup Phase
if not st.session_state.context_locked:
    st.info("Configure your prompt below to begin the generation session.")
    st.session_state.system_prompt = st.text_area(
        "System Prompt", st.session_state.system_prompt
    )
    st.session_state.user_prompt = st.text_area(
        "User Prompt", st.session_state.user_prompt
    )

    if st.button("🚀 Lock & Start", type="primary"):
        st.session_state.context_locked = True
        st.rerun()

# 2. Interactive Phase
else:
    # --- A. Conversation Display ---
    st.markdown("### 📝 Conversation History")

    # User Message Bubble
    with st.chat_message("user"):
        st.write(st.session_state.user_prompt)

    # Assistant Message Bubble (Constructed from tokens)
    if st.session_state.history:
        full_response = "".join([item["token"] for item in st.session_state.history])
        with st.chat_message("assistant"):
            st.markdown(f"{full_response}▌")  # Added a cursor effect
    else:
        with st.chat_message("assistant"):
            st.caption("Waiting for first token...")

    # Undo Button (Small, below chat)
    if st.session_state.history:
        if st.button("↩️ Undo Last Step", help="Remove the last generated token"):
            undo_last_step()

    st.divider()

    # --- B. Analysis Engine ---
    st.markdown("### 🔍 Next Token Analysis")

    # Auto-Fetch Analysis if needed
    if st.session_state.current_analysis is None:
        with st.spinner("Calculating probabilities..."):
            analyze_next_step(api_key, model_name, temp, top_p, top_k)

            st.rerun()

    # Debug: View Prompt
    with st.expander("🛠️ Debug: Current Prompt Context"):
        st.text(build_full_prompt_debug())

    # Visualize Data
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        candidates = analysis["candidates"]

        # Prepare DataFrame
        data = []
        dropdown_options = []
        for c in candidates:
            prob_pct = math.exp(c.logprob) * 100
            data.append(
                {"Token": c.token, "Probability (%)": prob_pct, "LogProb": c.logprob}
            )
            dropdown_options.append(c.token)

        df = pd.DataFrame(data)

        # Build chart title showing active filters
        active_filters = []
        stored_top_p = analysis.get("top_p", 1.0)
        stored_top_k = analysis.get("top_k", 0)
        if stored_top_p < 1.0:
            active_filters.append(f"top-p={stored_top_p}")
        if stored_top_k and stored_top_k > 0:
            active_filters.append(f"top-k={stored_top_k}")
        filter_label = f" [{', '.join(active_filters)}]" if active_filters else ""
        n_shown = min(10, len(df))
        chart_title = f"Top {n_shown} Probabilities (Next Step){filter_label}"

        # 2-Column Layout
        col_viz, col_ctrl = st.columns([3, 2], gap="large")

        with col_viz:
            # 1. Bar Chart
            fig = px.bar(
                df.head(10),
                x="Probability (%)",
                y="Token",
                orientation="h",
                title=chart_title,
                text_auto=".1f",
                color="Probability (%)",
                color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # 2. Data Table (Restored Feature)
            with st.expander("📊 View Raw Data Table", expanded=False):
                st.dataframe(df, use_container_width=True)

        with col_ctrl:
            st.subheader("🕹️ Controls")

            # 1. Step (Default)
            model_pick = analysis["predicted_token"]
            st.success(f"**Model Recommendation:** `{model_pick}`")
            if st.button(
                f"▶️ Step (Accept `{model_pick}`)",
                type="primary",
                use_container_width=True,
            ):
                commit_step(model_pick)

            st.markdown("---")

            # 2. Manual Select (Dropdown)
            st.caption(
                f"Option A: Choose from {len(dropdown_options)} candidates{filter_label}"
            )
            manual_choice = st.selectbox(
                "Select token:", options=dropdown_options, label_visibility="collapsed"
            )
            if st.button("Commit Selection", use_container_width=True):
                commit_step(manual_choice)

            # 3. Custom Token (New Feature)
            st.caption("Option B: Inject Custom Token")
            col_cust_input, col_cust_btn = st.columns([2, 1])
            with col_cust_input:
                custom_token_input = st.text_input(
                    "Custom",
                    label_visibility="collapsed",
                    placeholder="Type word...",
                    key=f"custom_token_{st.session_state.custom_token_key}",
                )
                smart_space = st.checkbox(
                    "Prepend with space",
                    value=False,
                    help="Manually add a leading space to your custom token.",
                )
            with col_cust_btn:
                if st.button("Inject", use_container_width=True):
                    if custom_token_input:
                        final_token = custom_token_input

                        # Smart Space Logic
                        if smart_space:
                            final_token = " " + final_token

                        st.session_state.custom_token_key += 1  # clears the text box
                        commit_step(final_token)

            st.markdown("---")

            # 4. Fast Forward
            st.caption("⏩ Fast Forward")
            ff_count = st.number_input(
                "Count",
                min_value=1,
                max_value=200,
                value=20,
                label_visibility="collapsed",
            )
            if st.button(f"Generate Next {ff_count} Tokens", use_container_width=True):
                with st.spinner("Fast forwarding..."):
                    fast_forward(api_key, model_name, temp, top_p, top_k, ff_count)
