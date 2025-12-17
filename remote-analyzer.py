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

# --- Helper Functions ---


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
    Runs inference for 1 token to 'peek' at the probabilities.
    Uses Text Completion API to ensure proper continuation.
    """
    client = get_client(api_key)
    full_prompt = build_llama_prompt()

    try:
        response = client.completions.create(
            model=model,
            prompt=full_prompt,
            max_tokens=1,
            logprobs=20,  # In Completions API, this is an int
            temperature=temp,
            top_p=top_p,
            # extra_body={"top_k": top_k},
        )

        choice = response.choices[0]
        predicted_token = choice.text
        # In Completions, logprobs is a wrapper object, top_logprobs is a list of dicts
        # We want the first position's top logprobs
        top_dict = choice.logprobs.top_logprobs[0]

        # Convert dict to expected object structure (or list of objects) for consistency?
        # The existing UI expects objects with .token and .logprob attributes
        # top_dict is {token: logprob, ...}

        # Correction: check OpenAI python lib structure for Completions.
        # It usually returns dict {token: logprob}.
        # But UI expects object with .token and .logprob.
        # Let's create a simple shim class.

        class TokenProb:
            def __init__(self, token, logprob):
                self.token = token
                self.logprob = logprob

        candidates = []
        for t, lp in top_dict.items():
            candidates.append(TokenProb(t, lp))

        # Sort by logprob desc
        candidates.sort(key=lambda x: x.logprob, reverse=True)

        st.session_state.current_analysis = {
            "predicted_token": predicted_token,
            "candidates": candidates,
        }

    except Exception as e:
        st.error(f"Error calling Cerebras API: {e}")


def fast_forward(api_key, model, temp, top_p, top_k, num_tokens):
    full_prompt = build_llama_prompt()
    client = get_client(api_key)
    try:
        response = client.completions.create(
            model=model,
            prompt=full_prompt,
            max_tokens=num_tokens,
            logprobs=5,  # Minimal logprobs for history
            temperature=temp,
            top_p=top_p,
            # extra_body={"top_k": top_k},
        )

        if response.choices[0].logprobs:
            # Complicated parsing for completion logprobs
            # response.choices[0].logprobs.top_logprobs is a list of dicts {token: lp}
            # response.choices[0].logprobs.tokens is a list of chosen tokens

            c = response.choices[0]
            tokens = c.logprobs.tokens
            top_logprobs_list = c.logprobs.top_logprobs

            # Helper class shim
            class TokenProb:
                def __init__(self, token, logprob):
                    self.token = token
                    self.logprob = logprob

            for i, token_str in enumerate(tokens):
                # top_logprobs_list[i] is a dict {token: lp}
                # We need to convert it to a list of TokenProb objects
                current_cands = []
                if top_logprobs_list and i < len(top_logprobs_list):
                    for t, lp in top_logprobs_list[i].items():
                        current_cands.append(TokenProb(t, lp))
                    current_cands.sort(key=lambda x: x.logprob, reverse=True)

                st.session_state.history.append(
                    {"token": token_str, "candidates": current_cands}
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
                    "Probability": prob,
                    "Log_Probability": cand.logprob,
                    "Is_Actual_Choice": (cand.token == chosen),
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("Cerebras API Key", type="password")
    model_name = st.text_input("Model Name", value=DEFAULT_MODEL)

    st.divider()
    st.subheader("Sampling Parameters")

    temp = st.slider(
        "Temperature", 0.0, 1.5, 1.0, 0.1, help="Higher = More Creative/Random"
    )
    top_p = st.slider("Top-P", 0.0, 1.0, 1.0, 0.05, help="Nucleus Sampling cutoff")
    top_k = st.slider(
        "Top-K",
        0,
        100,
        0,
        5,
        help="Limit to top K tokens (Currently unsupported for Cerebras)",
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
        if not api_key:
            st.error("Please enter an API Key in the sidebar.")
        else:
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

    if not api_key:
        st.error("API Key missing.")
        st.stop()

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

        # 2-Column Layout
        col_viz, col_ctrl = st.columns([3, 2], gap="large")

        with col_viz:
            # 1. Bar Chart
            fig = px.bar(
                df.head(10),
                x="Probability (%)",
                y="Token",
                orientation="h",
                title="Top 10 Probabilities (Next Step)",
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
            st.caption("Option A: Choose from Top-20")
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
                    "Custom", label_visibility="collapsed", placeholder="Type word..."
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
