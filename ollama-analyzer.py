import streamlit as st
import requests
import pandas as pd
import math
import plotly.express as px

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "llama3.1:8b" # Your installed model
MODEL_NAME = "llama3.2:3b"  # Your installed model
MAX_LOGPROBS = 20  # Number of top tokens to display


# --- API Function ---
def get_ollama_logprobs(prompt, model, temp, top_p, top_k):
    """Calls the Ollama API to get log probabilities for the next token."""

    # Ollama requires these options to return logprobs for the next token
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Wait for full response
        "logprobs": True,  # Request log probabilities
        "top_logprobs": MAX_LOGPROBS,  # Request the top N candidates
        "options": {
            "num_predict": 1,  # Only predict one token (the next one)
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "seed": st.session_state.get(
                "seed", 0
            ),  # Use a fixed seed for reproducibility
        },
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        # Check if logprobs data is present and valid
        if (
            "logprobs" in data
            and data["logprobs"]
            and "top_logprobs" in data["logprobs"][0]
        ):
            return data["logprobs"][0]["top_logprobs"]
        else:
            st.error(
                "Ollama response did not contain top log probabilities. Ensure Ollama is running and supports logprobs (v0.12.11+)."
            )
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        st.caption("Ensure Ollama server is running on http://localhost:11434")
        return None


# --- Visualization & Data Processing ---
def process_and_visualize(top_logprobs):
    """Converts logprobs to percentages, calculates nucleus/top-k cutoffs, and plots."""

    # 1. Convert Logprobs to Probabilities (Exponentiate)
    df = pd.DataFrame(top_logprobs)
    df["probability"] = df["logprob"].apply(math.exp)

    # 2. Re-normalize to 100% over the displayed tokens
    # This chart shows the *share* of probability mass among the top MAX_LOGPROBS tokens
    total_prob = df["probability"].sum()
    df["percentage"] = (df["probability"] / total_prob) * 100

    # 3. Clean up the token text for display
    # Tokens often have leading spaces (represented by ' ') or are sub-words
    df["token"] = df["token"].str.replace(" ", " ", regex=False).str.strip()

    # 4. Calculate cumulative probability for Top-P (Nucleus) boundary
    df["cumulative_prob"] = df["probability"].cumsum()
    top_p_cutoff = st.session_state["top_p"] * total_prob
    df["in_nucleus"] = df["cumulative_prob"] <= top_p_cutoff

    # Add a marker for the Top-K cutoff
    top_k_cutoff = st.session_state["top_k"]
    df["in_top_k"] = df.index < top_k_cutoff

    # Determine the token that will be sampled based on current parameters (simplified)
    # The actual sampling is more complex, but we can highlight the candidates.
    df["color_group"] = "Outside Sampling Pool"

    # Apply Top-K filter first (index-based)
    df.loc[df.index < top_k_cutoff, "color_group"] = "In Top-K"

    # Apply Top-P filter second (dynamic probability-based)
    # The true pool is the intersection of T and the *active* sampler (often P)
    if st.session_state["top_p"] < 1.0:
        # If Top-P is active, we highlight the Nucleus candidates
        df.loc[df["in_nucleus"], "color_group"] = "In Nucleus (Top-P)"

    # --- Plotting ---
    fig = px.bar(
        df.head(MAX_LOGPROBS),
        x="percentage",
        y="token",
        orientation="h",
        color="color_group",
        color_discrete_map={
            "In Nucleus (Top-P)": "#1f77b4",  # Blue
            "In Top-K": "#ff7f0e",  # Orange
            "Outside Sampling Pool": "#cccccc",  # Gray
        },
        title=f"Top {MAX_LOGPROBS} Next-Token Probability Distribution",
        labels={"token": "Token", "percentage": "Probability Share (%)"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- Data Table ---
    df_display = df[
        ["token", "probability", "percentage", "cumulative_prob", "color_group"]
    ]
    df_display.columns = [
        "Token",
        "Raw Probability",
        "Share (%)",
        "Cumulative Share",
        "Sampling Pool",
    ]
    st.markdown("### 📊 Raw Log Probability Data")
    st.dataframe(df_display, use_container_width=True)


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Ollama LLM Probability Analyzer")

st.title("🔬 Ollama LLM Next-Token Analyzer")
st.markdown(
    "Visualize the next-token probability distribution and see how sampling parameters (`Temperature`, `Top-P`, `Top-K`) affect the available candidates for your **`llama3.2`** model."
)

# Initialize session state for parameters
if "temp" not in st.session_state:
    st.session_state["temp"] = 0.7
if "top_p" not in st.session_state:
    st.session_state["top_p"] = 0.9
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 40
if "seed" not in st.session_state:
    st.session_state["seed"] = 1234


# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Sampling Controls")

    # Model Selector (Fixed for this exercise)
    st.text_input("Ollama Model", value=MODEL_NAME, disabled=True)

    # Temperature Slider
    st.session_state["temp"] = st.slider(
        "Temperature (T)",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state["temp"],
        step=0.05,
        help="Higher values flatten the distribution (more random).",
    )

    # Top-P Slider
    st.session_state["top_p"] = st.slider(
        "Top-P (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["top_p"],
        step=0.01,
        help="Cumulative probability cutoff. Limits selection to tokens that make up this much probability mass.",
    )

    # Top-K Slider
    st.session_state["top_k"] = st.slider(
        "Top-K Sampling",
        min_value=0,
        max_value=MAX_LOGPROBS,
        value=st.session_state["top_k"],
        step=1,
        help=f"Limits selection to the K most likely tokens (Max: {MAX_LOGPROBS}). 0 disables it.",
    )

    # Seed Input
    st.session_state["seed"] = st.number_input(
        "Random Seed",
        min_value=0,
        value=st.session_state["seed"],
        step=1,
        help="Use a fixed seed to ensure the model's logits are reproducible.",
    )

    st.markdown("---")
    st.markdown("### 🔑 Key Parameters")
    st.markdown(f"* **Model:** `{MODEL_NAME}`")
    st.markdown(f"* **Logprobs Displayed:** `{MAX_LOGPROBS}`")
    st.markdown(f"* **Ollama URL:** `{OLLAMA_API_URL}`")


# --- Main Content Area ---
st.markdown("### 1. Enter Context (The Start of the Sentence)")
context_prompt = st.text_area(
    "Context/Prefix",
    "The secret to a great fantasy novel is to establish three core things: The",
    height=100,
)

# Button to trigger analysis
if st.button("🔍 Analyze Next-Token Distribution", type="primary"):
    if not context_prompt:
        st.warning("Please enter some text for the model to continue.")
    else:
        with st.spinner(f"Requesting top {MAX_LOGPROBS} logprobs from {MODEL_NAME}..."):
            logprobs_data = get_ollama_logprobs(
                context_prompt,
                MODEL_NAME,
                st.session_state["temp"],
                st.session_state["top_p"],
                st.session_state["top_k"],
            )

            if logprobs_data:
                st.markdown("---")
                st.markdown("### 2. Resulting Probability Distribution")
                process_and_visualize(logprobs_data)

                # Show the sampled token for context
                # NOTE: This only shows the very next token, which may be sampled!
                full_response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": MODEL_NAME,
                        "prompt": context_prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 1,
                            "temperature": st.session_state["temp"],
                            "top_p": st.session_state["top_p"],
                            "top_k": st.session_state["top_k"],
                            "seed": st.session_state["seed"],
                        },
                    },
                ).json()

                sampled_token = full_response["response"]
                st.info(
                    f"The token **actually sampled** (after applying T/P/K) was: **`{sampled_token}`**"
                )
