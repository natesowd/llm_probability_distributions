# Word Mode Greedy Lookahead Rework

## Context

The "word mode" in `remote-analyzer.py` has a fundamental flaw: when the first sub-token is a word fragment (e.g., `"Ag"`), all 20 alternative candidates are completed using the same greedy suffix from the top-1 path (e.g., `"rees"`). This produces nonsensical completions like `"Unrees"` for the fragment `"Un"`, since every fragment gets the suffix that was only correct for the greedy pick. The probabilities shown are also misleading — they represent only the first sub-token, not the full word.

The goal is to rework word mode so that each fragment candidate is completed with its own greedy lookahead — iteratively fetching the next most-likely token until a word boundary is reached.

## File to modify

- `/home/user/llm_probability_distributions/remote-analyzer.py` (sole file, ~795 lines)

## Plan

### Step 1: Promote `TokenProb` to module-level class

`TokenProb` is currently defined identically inside three functions (lines 133, 214, 319). Move it to module level (after `is_continuation`, ~line 57) with a `__repr__` for debugging. Remove the three inline definitions.

### Step 2: Keep `word_mode` as a boolean toggle

The toggle stays as-is (on/off). When on, word mode now uses per-fragment greedy lookahead instead of the broken shared-suffix approach. Default remains `True`.

No UI changes to the toggle itself. Add a migration guard in case session state has an unexpected value:

```python
if "word_mode" not in st.session_state:
    st.session_state.word_mode = True
```

(This already exists at line 25–26, so no change needed here.)

### Step 3: Create `_complete_fragment()` helper

New function (~line 82 area) that completes a single fragment token to a full word in one API call:

- **Input:** `client`, `model`, `base_prompt` (full prompt up to the decision point), `fragment_token` (e.g., `"Ag"`)
- **Algorithm:**
  1. Build prompt: `base_prompt + fragment_token`
  2. Call API with `max_tokens=5`, `temperature=0` (deterministic greedy), `logprobs=1` (to access `choice.logprobs.tokens`)
  3. Scan the returned tokens for the word boundary: iterate through returned tokens, collecting continuation tokens until we hit one that starts with space/newline/tab (i.e., not `is_continuation(token)`)
  4. Join the collected continuation tokens into a suffix
  5. Return `fragment_token + suffix`
- **Returns:** The completed word string (e.g., `"Agreement"`)
- **Returns** `fragment_token` unchanged on API error
- Single API call per fragment — `max_tokens=5` is enough since most words are 1–3 sub-tokens after the first fragment

This is the key insight: each fragment gets its own greedy completion via one API call, scanning the returned tokens for the word boundary. The probability used is still the original first sub-token's logprob.

### Step 4: Create `build_word_candidates()` shared helper

New function (~line 82 area) that replaces the duplicated word-mode logic in both `analyze_next_step` and `get_candidates_for_prompt`:

- **Input:** `tokens` (greedy path from initial API call), `top_logprobs_list`, `client`, `model`, `base_prompt`
- **Returns:** `(predicted_token, candidates_list)`

**Logic:**

1. Build `predicted_token` by following greedy path tokens until word boundary (same as current lines 144–149):

   ```python
   word_tokens = []
   for i, t in enumerate(tokens):
       if i > 0 and not is_continuation(t):
           break
       word_tokens.append(t)
   predicted_token = "".join(word_tokens)
   ```

2. Build candidates from `top_logprobs_list[0]`:
   - For each `(token, logprob)` in the top-20 dict:
     - If not `is_continuation(token)`: already a full word (has leading space), use as-is
     - If `is_continuation(token)`: call `_complete_fragment(client, model, base_prompt, token)` to get the full word
   - Create `TokenProb(completed_word, logprob)` for each

### Step 5: Refactor `analyze_next_step()` (lines 109–186)

Replace the inline word-mode block (lines 138–167) with a call to `build_word_candidates()`:

```python
if st.session_state.word_mode and choice.logprobs:
    tokens = choice.logprobs.tokens
    top_logprobs_list = choice.logprobs.top_logprobs
    predicted_token, candidates = build_word_candidates(
        tokens, top_logprobs_list, client, model, full_prompt
    )
else:
    predicted_token = choice.text
    top_dict = choice.logprobs.top_logprobs[0]
    candidates = [TokenProb(t, lp) for t, lp in top_dict.items()]
```

The `max_tokens=10` for the initial call stays the same (needed to find the greedy word boundary for the predicted token).

### Step 6: Refactor `get_candidates_for_prompt()` (lines 313–369)

Same refactor as Step 5. Replace lines 335–361 with a call to `build_word_candidates()`. This function already receives `client` and `model`, and has `prompt` available as the base.

### Step 7: Update `fast_forward()` (lines 189–266)

`fast_forward` processes an already-generated token sequence by grouping sub-tokens into words. Its word-mode logic is fundamentally different — it doesn't need per-fragment completion because it already has the actual greedy continuation.

**Changes:**

- Keep the existing grouping logic (lines 231–246) as-is — it correctly joins sequential continuation tokens into words
- The candidates stored for each word still come from the first sub-token position only. This is acceptable for fast-forward since it's about speed, and the greedy path already produces correct words.

No functional change here, just ensure it still works correctly with the refactored `TokenProb` class.

### Step 8: Update `explore_tree()` and callers

- `explore_tree` passes `word_mode` to `get_candidates_for_prompt()`, which is already refactored in Step 6 — no changes needed.
- Add a note/warning in the tree explorer UI when word mode is on, since each tree node may trigger additional API calls for fragment completion.

### Step 9: Add probability semantics caption

Near the chart display (~line 704), when word mode is on, add a caption:

> *"Word mode: fragment tokens completed via greedy lookahead. Probabilities reflect the first sub-token only."*

## Summary of changes by function

| Function | Change |
|---|---|
| `TokenProb` (new, module-level) | Consolidate 3 inline definitions |
| `_complete_fragment` (new) | Single-call per-fragment greedy completion (`max_tokens=5`) |
| `build_word_candidates` (new) | Shared word-mode logic replacing duplicated code |
| `analyze_next_step` | Replace inline word-mode block with `build_word_candidates` call |
| `get_candidates_for_prompt` | Same refactor |
| `fast_forward` | No functional change; uses module-level `TokenProb` |
| `explore_tree` | Add word-mode API cost warning |
| Sidebar UI | Add probability semantics caption when word mode is on |

## Verification

1. **Word mode off:** Run the app, disable word mode, click Step — should show raw sub-tokens (same as current token mode)
2. **Word mode on:** Enable word mode, click Step — fragment candidates should each have their own meaningful completion (e.g., `"Ag"` → `"Agreement"`, `"Un"` → `"Unified"`), NOT all sharing the same suffix
3. **Full-word tokens:** Candidates that are already full words (start with space) should appear unchanged
4. **Fast Forward:** Word mode should produce correctly grouped word history
5. **Tree Explorer:** Should work with word mode; note about API cost should appear
6. **CSV Export:** Download CSV and verify token/candidate data is correct
7. **Edge cases:** Test when API lookahead fails (fragment should display as-is); verify that if all 5 returned tokens are continuations, the word still gets capped at whatever was returned
