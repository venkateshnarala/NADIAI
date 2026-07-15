"""
nadi_chatbot.py
NADI AI - Hydrological Analysis Chatbot
------------------------------------------
An AI interpreter (NOT a computation engine) that answers natural-language
questions about a station's hydrological analysis results.

All numbers come from `nadi_report.build_analysis_summary(station_data)`,
which itself simply calls the existing, already-tested computation modules
(nadi_quality, nadi_statisticaltests, nadi_distfit, nadi_ratingcurve,
nadi_data_collec) and packages their outputs into a plain JSON-serializable
dict. This module performs NO hydrological calculations of its own -- it
only formats that dict as grounding context for an LLM call and relays the
model's natural-language answer back to the user.

Model: OpenRouter, google/gemma-3-27b-it:free (configurable below).
"""

import json
import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
REQUEST_TIMEOUT_SECONDS = 45
MAX_HISTORY_TURNS = 8  # how many past user/assistant turns to resend for context

SYSTEM_PROMPT = """You are the NADI AI Assistant, an interpreter for hydrological analysis results.

STRICT RULES:
1. You must answer ONLY using the ANALYSIS_DATA JSON provided below. Do not use outside
   hydrology knowledge to invent numbers, station facts, or statistics that are not in
   ANALYSIS_DATA.
2. You NEVER perform calculations, unit conversions, interpolation, or re-derive statistics
   yourself. If the exact number the user wants is not present in ANALYSIS_DATA, say so
   plainly instead of estimating or computing it.
3. You may explain, summarize, compare, and interpret the numbers already present in
   ANALYSIS_DATA in plain language (e.g. explaining why a distribution ranked first based on
   its listed KS/AIC/RMSE/R2 values, or comparing two return-period discharge values that are
   both present in design_flood_magnitudes).
4. You may also provide broader, narrative-style responses when asked -- such as an overall
   summary, a plain-language explanation, a discussion of limitations, a suggested thesis/report
   discussion or conclusion paragraph, or an opinion on which result is most important. These
   are still interpretations, not new findings: every claim in them must be traceable to a
   number, flag, or result already present in ANALYSIS_DATA. Do not introduce outside facts,
   outside literature, or invented figures even in a summary or conclusion.
5. When asked for limitations, base them only on what ANALYSIS_DATA itself shows (e.g. short
   record length, insufficient years for full analysis, missing rating curve, non-significant
   trend tests, wide spread between top-ranked distributions, etc.) -- not on generic textbook
   caveats that aren't reflected in this station's actual results.
6. If a section of ANALYSIS_DATA is empty, missing, or marked unavailable (e.g. rating curve
   not available, insufficient data for trend tests), clearly tell the user that this
   information was not available or not computed for this station, rather than guessing.
7. Keep answers concise, clear, and grounded -- cite the specific numbers you used from
   ANALYSIS_DATA when relevant. For summary/conclusion-style answers, a short paragraph (or a
   few tight bullet points) is preferred over a long essay unless the user asks for more detail.
8. You are not a licensed hydrologist. For any question implying an engineering, planning,
   or design decision, remind the user to verify results with a qualified professional before
   practical use.

ANALYSIS_DATA (JSON):
{analysis_json}
"""


# ---------------------------------------------------------------------------
# ERROR TYPES
# ---------------------------------------------------------------------------

class ChatbotError(Exception):
    """Base class for user-facing chatbot errors (message is already friendly)."""
    pass


class MissingAnalysisError(ChatbotError):
    pass


class InvalidAPIKeyError(ChatbotError):
    pass


class APITimeoutError(ChatbotError):
    pass


class ConnectionFailedError(ChatbotError):
    pass


class EmptyResponseError(ChatbotError):
    pass


# ---------------------------------------------------------------------------
# CORE OPENROUTER CALL
# ---------------------------------------------------------------------------

def _build_messages(analysis_summary, chat_history, user_question):
    """
    Build the OpenRouter messages list: one system message carrying the
    analysis JSON + rules, followed by a trimmed slice of prior chat turns,
    followed by the new user question.
    """
    try:
        analysis_json = json.dumps(analysis_summary, indent=2, default=str)
    except Exception:
        analysis_json = "{}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(analysis_json=analysis_json)}
    ]

    trimmed = chat_history[-(MAX_HISTORY_TURNS * 2):] if chat_history else []
    for turn in trimmed:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_question})
    return messages


def ask_chatbot(analysis_summary, chat_history, user_question, api_key,
                 model=OPENROUTER_MODEL, stream=True):
    """
    Send a question to OpenRouter, grounded strictly in analysis_summary.

    Parameters
    ----------
    analysis_summary : dict
        Output of nadi_report.build_analysis_summary(station_data). Must not
        be None/empty -- raises MissingAnalysisError otherwise.
    chat_history : list[dict]
        Prior turns as [{"role": "user"/"assistant", "content": "..."}].
    user_question : str
    api_key : str
        OpenRouter API key.
    model : str
    stream : bool
        If True, returns a generator yielding text chunks. If False, returns
        the full answer string in one go.

    Raises
    ------
    MissingAnalysisError, InvalidAPIKeyError, APITimeoutError,
    ConnectionFailedError, EmptyResponseError
    """
    if not analysis_summary:
        raise MissingAnalysisError(
            "No analysis results are available yet. Please run the "
            "hydrological analysis for this station first, then ask "
            "your question."
        )
    if not api_key:
        raise InvalidAPIKeyError(
            "No OpenRouter API key is configured. Please add a valid API "
            "key to use the chatbot."
        )
    if not user_question or not user_question.strip():
        raise EmptyResponseError("Please type a question to ask the assistant.")

    messages = _build_messages(analysis_summary, chat_history, user_question)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nadi-ai.streamlit.app",
        "X-Title": "NADI AI",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS, stream=stream,
        )
    except requests.exceptions.Timeout:
        raise APITimeoutError(
            "The request to the AI service timed out. Please try again "
            "in a moment."
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionFailedError(
            "Could not connect to the AI service. Please check your "
            "internet connection and try again."
        )
    except requests.exceptions.RequestException as e:
        raise ConnectionFailedError(f"Connection to the AI service failed: {e}")

    if response.status_code == 401:
        raise InvalidAPIKeyError(
            "The OpenRouter API key was rejected (invalid or expired). "
            "Please check the configured API key."
        )
    if response.status_code == 408 or response.status_code == 504:
        raise APITimeoutError(
            "The AI service took too long to respond. Please try again."
        )
    if response.status_code == 429:
        raise ConnectionFailedError(
            "The AI service is rate-limiting requests right now (free-tier "
            "limit reached). Please wait a moment and try again."
        )
    if response.status_code >= 400:
        try:
            err_body = response.json()
            err_msg = err_body.get("error", {}).get("message", response.text[:300])
        except Exception:
            err_msg = response.text[:300]
        raise ConnectionFailedError(f"The AI service returned an error: {err_msg}")

    if stream:
        return _stream_response(response)
    else:
        return _parse_full_response(response)


def _parse_full_response(response):
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise EmptyResponseError(
            "The AI service returned an unreadable response. Please try again."
        )
    if not content or not content.strip():
        raise EmptyResponseError(
            "The AI service returned an empty response. Please try rephrasing "
            "your question or try again."
        )
    return content


def _stream_response(response):
    """
    Generator that yields text chunks from an OpenRouter SSE stream.
    Raises EmptyResponseError at the end if nothing was ever yielded.
    """
    got_any_content = False
    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                got_any_content = True
                yield piece
    except requests.exceptions.ChunkedEncodingError:
        raise ConnectionFailedError(
            "The connection to the AI service was interrupted mid-response. "
            "Please try again."
        )
    except requests.exceptions.RequestException as e:
        raise ConnectionFailedError(f"Connection to the AI service failed: {e}")

    if not got_any_content:
        raise EmptyResponseError(
            "The AI service returned an empty response. Please try rephrasing "
            "your question or try again."
        )


# ---------------------------------------------------------------------------
# STREAMLIT CHAT UI
# ---------------------------------------------------------------------------

SUGGESTED_QUESTIONS = [
    "Summarize the complete station analysis.",
    "What is the maximum annual flow?",
    "Were any outliers detected?",
    "Explain the Mann-Kendall trend results.",
    "Which probability distribution fits best, and why?",
    "Compare the 25-year and 100-year design flood.",
    "Which fitting method should I trust?",
    "Is this station suitable for flood frequency analysis?",
    "What are the limitations of this analysis?",
    "Which result is the most important for this station?",
    "Write a conclusion paragraph for this station.",
    "What should I include in my thesis discussion for this station?",
    "Explain these results in simple language.",
    "Summarize the findings in one paragraph.",
]


def render_chatbot_ui(analysis_summary, api_key, station_label="this station"):
    """
    Render a ChatGPT-style chat interface in Streamlit, grounded in
    analysis_summary. Call this after the hydrological analysis has been
    run for the selected station.
    """
    st.markdown("### \U0001F4AC Ask NADI AI About This Analysis")

    if not analysis_summary:
        st.info("⚠️ Run the analysis to enable the NADI AI chatbot.")
        return

    st.caption(
    "📝 **Note:** NADI AI answers only questions related to the above analysis results."
        )
    history_key = "nadi_chat_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []


    # Render existing history
    for turn in st.session_state[history_key]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


    user_question = st.chat_input("Ask about the analysis results...")
    question_to_ask = user_question

    if question_to_ask:
        st.session_state[history_key].append({"role": "user", "content": question_to_ask})
        with st.chat_message("user"):
            st.markdown(question_to_ask)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            try:
                if not api_key:
                    raise InvalidAPIKeyError(
                        "No OpenRouter API key is configured. Please add a "
                        "valid API key to use the chatbot."
                    )
                stream = ask_chatbot(
                    analysis_summary,
                    st.session_state[history_key][:-1],  # history excluding the just-added question
                    question_to_ask,
                    api_key,
                    stream=True,
                )
                for chunk in stream:
                    full_answer += chunk
                    placeholder.markdown(full_answer + "\u258c")
                placeholder.markdown(full_answer)

            except MissingAnalysisError as e:
                placeholder.error(f"\u26A0\uFE0F {e}")
                full_answer = None
            except InvalidAPIKeyError as e:
                placeholder.error(f"\U0001F511 {e}")
                full_answer = None
            except APITimeoutError as e:
                placeholder.error(f"\u23F1\uFE0F {e}")
                full_answer = None
            except ConnectionFailedError as e:
                placeholder.error(f"\U0001F50C {e}")
                full_answer = None
            except EmptyResponseError as e:
                placeholder.error(f"\U0001F4ED {e}")
                full_answer = None
            except ChatbotError as e:
                placeholder.error(f"\u26A0\uFE0F {e}")
                full_answer = None
            except Exception as e:
                placeholder.error(
                    f"\u26A0\uFE0F Something went wrong while contacting the AI "
                    f"service: {e}"
                )
                full_answer = None

        if full_answer:
            st.session_state[history_key].append({"role": "assistant", "content": full_answer})
        else:
            # Remove the user question that couldn't be answered so the
            # history doesn't get out of sync with what was actually shown.
            if st.session_state[history_key] and st.session_state[history_key][-1]["role"] == "user":
                st.session_state[history_key].pop()

    if st.session_state[history_key]:
        if st.button("\U0001F5D1\uFE0F Clear chat", key="clear_nadi_chat"):
            st.session_state[history_key] = []
            st.rerun()