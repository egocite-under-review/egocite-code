"""
EpisodicAgent — tool-use retrieval agent over the episodic memory DAG.

Architecture (tool-use path):
  1. Retrieval agent  — Claude native tool-use loop.  The LLM calls
                        search_action_speech / search_activity / search_conversation
                        to gather evidence, curate_evidence to prune, then
                        answer() to end the loop.
  2. Curation agent   — separate LLM call (with thinking) invoked when the
                        deduplicated evidence pool exceeds
                        _CURATE_EVIDENCE_THRESHOLD.  Only the curate_evidence
                        tool is exposed; the model selects ≤15 items.
  3. Answering agent  — final LLM call (no tools).  Receives the raw 30-sec
                        captions expanded from the curated pool and returns
                        structured reasoning + an A/B/C/D letter.

Key tunables live in the AGENT CONFIGURATION block below.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIGURATION
# One dict per agent role.  model= is the intended backend; the actual instance
# is injected via EpisodicAgent(llm=..., retriever_llm=...) — see __init__.
# enable_thinking / effort / tool_call are applied directly from this dict.
#
# Switch the active config by changing MODEL_CONFIG at the bottom of this block.
# =============================================================================

MODEL_CONFIG_SONNET_4_6: Dict[str, Any] = {
    # ---- Tool-use retrieval loop ----------------------------------------
    "retrieval_agent": {
        "model":                 "claude-sonnet-4-6",
        "max_tokens":            4096,
        "enable_thinking":       True,    # interleaved thinking between tool calls
        "effort":                "medium",
        "tool_call":             True,    # search_action_speech / search_activity / search_conversation / curate_evidence / answer
        "retrieval_curate_tool": True,    # include curate_evidence in retrieval tool schema
        "parallel_tool_calls":   True,    # Claude ignores this; kept for config parity
    },
    # ---- Evidence curation (invoked when pool > _CURATE_EVIDENCE_THRESHOLD) -
    "curation_agent": {
        "model":           "claude-sonnet-4-6",
        "max_tokens":      4096,
        "enable_thinking": True,
        "effort":          "medium",
        "tool_call":       True,    # curate_evidence tool only
    },
    # ---- Final QA call (no tools, pure reasoning) -----------------------
    "answering_agent": {
        "model":           "claude-sonnet-4-6",
        "max_tokens":      4096,
        "enable_thinking": True,
        "effort":          "high",
        "tool_call":       False,
    },
    # ---- Auxiliary helpers (legacy non-tool-use ReAct path only) --------
    "helper_agent": {
        "model":           "claude-sonnet-4-6",
        "max_tokens":      1024,
        "enable_thinking": False,
        "effort":          "low",
        "tool_call":       False,
    },
}

MODEL_CONFIG_OPUS_4_8: Dict[str, Any] = {
    # ---- Tool-use retrieval loop ----------------------------------------
    "retrieval_agent": {
        "model":              "claude-opus-4-8",
        "max_tokens":         4096,
        "enable_thinking":    True,
        "effort":             "medium",
        "tool_call":             True,
        "retrieval_curate_tool": True,
        "parallel_tool_calls":   True,    # Claude ignores this; kept for config parity
    },
    # ---- Evidence curation --------------------------------------------------
    "curation_agent": {
        "model":           "claude-opus-4-8",
        "max_tokens":      4096,
        "enable_thinking": True,
        "effort":          "medium",
        "tool_call":       True,
    },
    # ---- Final QA call ------------------------------------------------------
    "answering_agent": {
        "model":           "claude-opus-4-8",
        "max_tokens":      4096,
        "enable_thinking": True,
        "effort":          "high",
        "tool_call":       False,
    },
    # ---- Auxiliary helpers --------------------------------------------------
    "helper_agent": {
        "model":           "claude-opus-4-8",
        "max_tokens":      1024,
        "enable_thinking": False,
        "effort":          "low",
        "tool_call":       False,
    },
}

MODEL_CONFIG_QWEN_3_6_27B: Dict[str, Any] = {
    # Qwen tool_call mapping (vLLM chat_template_kwargs):
    #   tool_call=True  → preserve_thinking=True + enable_thinking=True
    #   tool_call=False → preserve_thinking=False  (enable_thinking controlled separately)
    #
    # ---- Tool-use retrieval loop ----------------------------------------
    "retrieval_agent": {
        "model":              "Qwen/Qwen3.6-27B-FP8",
        "max_tokens":         4096,
        "enable_thinking":    True,
        "tool_call":             True,
        "retrieval_curate_tool": False,   # no curate_evidence in retrieval; curation is a separate agent call
        "parallel_tool_calls":   False,    # allow multiple search calls per round
    },
    # ---- Evidence curation (invoked when pool > _CURATE_EVIDENCE_THRESHOLD) -
    "curation_agent": {
        "model":           "Qwen/Qwen3.6-27B-FP8",
        "max_tokens":      4096,
        "enable_thinking": True,
        "tool_call":       True, 
    },
    # ---- Final QA call (no tools, pure reasoning) -----------------------
    "answering_agent": {
        "model":           "Qwen/Qwen3.6-27B-FP8",
        "max_tokens":      4096,
        "enable_thinking": True, 
        "tool_call":       False,
    },
    # ---- Auxiliary helpers (legacy non-tool-use ReAct path only) --------
    "helper_agent": {
        "model":           "Qwen/Qwen3.6-27B-FP8",
        "max_tokens":      1024,
        "enable_thinking": False,
        "tool_call":       False, 
    },
}

MODEL_CONFIG_GPT_5_4: Dict[str, Any] = {
    # GPT is a DIFFERENT system from Anthropic: tool calling goes through the
    # OpenAI Responses API (function_call / function_call_output over an `input`
    # list), driven by the dedicated `_retrieve_with_tools_once_gpt` harness —
    # NOT the Anthropic content-block loop. `enable_thinking`+`effort` map to the
    # Responses `reasoning.effort`; `parallel_tool_calls` is kept False because
    # the harness runs one tool per turn and re-feeds `response.output`
    # (reasoning items included) so multi-turn reasoning stays paired.
    # ---- Tool-use retrieval loop ----------------------------------------
    "retrieval_agent": {
        "model":                 "gpt-5.4",
        "max_tokens":            4096,
        "enable_thinking":       True,
        "effort":                "medium",
        "tool_call":             True,    # search_action_speech / search_activity / search_conversation / curate_evidence / answer
        "retrieval_curate_tool": True,    # include curate_evidence in the retrieval tool schema
        "parallel_tool_calls":   False,   # Responses harness issues one tool call per turn
    },
    # ---- Evidence curation (invoked when pool > _CURATE_EVIDENCE_THRESHOLD) -
    "curation_agent": {
        "model":           "gpt-5.4",
        "max_tokens":      4096,
        "enable_thinking": True,
        "effort":          "medium",
        "tool_call":       True,    # curate_evidence tool only
    },
    # ---- Final QA call (no tools, pure reasoning) -----------------------
    "answering_agent": {
        "model":           "gpt-5.4",
        "max_tokens":      4096,
        "enable_thinking": True,
        "effort":          "medium",
        "tool_call":       False,
    },
    # ---- Auxiliary helpers (legacy non-tool-use ReAct path only) --------
    "helper_agent": {
        "model":           "gpt-5.4",
        "max_tokens":      1024,
        "enable_thinking": False,
        "effort":          "nonw",
        "tool_call":       False,
    },
}

# Active config — change this line to switch models across all agents.
MODEL_CONFIG: Dict[str, Any] = MODEL_CONFIG_SONNET_4_6

# =============================================================================
# AGENT CONFIGURATION
# All tunables in one place — change these without touching logic below.
# =============================================================================

# ---- Retrieval agent (tool-use loop) ----------------------------------------
_MAX_TOOL_ITERATIONS = 5       # max rounds (search + curate + answer) per question

_RETRIEVAL_TOP_K = {            # FAISS candidates returned per search_* call
    "action":       20,         # action = fine-grained 30-sec clips → larger pool
    "activity":     10,
    "conversation": 10,
}

# ---- Time-decay (FAISS score × lambda^delta_hours when time_query is given) -
# 0.99 / hr ≈ 21 % drop over 24 h.  Per-instance override via lambda_* args.
# Set enable_time_decay=False in EpisodicAgent() to disable entirely.
_DEFAULT_DECAY_LAMBDA = 0.99    # mirrors agent.search._DEFAULT_DECAY_LAMBDA

# ---- Answering pipeline (curation → expand → QA) ----------------------------
_CURATE_EVIDENCE_THRESHOLD = 15  # invoke LLM curation when deduped pool exceeds this (was 30)
_MAX_ANSWERING_EVIDENCE    = 30  # hard cap on items fed to the final QA LLM

# ---- Legacy ReAct loop (non-tool-use path only) -----------------------------
_MAX_INROUND_RETRIES  = 3       # retry limit when LLM emits a malformed decision
_MAX_QUESTION_RETRIES = 2       # full-loop retry limit when zero evidence collected

# =============================================================================


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchRound:
    round_num: int
    activity_query: str
    action_query: str
    activity_hits:     List[dict] = field(default_factory=list)
    action_hits:       List[dict] = field(default_factory=list)
    conversation_hits: List[dict] = field(default_factory=list)
    raw_captions:      List[dict] = field(default_factory=list)  # raw 30-sec captions for all hits


@dataclass
class AgentResult:
    answer: str                          # A/B/C/D or "?"
    rounds: List[SearchRound] = field(default_factory=list)
    evidence: Dict[str, List[dict]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System prompts and tool schemas
# ---------------------------------------------------------------------------

def _build_tool_use_system(use_curate: bool) -> str:
    curate_tool_entry = """\
- curate_evidence(keep)                    — prune the working memory to the
                                              listed evidence IDs (drops others),
                                              leaving no more than 30 evidences.
                                              Do NOT over-curate: keep between 5 and 15
                                              items — dropping to 1 or 2 risks discarding
                                              related evidence the QA agent needs.
                                              Curate based on the question-specific requirement to
                                              determine curation policy. For example,
                                              "usually" means you need curate a diverse set
                                              of evidences across time, while "last" means you
                                              need to curate the latest evidences.
""" if use_curate else ""

    answer_desc = (
        "trigger the final QA step; the\n"
        "                                              harness feeds your curated captions\n"
        "                                              to the QA model and records the\n"
        "                                              answer. Call ONLY after\n"
        "                                              curate_evidence."
    ) if use_curate else (
        "END the retrieval phase. The answering\n"
        "                                              agent will receive your evidence and\n"
        "                                              produce the final answer letter."
    )

    workflow_step3 = (
        "3. After each search, assess whether the current working memory already\n"
        "   provides SUFFICIENT evidence to confidently distinguish between the choices.\n"
        "   If YES — stop searching, call `curate_evidence` + `answer()` immediately.\n"
        "   Do NOT keep searching out of habit or thoroughness if the answer is already\n"
        "   clear. Extra searches waste budget and risk overwriting good evidence."
    ) if use_curate else (
        "3. After each search, assess whether the current working memory already\n"
        "   provides SUFFICIENT evidence to confidently distinguish between the choices.\n"
        "   If YES — stop searching, call `answer()` immediately.\n"
        "   Do NOT keep searching out of habit or thoroughness if the answer is already\n"
        "   clear. Extra searches waste budget and risk overwriting good evidence."
    )

    workflow_step5 = (
        "5. **Before calling `answer()`, you MUST call `curate_evidence` one last time**\n"
        "   to commit your FINAL evidence selection. A SEPARATE ANSWERING AGENT will\n"
        "   receive the captions in working memory at the moment you call `answer()`,\n"
        "   plus the question and choices, and will produce the final reasoning + letter\n"
        "   itself. A `curate_evidence` call must immediately precede every `answer()`\n"
        "   call.\n"
        "6. Calling `answer()` ENDS the retrieval phase. You do NOT decide the final\n"
        "   letter — the answering agent does, from your curated evidence. So your job\n"
        "   is to commit the BEST evidence, then call `answer()`."
    ) if use_curate else (
        "5. Calling `answer()` ENDS the retrieval phase. You do NOT decide the final\n"
        "   letter — the answering agent does, from the collected evidence."
    )

    sufficiency_end = (
        "If sufficiency is met, proceed to curate + answer. Do not search further."
        if use_curate else
        "If sufficiency is met, call `answer()`. Do not search further."
    )

    anti_bias_end = (
        "- Reserve judgment until `curate_evidence` + `answer()` — that is the only\n"
        "  moment you should commit."
        if use_curate else
        "- Reserve judgment until `answer()` — that is the only moment you should commit."
    )

    evidence_ids_note = (
        "working memory across tool calls until you drop them via curate_evidence."
        if use_curate else
        "working memory across tool calls until the retrieval phase ends."
    )

    return f"""\
You are a retrieval agent for a personal egocentric memory system.
You answer multiple-choice questions about the wearer's past activities by
calling TOOLS to gather evidence, then committing to a single answer letter.

# Available tools
- search_action_speech(query, time_query?) — action/speech search: atomic 30-sec fine-grained
                                              observations of objects / people / physical movements,
                                              detailed content of conversations / discussions / claims
- search_activity(query, time_query?)      — 5-min coarse-grained activity summaries
- search_conversation(query, time_query?)  — 5-min coarse-grained conversation-topic summaries.
{curate_tool_entry}- answer()                                 — {answer_desc}

# Workflow (MANDATORY)
1. NEVER call `answer()` before issuing at least one search.
2. Issue one or more `search_*` calls per topic.
{workflow_step3}
4. If working memory does NOT yet clearly support one of the choices, search
   again with a refined query or a different level / time window.
{workflow_step5}

# Sufficiency test (apply after every search)
Evidence is SUFFICIENT when ALL of the following hold:
  - At least one caption directly names or depicts the key entity / action /
    object that the question asks about.
  - The caption timestamps allow you to resolve any time qualifier in the
    question (e.g. "last", "yesterday", "first") unambiguously.
  - No other choice can plausibly be supported by the evidence — OR you have
    searched for the alternative choices and found no supporting captions.
{sufficiency_end}

# Anti-bias rule — NO EARLY VERDICT
Do NOT form or state a preferred answer while still searching. Committing
mentally to a choice BEFORE all evidence is gathered will bias your subsequent
queries toward confirming that choice and away from disconfirming evidence.
- Do not say "I think the answer is X" or "likely X" during search rounds.
- Treat all choices as equally possible until you call `answer()`.
- If you find strong evidence for one choice, still issue at least one more
  search targeting the other plausible choices to rule them out before
  committing. Absence of disconfirming evidence is NOT the same as proof.
{anti_bias_end}

# Query writing rules (apply to ALL search_* tools)
The query MUST be GROUNDED — never abstract. Two valid families:

(A) ACTION queries (for action / activity searches): object, action+object,
    person+action, or person+action+object — grounded to a concrete entity a
    caption-writer would describe. 3–5 words.
       e.g. "red screwdriver", "open cardboard box",
            "Alice pick up screwdriver", "Jake serve noodles", "Bob go shopping".

(B) UTTERANCE queries (for conversation searches): TELEGRAPHIC keyword
    phrase, 3–5 words; speaker + speech-verb + topic, or statement / fact. Drop articles/copulas.
       e.g. "I suggest pizza dinner", "Alice forgot buy eggs",
            "Jake complain wifi", "Hema Fresh distance".

Reject any query that is a category/umbrella noun ("the tool", "the meal"),
a meta-description of a conversation ("the discussion about X"), an abstract
attribute ("the reason", "the relationship"), or a paraphrase of the question
("who used it first"). Use COMMON everyday words ("pour" not "decant", "eat"
not "consume"). Use real housemate names (Jake/Alice/Tasha/Lucia/Katrina/
Shure) or "I" — never "someone" / "they" / "people".

# Time query (optional, per tool)
A time_query LOCALIZES where the answer lives in the timeline.

Grounding (NO HALLUCINATION): build time_query ONLY from words in the
question and the dataset's recorded range; never from outside common knowledge
(assumed schedules, typical habits, weather, holidays). If the question
contains no time / order / recurrence cue, OMIT time_query.

Format: "DAYn HHMMSSFF" (point) or "DAYn HHMMSSFF to DAYm HHMMSSFF" (interval).
HHMMSSFF is 8 digits (HH MM SS FF, frame=00). e.g. noon = 12000000.

"Last" / "previous" vs CURRENT event: when the question asks about a PRIOR
occurrence, matches very close to the CURRENT TIME may BE the current event
the question refers to as "just now" — push the window further back.

# How the search harness uses time_query (so you can be aggressive)
Each candidate's similarity score is multiplied by lambda^(hours away), with
lambda≈0.99. Inside the window → full weight. ~1h away → ×0.99; ~12h → ×0.89;
a full day → ×0.79. Soft prior, NOT a hard filter. COMMIT to a tight window
around your single best guess; only widen when you genuinely have no signal.

# Evidence IDs
Each search returns NEW captions with fresh integer IDs. They persist in the
{evidence_ids_note}
Duplicates across searches are deduplicated by caption — the same caption
keeps its first ID.

# Chinese-native word translation — ALWAYS try synonyms
The captions are machine-translated from Chinese. This creates a critical
vocabulary gap: the question may use one English word while the captions use
a completely different English word for the SAME underlying Chinese concept.
Your query embedding will NOT match the caption if the words differ — so a
failed or weak search does not mean the event didn't happen; it may just mean
the translation used different vocabulary.

**MANDATORY synonym rule**: after EVERY search that returns weak or no
matches, examine the returned captions for clues about what vocabulary the
translator chose, and issue at least one follow-up search with the
alternative word. Especially critical for Chinese-native terms that may
appear as pinyin or a literal but uncommon English rendering — the sentence
embedding has NO way to bridge these if the surface words don't match:
  - 串串/烤串   "skewer" ↔ "kebab" / "BBQ stick" / "meat stick"
  - 外卖        "take-out" ↔ "takeaway" / "food delivery" / "delivery order"
  - 盒马/超市   "Hema" ↔ "Hema Fresh" / "supermarket" / "the store"
  - 微信        "WeChat" ↔ "WeChat message" / "message" / "chat"
  - 麻辣烫      "malatang" ↔ "spicy hotpot" / "hotpot" / "spicy soup"
  - 电饭锅      "rice cooker" ↔ "cooker" / "pot"
  - 快递        "delivery" ↔ "package" / "parcel" / "courier"

Beyond the list above: whenever you see a word in the returned captions that
you believe refers to the same concept as your original query — even if it is
not listed here — treat it as a synonym and search for it. Your own judgment
about synonymy is valid. A pinyin word (e.g. "malatang", "baozi", "congee")
that appears in captions may be the only form the translator used; if the
question uses a descriptive English phrase, you MUST search the pinyin form
too, and vice versa.

# Never:
- Call `answer` without searches.
- Call `answer()` while working memory is empty or clearly off-topic.
- Invent times not in the question.
- Issue an abstract query.
"""

_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "search_action_speech",
        "description": "Search the ACTION index (atomic 30-sec observations of "
                       "objects/people/physical movements). Adds matching captions "
                       "to the working evidence pool with fresh integer IDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Grounded action query (object / action+object / "
                                         "person+action / person+action+object), ≤12 words."},
                "time_query": {"type": "string",
                               "description": "Optional time window in dataset format "
                                              "('DAYn HHMMSSFF' or 'DAYn HHMMSSFF to "
                                              "DAYm HHMMSSFF'). Omit if no time signal."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_activity",
        "description": "Search the ACTIVITY index (5-min activity summaries). "
                       "Adds matching captions to the working evidence pool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":      {"type": "string"},
                "time_query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_conversation",
        "description": "Search the CONVERSATION index (conversation-topic summaries). "
                       "Use TELEGRAPHIC keyword utterance phrasing. Adds matching "
                       "captions to the working evidence pool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":      {"type": "string"},
                "time_query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "curate_evidence",
        "description": "Prune the working evidence memory: keep only the listed "
                       "evidence IDs, drop all others. Use after each batch of "
                       "searches to maintain a focused memory of 5–15 items. "
                       "Do NOT over-curate: keeping fewer than 5 items risks "
                       "discarding related evidence the QA agent needs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keep": {"type": "array", "items": {"type": "integer"},
                         "description": "Evidence IDs to keep in working memory."},
            },
            "required": ["keep"],
        },
    },
    {
        "name": "answer",
        "description": "END the retrieval phase. Takes no arguments. A SEPARATE "
                       "ANSWERING AGENT will be invoked with the captions in "
                       "your working memory + the question + the choices, and "
                       "will produce the final reasoning and answer letter "
                       "itself. You do NOT pick the letter — you commit the "
                       "best evidence, then call answer(). Call this ONLY "
                       "after curate_evidence (the workflow rejects answer() "
                       "if curate_evidence is not the immediately preceding "
                       "call).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# Schema without curate_evidence — used when retrieval_curate_tool=False (e.g. Qwen).
# Curation is handled by a separate agent call after retrieval ends.
_TOOLS_SCHEMA_NO_CURATE: List[Dict[str, Any]] = [
    t for t in _TOOLS_SCHEMA if t["name"] != "curate_evidence"
]
# Update answer() description to remove the curate_evidence prerequisite.
_TOOLS_SCHEMA_NO_CURATE = [
    {**t, "description": (
        "END the retrieval phase. Takes no arguments. When you have gathered "
        "sufficient evidence, call answer() to hand off to the answering agent."
    )} if t["name"] == "answer" else t
    for t in _TOOLS_SCHEMA_NO_CURATE
]


# -----------------------------------------------------------------------------
# GPT (OpenAI Responses API) tool schema.
#
# GPT and Anthropic are DIFFERENT systems: Anthropic tools use `input_schema`
# and are exchanged via content blocks (tool_use / tool_result); GPT tools are
# flat `{"type":"function","name","description","parameters"}` and are exchanged
# via Responses `input` items (function_call / function_call_output). These
# constants are the GPT-native rendering of the same five tools above, consumed
# only by the GPT harness (`_retrieve_with_tools_once_gpt`).
# -----------------------------------------------------------------------------
def _to_openai_tools(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
        for t in anthropic_tools
    ]


_TOOLS_SCHEMA_OPENAI: List[Dict[str, Any]] = _to_openai_tools(_TOOLS_SCHEMA)
_TOOLS_SCHEMA_OPENAI_NO_CURATE: List[Dict[str, Any]] = _to_openai_tools(_TOOLS_SCHEMA_NO_CURATE)

class EpisodicAgent:
    """
    ReAct-style agent that issues search queries against the episodic DAG.
    """

    def __init__(
        self,
        llm,
        dag_file: str,
        raw_captions_file: Optional[str] = None,
        max_rounds: int = 5,
        top_k: int = 5,
        retriever_llm=None,
        lambda_action: Optional[float] = None,
        lambda_activity: Optional[float] = None,
        lambda_conversation: Optional[float] = None,
        enable_time_decay: bool = True,
    ) -> None:
        """
        Args:
            llm:           the MAIN reasoning model — used for the per-round
                           search/answer decision (thinking mode).
            retriever_llm: the AUXILIARY model used for non-thinking helpers:
                             - decision JSON parser (`_parse_decision_via_llm`)
                             - thinking-block summarizer (`_log_thinking`)
                             - time-query normalizer (`_normalize_time_query`)
                           (The caption reranker `_llm_rerank_captions` uses the MAIN
                           `llm`, not this one.)
                           Defaults to `llm` (backwards compatible single-LLM mode).
                           In split-model setups, this is typically the cheaper
                           local model (e.g. local-vLLM Qwen3.6-27B-FP8).
            raw_captions_file:
                           optional override for the raw 30-sec caption file used
                           by the final answering agent. When omitted, EpisodicSearch
                           falls back to {dirname(dag_file)}/captions/raw.json.
            lambda_action / lambda_activity:
                           per-hour time-decay base for the action / activity
                           similarity scores (see agent.search). The decay is
                           applied only when the agent emits a `time_query` that
                           resolves to a valid target window. Default 0.99.
            enable_time_decay:
                           master switch. When False, time decay is OFF entirely
                           (lambda_action / lambda_activity are forced to None, so
                           no decay is ever applied and logs show λ^Δt=off). The
                           agent still extracts the time_query (it still biases the
                           caption reranker) — only the embedding-score decay is
                           disabled.
        """
        from agent.search import _DEFAULT_DECAY_LAMBDA
        self.llm           = llm
        self.retriever_llm = retriever_llm if retriever_llm is not None else llm
        self.max_rounds    = max_rounds
        self.top_k         = top_k
        self.enable_time_decay = enable_time_decay
        if not enable_time_decay:
            self.lambda_action       = None
            self.lambda_activity     = None
            self.lambda_conversation = None
        else:
            self.lambda_action       = (_DEFAULT_DECAY_LAMBDA if lambda_action       is None
                                        else lambda_action)
            self.lambda_activity     = (_DEFAULT_DECAY_LAMBDA if lambda_activity     is None
                                        else lambda_activity)
            self.lambda_conversation = (_DEFAULT_DECAY_LAMBDA if lambda_conversation is None
                                        else lambda_conversation)

        from agent.search import EpisodicSearch
        self._search = EpisodicSearch(
            dag_file=dag_file,
            raw_captions_file=raw_captions_file,
        )
        self._search.build_all_indexes()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _NEAR_GT_SEC = 5 * 60   # ±5 min band for the blue "Next to GT" marker

    def _mark_gt(self, line: str, date: str, start_time: str, end_time: str,
                 target_abs: Optional[int]) -> str:
        """Color a log line by proximity to the GT target_abs:
        red + '*** GT ***' if the node covers it, blue + '** Next to GT **'
        if within ±_NEAR_GT_SEC, otherwise unchanged."""
        if target_abs is None:
            return line
        from agent.search import _abs_ts as _ts
        try:
            s = _ts(date, start_time)
            e = _ts(date, end_time)
        except Exception:
            return line
        if s <= target_abs <= e:
            return f"\033[91m{line}  *** GT ***\033[0m"
        if min(abs(target_abs - s), abs(target_abs - e)) <= self._NEAR_GT_SEC:
            return f"\033[94m{line}  ** Next to GT **\033[0m"
        return line

    def _log_thinking(self, text: str) -> None:
        """Extract <think> block, summarize with LLM (5 sentences), and log."""
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if not m:
            return
        thinking = m.group(1).strip()
        if not thinking:
            return
        try:
            summary = self.retriever_llm.generate(
                [
                    {"role": "system", "content":
                        "You are a reasoning summarizer. Output plain prose only — no JSON, no bullet points, no decision fields. "
                        "Summarize the reasoning process in exactly 5 sentences, citing specific timestamps and caption text as evidence."},
                    {"role": "user", "content":
                        f"Summarize the following reasoning in exactly 5 sentences. "
                        f"Explicitly cite specific timestamps and caption content that were used as evidence (e.g. '[DAY2 11:20-11:21] Jake picked up the screwdriver'). "
                        f"Do not generalize — name the exact moments and facts the reasoning relied on.\n\n{thinking}"},
                ],
            )
            logger.info("\033[90m[Thinking]\n%s\033[0m", summary.strip())
        except Exception as e:
            logger.warning("Failed to summarize thinking: %s", e)

    def _parse_decision(self, text: str) -> Optional[dict]:
        """Extract the first JSON object from LLM output. Returns None on parse failure."""
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return None

    _REPAIR_SYSTEM = (
        "You convert a retrieval-agent's free-form response into a strict JSON decision.\n"
        "Schema (output ONLY one of these):\n"
        '  {"decision": "search", "query": "<short specific moment, who did what or what happened, <=12 words, no and/or/while/then>", "time_query": "<optional point or interval>"}\n'
        '  {"decision": "search", "query": "<...>"}    (when the response carries no time reference — omit the field)\n'
        '  {"decision": "answer"}\n'
        "Rules:\n"
        "- The input may contain reasoning, <think> blocks, prose, markdown, or partial JSON. Ignore everything except the final intent.\n"
        "- If the response indicates a search, extract the most concrete query string. If the response gave reasoning but no clear query, pick the most specific moment named.\n"
        "- PRESERVE the agent's `time_query` verbatim if present in the response. If absent, OMIT the field — do not invent one.\n"
        "- If the response intends to answer (or commits to a choice), output {\"decision\": \"answer\"}.\n"
        "- Output JSON ONLY. No prose, no <think>, no markdown."
    )

    def _parse_decision_via_llm(self, raw_text: str) -> Optional[dict]:
        """
        Primary parser: send the agent's raw output (which may contain <think>,
        prose, markdown, JSON) to a NON-THINKING LLM call and ask it to emit the
        strict JSON decision. Robust against thinking-mode noise and partial JSON.
        """
        try:
            out = self.retriever_llm.generate(
                [{"role": "system", "content": self._REPAIR_SYSTEM},
                 {"role": "user",   "content": raw_text}],
            )
        except Exception as e:
            logger.debug("[Agent] non-thinking decision-parser call failed: %s", e)
            return None
        return self._parse_decision(out)

    # Back-compat alias — older callers (or external scripts) may reference
    # _repair_decision. Keep the name pointing at the new primary parser.
    _repair_decision = _parse_decision_via_llm

    # ------------------------------------------------------------------
    # Time-query normalization
    # ------------------------------------------------------------------
    _TIME_QUERY_NORMALIZER_SYSTEM = (
        "You normalize a retrieval agent's time query into ONE strict canonical line.\n"
        "Canonical format (dataset format):\n"
        "  point    : DAYn HHMMSSFF\n"
        "  interval : DAYn HHMMSSFF to DAYm HHMMSSFF   (may span days)\n"
        "HHMMSSFF is 8 digits: HH(00-23) MM SS FF(frame, use 00). "
        "noon=12000000, 3pm=15000000, 9:30am=09300000, midnight=00000000.\n"
        "\n"
        "You are given the CURRENT TIME and the agent's (possibly fuzzy) time query.\n"
        "Resolve relative phrases using the current time:\n"
        "  usually / typically / often / every time -> DAY1 00000000 to <current>\n"
        "  last / most recent / just now / currently -> <current>\n"
        "  earlier / recently / a while ago -> (<current> minus 12 hours) to <current>\n"
        "  first / originally / at the beginning -> DAY1 00000000\n"
        "  DAYX morning 06000000-12000000, afternoon 12000000-18000000, "
        "evening 18000000-22000000, night 22000000-23595900\n"
        "  an explicit clock time -> the matching point\n"
        "\n"
        "Output ONLY the canonical line, or exactly NONE when there is no usable "
        "time reference. No prose, no JSON, no quotes, no explanation."
    )

    def _normalize_time_query(
        self,
        question: str,
        raw_time_query: str,
        before_date: Optional[str],
        before_time: Optional[str],
    ) -> str:
        """Normalize the agent's time_query into a canonical dataset-format string.

        Returns "" when there is no usable time reference (decay then stays off).
        Runs as a NON-THINKING call so the format is enforced deterministically.
        """
        cur = (f"{before_date} {before_time}"
               if before_date and before_time else "(unknown)")
        user = (
            f"Current time: {cur}\n"
            f"Question: {question}\n"
            f"Agent time query: {raw_time_query or '(none)'}\n\n"
            f"Output the canonical time query line, or NONE."
        )
        try:
            out = self.retriever_llm.generate(
                [{"role": "system", "content": self._TIME_QUERY_NORMALIZER_SYSTEM},
                 {"role": "user",   "content": user}],
            )
        except Exception as e:
            logger.debug("[Agent] time-query normalizer call failed: %s", e)
            return ""
        if not out or not out.strip():
            return ""
        line = out.strip().splitlines()[0].strip().strip('"').strip()
        if not line or line.upper() == "NONE":
            return ""
        return line

    @staticmethod
    def _validate_decision(decision: Optional[dict], has_prior_rounds: bool) -> tuple:
        """
        Return (ok, feedback). `ok=True` means the decision is structurally valid
        and may be acted on.
        Schema: {"decision": "search", "query": "..."} or {"decision": "answer"}.
        Legacy keys `action_query` / `activity_query` are still accepted on input.
        """
        if decision is None:
            return False, (
                "Your previous response was not valid JSON. Output ONLY a single JSON object, "
                'e.g. {"decision": "search", "query": "..."} or {"decision": "answer"}. '
                "No prose, no markdown."
            )
        d = decision.get("decision")
        if d not in ("search", "answer"):
            return False, (
                'Missing or invalid `decision` field. It must be exactly "search" or "answer". '
                "Re-emit your JSON."
            )
        if d == "answer" and not has_prior_rounds:
            return False, (
                "You cannot answer before retrieving any evidence. Issue at least one search first. "
                'Re-emit your JSON as {"decision": "search", "query": "..."} with a concrete query '
                "targeting the question."
            )
        if d == "search":
            q = (
                decision.get("query")
                or decision.get("action_query")
                or decision.get("activity_query")
                or ""
            ).strip()
            if not q:
                return False, (
                    '`decision: "search"` requires a non-empty `query` field. '
                    'Re-emit your JSON as {"decision": "search", "query": "<specific moment>"}.'
                )
        return True, ""

    def _collect_raw_captions(self, hits: List[dict], query: str,
                              target_abs: Optional[int] = None,
                              extra_caps: Optional[List[dict]] = None,
                              time_qualifier: Optional[str] = None,
                              target_window: Optional[tuple] = None,
                              decay_lambda: Optional[float] = None,
                              choices: Optional[Dict[str, str]] = None) -> List[dict]:
        """
        Caption selection by LLM rerank:
          1. Pool action nodes (direct hits + drilled from coarse hits)
          2. Re-rank pool against query (embedding cosine) → top-20 actions
          3. Collect their raw captions, dedup by caption index
          4. Merge in `extra_caps` (e.g. captions from activity-search hits)
          5. Send deduped captions to LLM (no thinking) to rerank
          6. Return top-K (default 6, see _RERANK_TOPK)
        """
        # Step 1: Pool all action nodes (direct hits + drilled from coarse hits)
        seen_action_ids: set = set()
        pooled_action_ids: List[int] = []

        direct_actions = [h for h in hits if h.get("level") == "action"]
        coarse_hits    = [h for h in hits if h.get("level") != "action"]

        for a in direct_actions:
            if a["id"] not in seen_action_ids:
                seen_action_ids.add(a["id"])
                pooled_action_ids.append(a["id"])

        for hit in coarse_hits:
            for a in self._search.drill_to_actions(hit["id"]):
                if a["id"] not in seen_action_ids:
                    seen_action_ids.add(a["id"])
                    pooled_action_ids.append(a["id"])

        logger.info("[Caption] pooled %d actions from %d hits", len(pooled_action_ids), len(hits))

        # Step 2: Re-rank pool by query similarity; use top-20 downstream.
        _ACTION_USE_K     = 20
        _ACTION_DISPLAY_K = 50
        # When the pool comes only from direct action hits (no activity drill-down),
        # the rerank is an identity and the rows duplicate the previous log block —
        # use the direct hits as-is and skip the redundant display.
        only_direct_actions = bool(direct_actions) and not coarse_hits
        if only_direct_actions:
            ranked_actions = direct_actions
        elif pooled_action_ids:
            ranked_actions = self._search.search_within(
                query=query, node_ids=pooled_action_ids,
                top_k=max(_ACTION_USE_K, _ACTION_DISPLAY_K),
                target_window=target_window, decay_lambda=decay_lambda,
            )
            logger.info("[Caption] showing top %d actions (using top %d downstream)",
                        len(ranked_actions), _ACTION_USE_K)
            for rank, a in enumerate(ranked_actions, start=1):
                marker = " <- used" if rank <= _ACTION_USE_K else ""
                final = a.get("score", 0)
                sim   = a.get("base_score", final)
                if decay_lambda is None:
                    decay_str = "off"
                else:
                    decay_str = "%.3f" % ((final / sim) if abs(sim) > 1e-9 else 1.0)
                line = "action #%02d [%s %s-%s] sim=%.3f λ^Δt=%s score=%.3f  %s%s" % (
                    rank, a["date"], a["start_time"], a["end_time"],
                    sim, decay_str, final, a["text"], marker)
                logger.info(self._mark_gt(line, a["date"], a["start_time"],
                                          a["end_time"], target_abs))
        else:
            ranked_actions = []

        top_actions = ranked_actions[:_ACTION_USE_K]

        # Step 3: Collect captions from top-20 actions, dedup by caption index
        seen_cap_keys: set = set()
        pooled_caps: List[dict] = []
        for action in top_actions:
            for cap in self._search.get_raw_captions(action):
                key = cap.get("id") or cap.get("index") or id(cap)
                if key in seen_cap_keys:
                    continue
                seen_cap_keys.add(key)
                pooled_caps.append(cap)

        logger.info("[Caption] %d unique captions pooled from top actions", len(pooled_caps))

        # Step 4: Merge in captions from activity-search hits.
        if extra_caps:
            pooled_caps.extend(extra_caps)
            logger.info("[Caption] + %d captions from activity hits → %d total in pool",
                        len(extra_caps), len(pooled_caps))

        return self._llm_rerank_captions(pooled_caps, query=query,
                                         target_abs=target_abs,
                                         time_qualifier=time_qualifier,
                                         choices=choices)

    # ------------------------------------------------------------------
    # Caption rerank
    # ------------------------------------------------------------------
    _RERANK_RANK_K     = 20  # how many ranked entries the reranker returns (with reasons)
    _RERANK_TOPK       = 6   # of those, how many to actually use downstream
    _EVIDENCE_MEMORY_K = 10  # working evidence memory size carried across rounds

    _EVIDENCE_SYSTEM = (
        "You curate a working EVIDENCE MEMORY for an egocentric video memory "
        "question. Each round you see your CURRENT memory (kept from prior rounds) "
        "and NEW candidate captions from this round's reranker. Pick UP TO "
        "{max_keep} captions to KEEP in memory.\n"
        "\n"
        "Keep a caption when it:\n"
        "  - directly supports or refutes a candidate answer to the question;\n"
        "  - provides useful surrounding context (immediately before/after a key "
        "moment);\n"
        "  - is needed to triangulate (corroborates a different caption).\n"
        "\n"
        "Drop a caption when it is redundant with a stronger one, off-topic, "
        "or only weakly relevant.\n"
        "\n"
        "Output ONLY a JSON object: "
        "{{\"keep\": [<int>, ...], \"reason\": \"<one short sentence on the "
        "overall keep strategy this round>\"}}.\n"
        "Indices refer to the combined list shown to you (memory first, then "
        "new candidates). Order does NOT matter; duplicates are ignored. "
        "Never output an empty array — at minimum keep the strongest few candidates."
    )

    _RERANK_SYSTEM = (
        "You are a caption reranker for an egocentric video memory system. "
        "You will see a focused QUERY, optionally a TARGET TIME, optionally the "
        "multiple-choice CHOICES for the underlying question, and a numbered list "
        "of candidate captions. Rank the candidates by a JOINT judgement of: "
        "(a) how well each caption SUPPORTS the query — concrete topical/evidential "
        "match (same object, action, or person doing the same thing); "
        "(b) how well it fits the TARGET TIME, when one is given; "
        "(c) when CHOICES are given, how directly the caption SUPPORTS or REFUTES "
        "one of the listed choices — captions that pin down or rule out a specific "
        "choice are MORE valuable than generic topical matches. "
        "For EACH ranked entry, write a ONE-SENTENCE reason naming the specific "
        "evidence (object, person, action) that makes it relevant; if it bears on "
        "a particular choice, name that letter (e.g. 'supports B', 'rules out A'); "
        "if a target time is given, why the time alignment matters. "
        "Output ONLY a JSON array of objects "
        "{\"index\": <int>, \"reason\": \"<one short sentence>\"}, ranked best first. "
        "Never output an empty array. No prose outside the JSON."
    )

    _RERANK_EVIDENCE_GUIDE = (
        "EVIDENCE TEST (apply to every candidate you rank):\n"
        "  - Does the caption contain concrete evidence that SUPPORTS the query — "
        "the same object, the same action, or the same person doing the same thing?\n"
        "  - Name that evidence in the reason. Concrete (\"shows Jake serving "
        "noodles\") beats vague (\"looks relevant\").\n"
        "  - If a caption only mentions the topic in passing, or does not actually "
        "depict it, rank it LOWER even if a keyword matches.\n"
        "\n"
        "GENERALIZATION RULES (apply BEFORE the joint judgement):\n"
        "  - Match the GENERAL intent of the query FIRST. Additional qualifiers "
        "(specific objects, persons, places, time, or modifiers in the question) "
        "are a SECONDARY filter — use them to break ties between similarly-strong "
        "candidates, not as hard requirements to disqualify a caption.\n"
        "  - Caption text is plain everyday English. Match by COMMONLY-USED "
        "wording: if the question uses an unusual, formal, or technical word, "
        "look for the common everyday equivalent in the captions "
        "(e.g. \"decant\" ≈ \"pour\", \"consume\" ≈ \"eat\", \"commence\" ≈ "
        "\"start\", \"vehicle\" ≈ \"car\", \"converse\" ≈ \"talk\"). "
        "The depicted action and objects matter, not the surface vocabulary.\n"
        "\n"
        '"LAST" / "PREVIOUS" vs CURRENT EVENT:\n'
        "  - When the question asks about the LAST / PREVIOUS / PRIOR occurrence "
        "of something (e.g. \"the last meeting\", \"what did I cook last time\"), "
        "a candidate whose timestamp is very close to the CURRENT TIME may BE the "
        "same event the question calls \"current / just now\" — NOT the prior one. "
        "Rank such near-now candidates LOWER and PREFER an earlier match that "
        "clearly precedes the most recent moment; note this in the reason when "
        "it applies (e.g. \"too close to current time — likely the current event, "
        "not the prior one\").\n"
        "  - For \"current / just-now / right-now\" queries the same near-now "
        "candidates are the BEST match — rank them HIGHER.\n"
        "  - Use your own judgement on what counts as \"close\" given the "
        "candidates' time distribution; do not apply a fixed numeric threshold.\n"
        "\n"
        "The final rank is the JOINT product of evidence strength (a) and time fit (b)."
    )

    _RERANK_TARGET_WINDOW_GUIDE = (
        "A TARGET TIME (a point or interval in 'DAYn HHMMSSFF' dataset format, where "
        "HHMMSSFF is hour-minute-second-frame) may be given. Apply it on top of "
        "topical relevance:\n"
        "  - Prefer captions whose timestamp falls INSIDE the target window.\n"
        "  - If few or none fall inside, prefer the captions CLOSEST to it in time.\n"
        "  - For a wide interval (e.g. one spanning most of the history) treat it as a "
        "weak prior and rank mainly by topical relevance.\n"
        "  - When no target time is given, rank purely by topical relevance."
    )

    def _llm_rerank_captions(
        self,
        pooled_caps: List[dict],
        query: str,
        target_abs: Optional[int] = None,
        time_qualifier: Optional[str] = None,
        choices: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """Dedup an already-pooled caption list, LLM-rerank to top-K (default 6).

        Inputs to the rerank LLM:
          * `query`          — the agent's focused per-round search query (topical signal).
          * `time_qualifier` — the canonical target time (point/interval in dataset
                               format, e.g. "DAY2 12000000 to DAY2 18000000") produced
                               by the time-query normalizer. When non-empty, the
                               reranker applies it on top of topical relevance.
          * `choices`        — the multiple-choice options for the underlying question
                               as {"A": "...", "B": "...", "C": "...", "D": "..."}.
                               When given, the reranker is asked to prefer captions
                               that directly support or refute one of the choices.
        """
        from agent.search import _abs_ts as _ts

        K = self._RERANK_TOPK
        RANK_K = self._RERANK_RANK_K

        # Dedup defensively (callers may not have already dedup'd)
        seen_keys: set = set()
        deduped: List[dict] = []
        for c in pooled_caps:
            key = c.get("id") or c.get("index") or id(c)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(c)
        pooled_caps = deduped

        if not pooled_caps:
            return []
        if len(pooled_caps) <= K:
            return pooled_caps

        # Sort chronologically so the LLM can apply time-qualifier guidance
        # by reading the list top-to-bottom (earliest → latest).
        def _cap_ts(c: dict) -> int:
            try:
                return _ts(c.get("date") or "DAY1", c.get("start_time") or "0")
            except Exception:
                return 0
        pooled_caps = sorted(pooled_caps, key=_cap_ts)

        # Cap the number of ranked entries at the pool size (no point asking for
        # 20 when the pool only has 12).
        n = len(pooled_caps)
        rank_n = min(RANK_K, n)

        # Log what's about to be fed into the rerank LLM — especially the
        # time qualifier, so it's easy to verify the outer-loop agent extracted
        # the expected value (or none).
        tq_display = (time_qualifier or "").strip() or "(none)"
        n_choices = sum(1 for v in (choices or {}).values() if str(v).strip())
        logger.info(
            "[Caption] rerank input: query=%r  target_time=%s  choices=%d  "
            "pool=%d  rank_K=%d  use_K=%d",
            query, tq_display, n_choices, n, rank_n, K,
        )

        cap_lines = [
            f"[{i}] {c.get('date','')} {c.get('start_time','')}–{c.get('end_time','')}  "
            f"{c.get('text') or c.get('caption','')}"
            for i, c in enumerate(pooled_caps)
        ]
        # The outer-loop agent extracts a concrete target time (point/interval) from
        # the user question, normalized to dataset format; we forward only that.
        tq = (time_qualifier or "").strip()
        # Choices block: list non-empty choices so the reranker can favor captions
        # that pin down or rule out a specific option.
        if choices:
            choice_lines = [
                f"  {letter}) {str(text).strip()}"
                for letter, text in sorted(choices.items())
                if str(text).strip()
            ]
        else:
            choice_lines = []
        if tq:
            query_block = (
                f"Query: {query}\n"
                f"Target time (from the user question): {tq}\n"
            )
            instruction_tail = (
                f"Rank up to {rank_n} captions by the JOINT criterion (evidence "
                "for the query + fit to the target time + support/refute the choices)."
            )
            time_guide = f"\n{self._RERANK_TARGET_WINDOW_GUIDE}\n"
        else:
            query_block = f"Query: {query}\n"
            instruction_tail = (
                f"Rank up to {rank_n} captions by how strongly they SUPPORT the "
                "query (and support/refute the choices when given)."
            )
            time_guide = ""   # no target time → no need to show the time guide
        choices_block = (
            "\nChoices for the underlying question:\n" + "\n".join(choice_lines) + "\n"
            if choice_lines else ""
        )

        example = (
            '[{"index": 3, "reason": "shows Jake serving noodles into his bowl — '
            'direct evidence for the query"}, '
            '{"index": 7, "reason": "same noodle bowl on the kitchen counter '
            'moments later"}, ...]'
        )
        base_prompt = (
            f"{query_block}"
            f"{choices_block}\n"
            f"Candidate captions (indices 0..{n - 1}, sorted chronologically — earliest first):\n"
            + "\n".join(cap_lines) + "\n\n"
            f"{instruction_tail}\n"
            f"\n{self._RERANK_EVIDENCE_GUIDE}\n"
            f"{time_guide}\n"
            f"Output ONLY a JSON array of up to {rank_n} objects "
            f'{{"index": <int in 0..{n - 1}>, "reason": "<one short sentence>"}}, '
            "ranked best first. The array MUST NOT be empty.\n"
            f"Example: {example}"
        )

        def _try_rerank(user_content: str) -> List[Tuple[dict, str]]:
            """Return List[(caption_dict, reason_str)] in rank order, deduped."""
            # Use the MAIN agent model (e.g. Claude, thinking mode) for reranking.
            raw = self.llm.generate(
                [
                    {"role": "system", "content": self._RERANK_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
            )
            # Strip any <think>...</think> first so we parse the final answer.
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            # Find the outermost JSON array — non-greedy []-balance approximation:
            # take from the first '[' to the last ']' in the post-think text.
            i = text.find("[")
            j = text.rfind("]")
            if i < 0 or j <= i:
                raise ValueError(f"no JSON array in rerank output: {raw!r}")
            arr = json.loads(text[i:j + 1])
            if not isinstance(arr, list) or not arr:
                raise ValueError(f"empty/invalid array: {arr!r}")
            ranked: List[Tuple[dict, str]] = []
            seen_idx: set = set()
            for entry in arr:
                # Tolerate the legacy plain-integer format too.
                if isinstance(entry, int):
                    idx, reason = entry, ""
                elif isinstance(entry, dict):
                    idx = entry.get("index")
                    reason = str(entry.get("reason") or "").strip()
                else:
                    continue
                if not isinstance(idx, int) or not (0 <= idx < n):
                    continue
                if idx in seen_idx:
                    continue
                seen_idx.add(idx)
                ranked.append((pooled_caps[idx], reason))
                if len(ranked) >= rank_n:
                    break
            if not ranked:
                raise ValueError(f"no valid indices parsed from {arr!r}")
            return ranked

        ranked: List[Tuple[dict, str]] = []
        attempts = [
            base_prompt,
            base_prompt + (
                f"\n\nIMPORTANT: your previous response was empty or invalid. "
                f"You MUST return at least {K} valid objects "
                "{\"index\": <int>, \"reason\": \"...\"} from the list above. "
                "Even if relevance is weak, still pick the strongest candidates "
                "by topical relevance and any time qualifier in the query."
            ),
        ]
        for attempt_idx, prompt_text in enumerate(attempts, start=1):
            try:
                ranked = _try_rerank(prompt_text)
                if attempt_idx > 1:
                    logger.info("[Caption] LLM rerank succeeded on retry %d", attempt_idx)
                break
            except Exception as e:
                logger.warning("[Caption] LLM rerank attempt %d failed (%s)", attempt_idx, e)
        if not ranked:
            logger.warning("[Caption] all rerank attempts failed — falling back to first %d", K)
            ranked = [(c, "(fallback: rerank failed)") for c in pooled_caps[:K]]

        # Per-entry log: timestamp + reason only (no caption text). GT / Near-GT
        # coloring via _mark_gt; first K marked "<- used" (flow downstream).
        logger.info("[Caption] LLM reranked %d entries (top %d used):",
                    len(ranked), min(K, len(ranked)))
        for rank, (cap, reason) in enumerate(ranked, start=1):
            marker     = "  <- used" if rank <= K else ""
            reason_str = f"  — {reason}" if reason else ""
            date       = cap.get("date", "")
            stime      = cap.get("start_time", "")
            etime      = cap.get("end_time", "")
            line = "[Caption] #%02d [%s %s-%s]%s%s" % (
                rank, date, stime, etime, reason_str, marker)
            logger.info(self._mark_gt(line, date, stime, etime, target_abs))

        # Return the top-K captions; attach the LLM's reason so the downstream
        # evidence-memory curation step can see why each was picked.
        out: List[dict] = []
        for cap, reason in ranked[:K]:
            cap = dict(cap)
            cap["_rerank_reason"] = reason
            out.append(cap)
        return out

    def _curate_evidence_memory(
        self,
        question: str,
        current_memory: List[dict],
        new_candidates: List[dict],
        max_keep: int,
        target_abs: Optional[int] = None,
    ) -> List[dict]:
        """Ask the agent which captions to KEEP in working memory.

        Inputs:
          - current_memory: the captions kept from prior rounds (≤ max_keep).
          - new_candidates: this round's top-K reranked captions (each with an
                            optional ``_rerank_reason`` field attached).
        Returns the new evidence memory (≤ max_keep, ordered as the LLM ranked
        them, falling back to memory order on parse failure).
        """
        if not current_memory and not new_candidates:
            return []
        if not current_memory:
            return new_candidates[:max_keep]

        pool: List[dict] = list(current_memory) + list(new_candidates)

        def _line(i: int, cap: dict, tag: str) -> str:
            reason = (cap.get("_rerank_reason") or "").strip()
            text = (cap.get("text") or cap.get("caption", "")).replace("\n", " ")
            base = (f"[{i}] ({tag}) {cap.get('date','')} "
                    f"{cap.get('start_time','')}-{cap.get('end_time','')}  {text}")
            return base + (f"  — {reason}" if reason else "")

        cur_lines = [_line(i, cap, "memory")
                     for i, cap in enumerate(current_memory)]
        offset = len(current_memory)
        new_lines = [_line(offset + j, cap, "new")
                     for j, cap in enumerate(new_candidates)]

        user = (
            f"Question: {question}\n\n"
            f"CURRENT memory ({len(current_memory)}):\n"
            + ("\n".join(cur_lines) if cur_lines else "(empty)")
            + f"\n\nNEW candidates from this round ({len(new_candidates)}):\n"
            + ("\n".join(new_lines) if new_lines else "(none)")
            + f"\n\nPick up to {max_keep} indices in 0..{len(pool) - 1} to KEEP. "
              "Output ONLY: "
              '{"keep": [<int>, ...], "reason": "<one short sentence>"}'
        )

        try:
            raw = self.llm.generate([
                {"role": "system",
                 "content": self._EVIDENCE_SYSTEM.format(max_keep=max_keep)},
                {"role": "user", "content": user},
            ])
        except Exception as exc:
            logger.warning("[Memory] curation LLM call failed (%s) — "
                           "falling back to new + recent memory", exc)
            return (new_candidates + current_memory)[:max_keep]

        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        i, j = text.find("{"), text.rfind("}")
        kept: List[dict] = []
        reason_summary = ""
        try:
            obj = json.loads(text[i:j + 1]) if i >= 0 and j > i else {}
            indices = obj.get("keep") or []
            reason_summary = str(obj.get("reason", "")).strip()
            seen: set = set()
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(pool) and idx not in seen:
                    seen.add(idx)
                    kept.append(pool[idx])
                if len(kept) >= max_keep:
                    break
        except Exception as exc:
            logger.warning("[Memory] curation parse failed (%s) — "
                           "falling back to new + recent memory", exc)
        if not kept:
            logger.warning("[Memory] empty curation result — falling back")
            kept = (new_candidates + current_memory)[:max_keep]

        # Log the curation outcome: kept entries (timestamp + GT/Near-GT marker
        # + reason), and the count of dropped ones. No caption text.
        kept_ids = {id(c) for c in kept}
        n_dropped = sum(1 for c in pool if id(c) not in kept_ids)
        logger.info(
            "[Memory] curated → keep %d / drop %d (was %d memory + %d new)%s",
            len(kept), n_dropped, len(current_memory), len(new_candidates),
            f"  — {reason_summary}" if reason_summary else "",
        )
        for rank, cap in enumerate(kept, start=1):
            r = (cap.get("_rerank_reason") or "").strip()
            date  = cap.get("date", "")
            stime = cap.get("start_time", "")
            etime = cap.get("end_time", "")
            line  = "[Memory] #%02d [%s %s-%s]%s" % (
                rank, date, stime, etime,
                f"  — {r}" if r else "",
            )
            logger.info(self._mark_gt(line, date, stime, etime, target_abs))
        return kept

    def _format_evidence(self, rounds: List[SearchRound], target_abs: Optional[int] = None) -> str:
        """Format accumulated search results as a readable block.
        Nodes that overlap with target_abs are marked with *** GT ***."""
        if not rounds:
            return "(no evidence retrieved yet)"

        from agent.search import _abs_ts as _ts

        def _overlaps(node: dict) -> bool:
            if target_abs is None or node.get("level") not in ("action", "activity", "conversation"):
                return False
            try:
                s = _ts(node["date"], node["start_time"])
                e = _ts(node["date"], node["end_time"])
                return s <= target_abs <= e
            except Exception:
                return False

        lines = []
        for r in rounds:
            lines.append(f"\n[Round {r.round_num} | act_q={r.activity_query!r} | action_q={r.action_query!r}]")
            for label, hits in (("activity", r.activity_hits),
                                ("conversation", r.conversation_hits),
                                ("action", r.action_hits)):
                if not hits:
                    continue
                lines.append(f"  -- {label} hits --")
                for h in hits:
                    marker = "  *** GT ***" if _overlaps(h) else ""
                    lines.append(
                        f"  [{h['date']} {h['start_time']}-{h['end_time']}] {h['text']}{marker}"
                    )
            if r.raw_captions:
                lines.append("  -- raw captions (chronological) --")
                sorted_caps = sorted(
                    r.raw_captions,
                    key=lambda c: _ts(c.get("date", "DAY1"), c.get("start_time", "0")) if c.get("date") else 0,
                )
                for cap in sorted_caps:
                    ts = f"{cap.get('date','')} {cap.get('start_time','')}–{cap.get('end_time','')}"
                    text = cap.get("text") or cap.get("caption", "")
                    marker = "  *** GT ***" if _overlaps(cap) else ""
                    lines.append(f"    [{ts}] {text}{marker}")
        return "\n".join(lines)

    def _build_round_user_msg(
        self,
        question: str,
        before_date: str,
        before_time: str,
        rounds: List[SearchRound],
        round_num: int,
        evidence_memory: List[dict],
        target_abs: Optional[int] = None,
    ) -> str:
        from agent.search import _abs_ts as _ts

        history_lines = []
        for r in rounds:
            history_lines.append(f"\n### Round {r.round_num}")
            history_lines.append(f"query: {r.action_query or r.activity_query}")
            if r.activity_hits:
                history_lines.append("Retrieved (activity level):")
                for h in r.activity_hits:
                    history_lines.append(f"  [{h['date']} {h['start_time']}-{h['end_time']}] {h['text']}")
            if r.conversation_hits:
                history_lines.append("Retrieved (conversation level):")
                for h in r.conversation_hits:
                    history_lines.append(f"  [{h['date']} {h['start_time']}-{h['end_time']}] {h['text']}")
            if r.action_hits:
                history_lines.append("Retrieved (action level):")
                for h in r.action_hits:
                    history_lines.append(f"  [{h['date']} {h['start_time']}-{h['end_time']}] {h['text']}")

        history = "\n".join(history_lines) if history_lines else "(none)"

        # Evidence memory: the curated working set (up to _EVIDENCE_MEMORY_K),
        # carried across rounds. Shown chronologically so the agent can reason
        # over the timeline.
        memory_sorted = sorted(
            evidence_memory,
            key=lambda c: _ts(c.get("date", "DAY1"), c.get("start_time", "0"))
                          if c.get("date") else 0,
        )
        cap_lines: List[str] = []
        for c in memory_sorted:
            base = (f"  [{c.get('date','')} {c.get('start_time','')}-"
                    f"{c.get('end_time','')}] "
                    f"{c.get('text') or c.get('caption','')}")
            reason = (c.get("_rerank_reason") or "").strip()
            cap_lines.append(base + (f"  — {reason}" if reason else ""))
        captions_block = (
            f"\n\nEvidence memory ({len(memory_sorted)}/"
            f"{self._EVIDENCE_MEMORY_K}, chronological):\n"
            + ("\n".join(cap_lines) if cap_lines else "  (empty)")
        )

        return (
            f"Question: {question}\n"
            f"(All memories before {before_date} {before_time})\n\n"
            f"Round History:\n{history}"
            f"{captions_block}\n\n"
            f"Round {round_num}/{self.max_rounds}: decide to search or answer."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        before_date: Optional[str] = None,
        before_time: Optional[str] = None,
        target_time: Optional[tuple] = None,  # (date, time) for GT marking in logs
        choices: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the ReAct loop, with up to _MAX_QUESTION_RETRIES restarts if the
        agent collects no evidence at all.

        `choices` (optional): the multiple-choice options for the question, as
        {"A": "...", "B": "...", "C": "...", "D": "..."}. When given, they are
        plumbed into the caption reranker so it can prefer captions that
        directly support or refute one of the listed choices.
        """
        last_result: Dict[str, Any] = {"activities": [], "raw_captions": [], "trace": []}
        for q_attempt in range(1, _MAX_QUESTION_RETRIES + 2):  # +1 retries → +2 to range
            result = self._retrieve_once(
                question=question,
                before_date=before_date,
                before_time=before_time,
                target_time=target_time,
                choices=choices,
            )
            n_caps = len(result.get("raw_captions", []))
            n_acts = len(result.get("activities", []))
            if n_caps > 0 or n_acts > 0:
                if q_attempt > 1:
                    logger.info("[Agent] Question retry %d succeeded with %d captions, %d activities",
                                q_attempt, n_caps, n_acts)
                return result
            last_result = result
            if q_attempt <= _MAX_QUESTION_RETRIES:
                logger.warning("[Agent] No evidence collected on attempt %d/%d — retrying full ReAct loop",
                               q_attempt, _MAX_QUESTION_RETRIES + 1)
        logger.warning("[Agent] All %d retrieve attempts produced no evidence — returning empty result",
                       _MAX_QUESTION_RETRIES + 1)
        return last_result

    # ------------------------------------------------------------------
    # Claude tool-use + interleaved-thinking path
    # ------------------------------------------------------------------
    def _retrieve_with_tools_once(
        self,
        question: str,
        before_date: Optional[str] = None,
        before_time: Optional[str] = None,
        target_time: Optional[tuple] = None,
        choices: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Claude-only: run the agent as a native tool-use loop with interleaved
        thinking. Returns the same dict shape as `_retrieve_once`, plus an
        ``answer`` field carrying the letter the agent submitted via the
        ``answer`` tool.
        """
        from agent.search import _abs_ts as _ts, _parse_target_window

        target_abs: Optional[int] = None
        if target_time:
            try:
                target_abs = _ts(target_time[0], target_time[1])
            except Exception:
                pass

        # ----- Initial user message (question + choices + current time) ------
        choices = choices or {}
        # Coerce values to str first — some datasets store numeric choices (e.g. a
        # count answer "20") as ints, which have no .strip(). None → "" (dropped).
        choice_lines = [f"  {k}) {('' if v is None else str(v)).strip()}"
                        for k, v in sorted(choices.items())
                        if ('' if v is None else str(v)).strip()]
        days = sorted(getattr(self._search, "_day_start", {}).keys()) \
               if hasattr(self._search, "_day_start") else []
        span_line = (f"Dataset spans DAY{days[0]} to DAY{days[-1]} "
                     f"(recorded hours per day vary)." if days else "")
        user_text = (
            f"Question: {question}\n"
            + ("Choices:\n" + "\n".join(choice_lines) + "\n" if choice_lines else "")
            + f"Current time: {(before_date or '')} {(before_time or '')}\n"
            + (span_line + "\n" if span_line else "")
            + "\nUse the tools to gather evidence and then call `answer`."
        )

        # ----- State --------------------------------------------------------
        pool: Dict[int, dict] = {}            # eid → caption-like dict
        cap_to_id: Dict[Any, int] = {}        # caption key → eid
        memory: List[int] = []                # current working memory (ordered)
        next_eid: List[int] = [1]
        tool_calls: List[dict] = []
        final_answer = "?"
        final_reasoning = ""
        n_search_calls = 0
        retrieval_done = False  # set True when the agent calls answer()

        def _gt_marker(date: str, stime: str, etime: str) -> str:
            """For LOG lines only — never include in tool result text sent to Claude."""
            if target_abs is None:
                return ""
            try:
                cs = _ts(date, stime); ce = _ts(date, etime)
            except Exception:
                return ""
            if cs <= target_abs <= ce:
                return "  *** GT ***"
            if min(abs(target_abs - cs), abs(target_abs - ce)) <= 5 * 60:
                return "  ** Next to GT **"
            return ""

        def _fmt_item(date: str, stime: str, etime: str, text: str) -> str:
            """Format one caption line for the tool result (NO GT marker)."""
            return f"  [{date} {stime}-{etime}] {text}"

        def _add_evidence(item: dict) -> Tuple[int, bool]:
            """Returns (evidence_id, was_new). Dedupes by (date, start, end, text)."""
            key = (item.get("date", ""), item.get("start_time", ""),
                   item.get("end_time", ""), (item.get("text") or "").strip())
            if key in cap_to_id:
                return cap_to_id[key], False
            eid = next_eid[0]
            next_eid[0] += 1
            pool[eid] = dict(item)
            cap_to_id[key] = eid
            return eid, True

        def _run_search(level: str, query: str, time_query: Optional[str]) -> str:
            nonlocal n_search_calls
            n_search_calls += 1
            target_window = _parse_target_window(time_query) if time_query else None
            lam = {"action":       self.lambda_action,
                   "activity":     self.lambda_activity,
                   "conversation": self.lambda_conversation}.get(level)
            level_top_k = _RETRIEVAL_TOP_K.get(level, 10)
            try:
                hits = self._search.search(
                    query=query, level=level, top_k=level_top_k,
                    before_date=before_date, before_time=before_time,
                    target_window=target_window, decay_lambda=lam,
                )
            except Exception as exc:
                return f"search_{level} failed: {exc}"

            added_ids: List[int] = []
            for h in hits:
                eid, was_new = _add_evidence(h)
                if eid not in memory:
                    memory.append(eid)
                if was_new:
                    added_ids.append(eid)

            tq_str = time_query or "(none)"
            log_level = "action_speech" if level == "action" else level
            logger.info(
                "[Tool Call] search_%s query=%r time_query=%s → %d hits (%d new in memory; mem=%d)",
                log_level, query, tq_str, len(hits), len(added_ids), len(memory),
            )
            lines = [f"search_{log_level}: query={query!r}  time_query={tq_str}  "
                     f"hits={len(hits)} new={len(added_ids)} memory={len(memory)}"]
            lines.append("--- This call returned ---")
            for h in hits:
                key = (h.get("date", ""), h.get("start_time", ""),
                       h.get("end_time", ""), (h.get("text") or "").strip())
                eid = cap_to_id.get(key, -1)
                date, stime, etime = h.get("date", ""), h.get("start_time", ""), h.get("end_time", "")
                text = (h.get("text") or "").replace("\n", " ")
                result_line = f"  [E{eid}] {_fmt_item(date, stime, etime, text)}"
                lines.append(result_line)
                logger.info("[Tool] %s",
                            self._mark_gt(result_line.strip(), date, stime, etime, target_abs))

            # Full evidence list: every item currently in working memory, sorted
            # chronologically so Claude can read the timeline straight through.
            mem_sorted = sorted(
                memory,
                key=lambda i: _ts(pool[i].get("date", "DAY1"),
                                  pool[i].get("start_time", "0"))
                              if pool.get(i, {}).get("date") else 0,
            )
            lines.append(f"--- Full evidence memory ({len(mem_sorted)} items) ---")
            for eid in mem_sorted:
                c = pool.get(eid, {})
                date, stime, etime = c.get("date", ""), c.get("start_time", ""), c.get("end_time", "")
                text = (c.get("text") or "").replace("\n", " ")
                lines.append(f"  [E{eid}] {_fmt_item(date, stime, etime, text)}")
            return "\n".join(lines)

        def _run_curate(keep: List[int]) -> str:
            keep_set = {int(i) for i in keep if isinstance(i, (int, str)) and str(i).lstrip('-').isdigit()}
            before_n = len(memory)
            new_mem = [eid for eid in memory if eid in keep_set]
            dropped = before_n - len(new_mem)
            memory.clear()
            memory.extend(new_mem)
            logger.info("[Tool Call] curate_evidence → keep=%d drop=%d (mem %d→%d)",
                        len(new_mem), dropped, before_n, len(new_mem))
            # Sort the kept memory chronologically for both the log and the tool result.
            kept_sorted = sorted(
                new_mem,
                key=lambda i: _ts(pool[i].get("date", "DAY1"),
                                  pool[i].get("start_time", "0"))
                              if pool.get(i, {}).get("date") else 0,
            )
            lines = [f"curate_evidence: kept {len(new_mem)} / {before_n} "
                     f"(dropped {dropped})",
                     f"--- Evidence remaining ({len(new_mem)} items, chronological) ---"]
            for eid in kept_sorted:
                c = pool.get(eid, {})
                date, stime, etime = c.get("date", ""), c.get("start_time", ""), c.get("end_time", "")
                text = (c.get("text") or "").replace("\n", " ")
                result_line = f"  [E{eid}] {_fmt_item(date, stime, etime, text)}"
                lines.append(result_line)
                logger.info("[Tool] %s",
                            self._mark_gt(result_line.strip(), date, stime, etime, target_abs))
            return "\n".join(lines)

        def _expand_memory_to_raw_captions() -> List[dict]:
            """Follow each working-memory item back to its underlying raw 30-sec
            captions (via `raw_caption_indices`); fall back to the item itself
            when it IS a raw caption. Deduped + sorted chronologically.
            """
            raw_captions = getattr(self._search, "_raw_captions", []) or []
            seen_cap_idxs: set = set()
            linked: List[dict] = []
            for eid in memory:
                item = pool.get(eid, {})
                cap_indices = item.get("raw_caption_indices") or []
                added_any = False
                if cap_indices:
                    for idx in cap_indices:
                        if idx not in seen_cap_idxs and idx < len(raw_captions):
                            seen_cap_idxs.add(idx)
                            linked.append(raw_captions[idx])
                            added_any = True
                if not added_any:
                    key = (item.get("date", ""), item.get("start_time", ""),
                           (item.get("text") or "").strip())
                    if key not in seen_cap_idxs:
                        seen_cap_idxs.add(key)
                        linked.append(item)
            linked.sort(
                key=lambda c: _ts(c.get("date", "DAY1"), c.get("start_time", "0"))
            )
            return linked

        def _run_answer() -> str:
            """Tool handler for `answer()`: signal end of retrieval. The
            answering agent (a SEPARATE LLM call) will run AFTER the tool-use
            loop exits and use the curated evidence to produce the final
            reasoning + answer letter.
            """
            if not memory:
                return ("answer() called but working memory is empty. "
                        "Search and curate evidence first, then call answer().")
            logger.info("[Tool Call] answer() — retrieval done; handing off to "
                        "answering agent (%d evidence items in memory)",
                        len(memory))
            return ("Retrieval complete. The ANSWERING AGENT will now read your "
                    "curated evidence and produce the final answer.")

        # ----- Tool-use loop ------------------------------------------------
        import time as _time
        question_t0  = _time.time()
        tok_in_total  = 0
        tok_out_total = 0
        tok_cache_in  = 0   # cache_read_input_tokens (cheap cache hits)
        tok_cache_w   = 0   # cache_creation_input_tokens (writing the cache)
        round_stats: List[dict] = []

        _cfg_retrieval = MODEL_CONFIG["retrieval_agent"]
        _use_curate_tool    = _cfg_retrieval.get("retrieval_curate_tool", True)
        _parallel_tool_calls = _cfg_retrieval.get("parallel_tool_calls", True)
        _retrieval_tools    = _TOOLS_SCHEMA if _use_curate_tool else _TOOLS_SCHEMA_NO_CURATE
        _retrieval_system   = _build_tool_use_system(_use_curate_tool)

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_text}]
        for it in range(_MAX_TOOL_ITERATIONS):
            # FINAL-ROUND push: on the last allowed iteration, inject a strong
            # user message forcing the agent to wrap up. Goes BEFORE the API
            # call so the model sees it as the most recent instruction.
            is_final_round = (it == _MAX_TOOL_ITERATIONS - 1)
            if is_final_round:
                logger.warning(
                    "[Agent] FINAL ROUND (%d/%d) — forcing answer",
                    it + 1, _MAX_TOOL_ITERATIONS,
                )
                final_round_msg = (
                    f"FINAL ROUND ({it + 1}/{_MAX_TOOL_ITERATIONS}). "
                    "You have used all your search budget. THIS TURN you "
                    "MUST: (1) call `curate_evidence` to commit your final "
                    "evidence selection, then (2) call `answer()` (no "
                    "arguments) — the harness will run the QA step on your "
                    "curated captions. Do NOT issue any more search_* "
                    "calls — they will be ignored."
                ) if _use_curate_tool else (
                    f"FINAL ROUND ({it + 1}/{_MAX_TOOL_ITERATIONS}). "
                    "You have used all your search budget. THIS TURN you "
                    "MUST call `answer()` to end the retrieval phase. "
                    "Do NOT issue any more search_* calls."
                )
                messages.append({"role": "user", "content": final_round_msg})

            round_t0 = _time.time()
            try:
                response = self.llm.create_with_tools(
                    messages=messages,
                    system=_retrieval_system,
                    tools=_retrieval_tools,
                    parallel_tool_calls=_parallel_tool_calls,
                )
            except Exception as exc:
                logger.warning("[Agent] tool-use call failed at iter %d: %s", it, exc)
                break
            round_latency = _time.time() - round_t0

            usage = getattr(response, "usage", None)
            tin   = int(getattr(usage, "input_tokens",                0) or 0) if usage else 0
            tout  = int(getattr(usage, "output_tokens",               0) or 0) if usage else 0
            tcr   = int(getattr(usage, "cache_read_input_tokens",     0) or 0) if usage else 0
            tcw   = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
            tok_in_total  += tin
            tok_out_total += tout
            tok_cache_in  += tcr
            tok_cache_w   += tcw
            round_stats.append({
                "round":           it + 1,
                "latency_sec":     round_latency,
                "input_tokens":    tin,
                "output_tokens":   tout,
                "cache_read":      tcr,
                "cache_write":     tcw,
            })
            logger.info(
                "[Round %d/%d] latency=%.2fs  in=%d  out=%d  cache_read=%d  "
                "cache_write=%d",
                it + 1, _MAX_TOOL_ITERATIONS, round_latency, tin, tout, tcr, tcw,
            )

            assistant_content: List[Any] = list(response.content)
            messages.append({"role": "assistant", "content": assistant_content})

            logger.info("----- Retrieval Agent Round %d -------", it + 1)
            tool_result_blocks: List[Dict[str, Any]] = []
            saw_answer = False
            # Track the LAST tool the agent called in this turn so we can enforce
            # "curate_evidence must immediately precede answer".
            for block in assistant_content:
                btype = getattr(block, "type", None)
                if btype == "thinking":
                    txt = (getattr(block, "thinking", "") or "").strip()
                    if txt:
                        logger.info("\033[90m[Agent thinking] %s\033[0m",
                                    txt.replace("\n", " ")[:400])
                elif btype == "text":
                    txt = (getattr(block, "text", "") or "").strip()
                    if txt:
                        logger.info("[Agent text] %s", txt[:400])
                elif btype == "tool_use":
                    tname = block.name
                    tinput = dict(getattr(block, "input", {}) or {})
                    tuid = block.id
                    tool_calls.append({"tool": tname, "input": tinput, "iter": it})

                    # Final-round gate: on the last iteration, reject any
                    # search_* tool so the agent is forced to curate + answer.
                    if is_final_round and tname.startswith("search_"):
                        logger.warning(
                            "[Tool Call] %s rejected on FINAL ROUND — "
                            "search budget exhausted; must curate + answer now.",
                            tname,
                        )
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tuid,
                            "content": (
                                f"`{tname}` is NOT allowed in the final round. "
                                "Your search budget is exhausted. Call "
                                "`curate_evidence` to commit your final "
                                "evidence selection, then call `answer()` — "
                                "the harness will run the QA step on your "
                                "curated captions."
                            ),
                        })
                        continue

                    if tname == "search_action_speech":
                        out = _run_search("action",
                                          tinput.get("query", ""),
                                          tinput.get("time_query"))
                    elif tname == "search_activity":
                        out = _run_search("activity",
                                          tinput.get("query", ""),
                                          tinput.get("time_query"))
                    elif tname == "search_conversation":
                        out = _run_search("conversation",
                                          tinput.get("query", ""),
                                          tinput.get("time_query"))
                    elif tname == "curate_evidence":
                        out = _run_curate(tinput.get("keep") or [])
                    elif tname == "answer":
                        # Soft enforcement (curate_evidence path only): the last
                        # prior tool must be curate_evidence before answer() is
                        # accepted. Skipped when curate_evidence is not in the
                        # tool schema (e.g. Qwen config).
                        prior_tools = [c["tool"] for c in tool_calls[:-1]]
                        last_prior = prior_tools[-1] if prior_tools else None
                        already_warned = any(
                            c["tool"] == "answer" and c.get("rejected")
                            for c in tool_calls[:-1]
                        )
                        if (_use_curate_tool
                                and last_prior != "curate_evidence"
                                and not already_warned):
                            tool_calls[-1]["rejected"] = "missing_curate"
                            logger.warning(
                                "[Tool Call] answer rejected — last tool was %r, "
                                "but the workflow requires curate_evidence "
                                "immediately before answer. Asking agent to "
                                "commit final selection first.",
                                last_prior,
                            )
                            out = ("Cannot accept `answer()` yet: the workflow "
                                   "requires you to call `curate_evidence` with "
                                   "your FINAL evidence selection IMMEDIATELY "
                                   "before `answer()`. Please call "
                                   "curate_evidence now, then call answer() "
                                   "again — the harness will run the QA step "
                                   "on the captions you committed.")
                            saw_answer = False
                        else:
                            out = _run_answer()
                            saw_answer = True
                    else:
                        out = f"Unknown tool: {tname}"
                        logger.warning("[Tool] unknown tool: %s", tname)
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tuid,
                        "content": out,
                    })

            if saw_answer:
                # End of retrieval — append the answer() tool result so the
                # transcript stays balanced, then break. The answering agent
                # runs AFTER the loop with the curated memory.
                if tool_result_blocks:
                    messages.append({"role": "user", "content": tool_result_blocks})
                retrieval_done = True
                break
            if not tool_result_blocks:
                # No tool calls and no answer → assistant chose to stop. Exit loop.
                logger.info("[Agent] no tool calls this turn — ending loop")
                break
            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            logger.warning("[Agent] tool-use loop hit max iterations (%d)",
                           _MAX_TOOL_ITERATIONS)

        kept_caps = [pool[eid] for eid in memory if eid in pool]

        # ----- Answering agent (dedup → curate → expand → QA) ----------
        ans_latency  = 0.0
        ans_tok_in   = 0
        ans_tok_out  = 0
        ans_cache_r  = 0
        ans_cache_w  = 0
        try:
            (ans_text, ans_letter, ans_latency,
             ans_tok_in, ans_tok_out, ans_cache_r, ans_cache_w,
             ans_deduped, ans_curated) = \
                self._run_answering_agent(
                    question=question,
                    choices=choices or {},
                    evidence_items=kept_caps,
                    before_date=before_date,
                    before_time=before_time,
                    target_abs=target_abs,
                )
            final_reasoning = ans_text
            final_answer    = ans_letter
            final_deduped   = ans_deduped
            final_curated   = ans_curated
        except Exception as exc:
            logger.warning("[Answering Agent] call failed: %s", exc)
            final_reasoning = ""
            final_answer    = "?"
            final_deduped   = []
            final_curated   = []

        question_latency = _time.time() - question_t0
        n_rounds = len(round_stats)

        # Retrieval-agent totals come from the per-round stats already accrued.
        retr_total_tokens = tok_in_total + tok_out_total + tok_cache_in + tok_cache_w
        retr_latency = sum(s["latency_sec"] for s in round_stats) if round_stats else 0.0

        # Answering-agent totals come from the single call above.
        ans_total_tokens = ans_tok_in + ans_tok_out + ans_cache_r + ans_cache_w

        combined_in       = tok_in_total  + ans_tok_in
        combined_out      = tok_out_total + ans_tok_out
        combined_cache_r  = tok_cache_in  + ans_cache_r
        combined_cache_w  = tok_cache_w   + ans_cache_w
        combined_total    = retr_total_tokens + ans_total_tokens

        logger.info(
            "[Retrieval Agent total] rounds=%d  latency=%.2fs  in=%d  out=%d  "
            "cache_read=%d  cache_write=%d  total=%d",
            n_rounds, retr_latency,
            tok_in_total, tok_out_total, tok_cache_in, tok_cache_w,
            retr_total_tokens,
        )
        logger.info(
            "[Answering Agent total] latency=%.2fs  in=%d  out=%d  "
            "cache_read=%d  cache_write=%d  total=%d",
            ans_latency, ans_tok_in, ans_tok_out, ans_cache_r, ans_cache_w,
            ans_total_tokens,
        )
        logger.info(
            "[Question total] retrieval+answering  wall=%.2fs  in=%d  out=%d  "
            "cache_read=%d  cache_write=%d  total=%d",
            question_latency, combined_in, combined_out,
            combined_cache_r, combined_cache_w, combined_total,
        )

        return {
            "activities":       [],
            "conversations":    [],
            "raw_captions":     kept_caps,
            "deduped_caps":     final_deduped,
            "curated_caps":     final_curated,
            "trace":            tool_calls,
            "answer":           final_answer,
            "answer_reasoning": final_reasoning,
            "usage": {
                # Wall-clock around the whole retrieve_with_tools_once pass.
                "rounds":        n_rounds,
                "latency_sec":   question_latency,
                # Combined retrieval + answering totals (legacy fields).
                "input_tokens":  combined_in,
                "output_tokens": combined_out,
                "cache_read":    combined_cache_r,
                "cache_write":   combined_cache_w,
                "per_round":     round_stats,
                # Explicit per-agent breakdown.
                "retrieval_agent": {
                    "rounds":        n_rounds,
                    "latency_sec":   retr_latency,
                    "input_tokens":  tok_in_total,
                    "output_tokens": tok_out_total,
                    "cache_read":    tok_cache_in,
                    "cache_write":   tok_cache_w,
                    "total_tokens":  retr_total_tokens,
                },
                "answering_agent": {
                    "latency_sec":   ans_latency,
                    "input_tokens":  ans_tok_in,
                    "output_tokens": ans_tok_out,
                    "cache_read":    ans_cache_r,
                    "cache_write":   ans_cache_w,
                    "total_tokens":  ans_total_tokens,
                },
            },
        }

    # ------------------------------------------------------------------
    # GPT (OpenAI Responses API) tool-use path
    #
    # GPT and Anthropic are different systems, so this is a SEPARATE harness
    # from `_retrieve_with_tools_once` above. It speaks the Responses API
    # natively — an `input` list of items, `function_call` / `function_call_output`
    # items, OpenAI-flat tool schemas — and re-feeds the whole `response.output`
    # each turn (`input_list += response.output`) so GPT's reasoning items stay
    # paired with the function calls they produced. The tool *business logic*
    # (search / curate / answer over the evidence pool) is identical to the
    # Anthropic loop; only the LLM protocol differs.
    # ------------------------------------------------------------------
    def _retrieve_with_tools_once_gpt(
        self,
        question: str,
        before_date: Optional[str] = None,
        before_time: Optional[str] = None,
        target_time: Optional[tuple] = None,
        choices: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """GPT-only: run the agent as a native OpenAI Responses tool-use loop.
        Returns the same dict shape as `_retrieve_with_tools_once`.
        """
        import json as _json
        import time as _time
        from agent.search import _abs_ts as _ts, _parse_target_window

        target_abs: Optional[int] = None
        if target_time:
            try:
                target_abs = _ts(target_time[0], target_time[1])
            except Exception:
                pass

        # ----- Initial user message (question + choices + current time) ------
        choices = choices or {}
        choice_lines = [f"  {k}) {('' if v is None else str(v)).strip()}"
                        for k, v in sorted(choices.items())
                        if ('' if v is None else str(v)).strip()]
        days = sorted(getattr(self._search, "_day_start", {}).keys()) \
               if hasattr(self._search, "_day_start") else []
        span_line = (f"Dataset spans DAY{days[0]} to DAY{days[-1]} "
                     f"(recorded hours per day vary)." if days else "")
        user_text = (
            f"Question: {question}\n"
            + ("Choices:\n" + "\n".join(choice_lines) + "\n" if choice_lines else "")
            + f"Current time: {(before_date or '')} {(before_time or '')}\n"
            + (span_line + "\n" if span_line else "")
            + "\nUse the tools to gather evidence and then call `answer`."
        )

        # ----- State (identical to the Anthropic loop) ----------------------
        pool: Dict[int, dict] = {}
        cap_to_id: Dict[Any, int] = {}
        memory: List[int] = []
        next_eid: List[int] = [1]
        tool_calls: List[dict] = []
        n_search_calls = 0

        def _fmt_item(date: str, stime: str, etime: str, text: str) -> str:
            return f"  [{date} {stime}-{etime}] {text}"

        def _add_evidence(item: dict) -> Tuple[int, bool]:
            key = (item.get("date", ""), item.get("start_time", ""),
                   item.get("end_time", ""), (item.get("text") or "").strip())
            if key in cap_to_id:
                return cap_to_id[key], False
            eid = next_eid[0]
            next_eid[0] += 1
            pool[eid] = dict(item)
            cap_to_id[key] = eid
            return eid, True

        def _run_search(level: str, query: str, time_query: Optional[str]) -> str:
            nonlocal n_search_calls
            n_search_calls += 1
            target_window = _parse_target_window(time_query) if time_query else None
            lam = {"action":       self.lambda_action,
                   "activity":     self.lambda_activity,
                   "conversation": self.lambda_conversation}.get(level)
            level_top_k = _RETRIEVAL_TOP_K.get(level, 10)
            try:
                hits = self._search.search(
                    query=query, level=level, top_k=level_top_k,
                    before_date=before_date, before_time=before_time,
                    target_window=target_window, decay_lambda=lam,
                )
            except Exception as exc:
                return f"search_{level} failed: {exc}"

            added_ids: List[int] = []
            for h in hits:
                eid, was_new = _add_evidence(h)
                if eid not in memory:
                    memory.append(eid)
                if was_new:
                    added_ids.append(eid)

            tq_str = time_query or "(none)"
            log_level = "action_speech" if level == "action" else level
            logger.info(
                "[Tool Call] search_%s query=%r time_query=%s → %d hits (%d new in memory; mem=%d)",
                log_level, query, tq_str, len(hits), len(added_ids), len(memory),
            )
            lines = [f"search_{log_level}: query={query!r}  time_query={tq_str}  "
                     f"hits={len(hits)} new={len(added_ids)} memory={len(memory)}",
                     "--- This call returned ---"]
            for h in hits:
                key = (h.get("date", ""), h.get("start_time", ""),
                       h.get("end_time", ""), (h.get("text") or "").strip())
                eid = cap_to_id.get(key, -1)
                date, stime, etime = h.get("date", ""), h.get("start_time", ""), h.get("end_time", "")
                text = (h.get("text") or "").replace("\n", " ")
                result_line = f"  [E{eid}] {_fmt_item(date, stime, etime, text)}"
                lines.append(result_line)
                logger.info("[Tool] %s",
                            self._mark_gt(result_line.strip(), date, stime, etime, target_abs))
            mem_sorted = sorted(
                memory,
                key=lambda i: _ts(pool[i].get("date", "DAY1"), pool[i].get("start_time", "0"))
                              if pool.get(i, {}).get("date") else 0,
            )
            lines.append(f"--- Full evidence memory ({len(mem_sorted)} items) ---")
            for eid in mem_sorted:
                c = pool.get(eid, {})
                date, stime, etime = c.get("date", ""), c.get("start_time", ""), c.get("end_time", "")
                text = (c.get("text") or "").replace("\n", " ")
                lines.append(f"  [E{eid}] {_fmt_item(date, stime, etime, text)}")
            return "\n".join(lines)

        def _run_curate(keep: List[int]) -> str:
            keep_set = {int(i) for i in keep
                        if isinstance(i, (int, str)) and str(i).lstrip('-').isdigit()}
            before_n = len(memory)
            new_mem = [eid for eid in memory if eid in keep_set]
            dropped = before_n - len(new_mem)
            memory.clear()
            memory.extend(new_mem)
            logger.info("[Tool Call] curate_evidence → keep=%d drop=%d (mem %d→%d)",
                        len(new_mem), dropped, before_n, len(new_mem))
            kept_sorted = sorted(
                new_mem,
                key=lambda i: _ts(pool[i].get("date", "DAY1"), pool[i].get("start_time", "0"))
                              if pool.get(i, {}).get("date") else 0,
            )
            lines = [f"curate_evidence: kept {len(new_mem)} / {before_n} (dropped {dropped})",
                     f"--- Evidence remaining ({len(new_mem)} items, chronological) ---"]
            for eid in kept_sorted:
                c = pool.get(eid, {})
                date, stime, etime = c.get("date", ""), c.get("start_time", ""), c.get("end_time", "")
                text = (c.get("text") or "").replace("\n", " ")
                result_line = f"  [E{eid}] {_fmt_item(date, stime, etime, text)}"
                lines.append(result_line)
                logger.info("[Tool] %s",
                            self._mark_gt(result_line.strip(), date, stime, etime, target_abs))
            return "\n".join(lines)

        def _run_answer() -> str:
            if not memory:
                return ("answer() called but working memory is empty. "
                        "Search and curate evidence first, then call answer().")
            logger.info("[Tool Call] answer() — retrieval done; handing off to "
                        "answering agent (%d evidence items in memory)", len(memory))
            return ("Retrieval complete. The ANSWERING AGENT will now read your "
                    "curated evidence and produce the final answer.")

        # ----- Config + tools (GPT-native) ----------------------------------
        _cfg = MODEL_CONFIG["retrieval_agent"]
        _use_curate_tool = _cfg.get("retrieval_curate_tool", True)
        _tools = _TOOLS_SCHEMA_OPENAI if _use_curate_tool else _TOOLS_SCHEMA_OPENAI_NO_CURATE
        _system = _build_tool_use_system(_use_curate_tool)
        _effort = _cfg.get("effort", "medium")
        _max_out = int(_cfg.get("max_tokens", 4096))
        _model = getattr(self.llm, "model_name", None) or _cfg.get("model", "gpt-5.4")
        _client = getattr(self.llm, "client", None)

        # ----- Tool-use loop (Responses API `input` list) -------------------
        question_t0 = _time.time()
        tok_in_total = tok_out_total = tok_cache_in = 0
        round_stats: List[dict] = []
        retrieval_done = False

        # Running Responses input; we grow it over time exactly like the
        # openai function-calling example (input_list += response.output).
        input_list: List[Any] = [{"role": "user", "content": user_text}]

        for it in range(_MAX_TOOL_ITERATIONS):
            is_final_round = (it == _MAX_TOOL_ITERATIONS - 1)
            if is_final_round:
                logger.warning("[Agent-GPT] FINAL ROUND (%d/%d) — forcing answer",
                               it + 1, _MAX_TOOL_ITERATIONS)
                final_round_msg = (
                    f"FINAL ROUND ({it + 1}/{_MAX_TOOL_ITERATIONS}). You have used all "
                    "your search budget. THIS TURN you MUST: (1) call `curate_evidence` "
                    "to commit your final evidence selection, then (2) call `answer()` "
                    "(no arguments). Do NOT issue any more search_* calls."
                ) if _use_curate_tool else (
                    f"FINAL ROUND ({it + 1}/{_MAX_TOOL_ITERATIONS}). You have used all "
                    "your search budget. THIS TURN you MUST call `answer()` to end the "
                    "retrieval phase. Do NOT issue any more search_* calls."
                )
                input_list.append({"role": "user", "content": final_round_msg})

            round_t0 = _time.time()
            try:
                response = _client.responses.create(
                    model=_model,
                    input=input_list,
                    instructions=_system,
                    tools=_tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    reasoning={"effort": _effort, "summary": "auto"},
                    max_output_tokens=_max_out,
                )
            except Exception as exc:
                logger.warning("[Agent-GPT] responses.create failed at iter %d: %s", it, exc)
                break
            round_latency = _time.time() - round_t0

            usage = getattr(response, "usage", None)
            tin = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
            tout = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            tdet = getattr(usage, "input_tokens_details", None) if usage else None
            tcr = int(getattr(tdet, "cached_tokens", 0) or 0) if tdet else 0
            tok_in_total += tin
            tok_out_total += tout
            tok_cache_in += tcr
            round_stats.append({
                "round": it + 1, "latency_sec": round_latency,
                "input_tokens": tin, "output_tokens": tout,
                "cache_read": tcr, "cache_write": 0,
            })
            logger.info("[Round %d/%d] latency=%.2fs  in=%d  out=%d  cache_read=%d",
                        it + 1, _MAX_TOOL_ITERATIONS, round_latency, tin, tout, tcr)

            # Re-feed the FULL model output (reasoning + function_call + text)
            # so GPT's reasoning stays paired with its calls across turns.
            output_items = list(getattr(response, "output", []) or [])
            input_list += output_items

            logger.info("----- Retrieval Agent (GPT) Round %d -------", it + 1)
            saw_answer = False
            any_tool = False
            for item in output_items:
                itype = getattr(item, "type", None)
                if itype == "reasoning":
                    for s in (getattr(item, "summary", None) or []):
                        txt = (getattr(s, "text", "") or "").strip()
                        if txt:
                            logger.info("\033[90m[Agent thinking] %s\033[0m",
                                        txt.replace("\n", " ")[:400])
                elif itype == "message":
                    for c in (getattr(item, "content", None) or []):
                        txt = (getattr(c, "text", "") or "").strip()
                        if txt:
                            logger.info("[Agent text] %s", txt[:400])
                elif itype == "function_call":
                    any_tool = True
                    tname = getattr(item, "name", "") or ""
                    call_id = getattr(item, "call_id", "") or getattr(item, "id", "")
                    try:
                        tinput = _json.loads(getattr(item, "arguments", "") or "{}")
                    except Exception:
                        tinput = {}
                    if not isinstance(tinput, dict):
                        tinput = {}
                    tool_calls.append({"tool": tname, "input": tinput, "iter": it})

                    # Final-round gate: reject search_* so the agent curates + answers.
                    if is_final_round and tname.startswith("search_"):
                        logger.warning("[Tool Call] %s rejected on FINAL ROUND", tname)
                        out = (f"`{tname}` is NOT allowed in the final round. Call "
                               "`curate_evidence` then `answer()`.")
                    elif tname == "search_action_speech":
                        out = _run_search("action", tinput.get("query", ""), tinput.get("time_query"))
                    elif tname == "search_activity":
                        out = _run_search("activity", tinput.get("query", ""), tinput.get("time_query"))
                    elif tname == "search_conversation":
                        out = _run_search("conversation", tinput.get("query", ""), tinput.get("time_query"))
                    elif tname == "curate_evidence":
                        out = _run_curate(tinput.get("keep") or [])
                    elif tname == "answer":
                        # Soft enforcement: curate_evidence must immediately precede
                        # answer() (curate path only), mirroring the Anthropic loop.
                        prior_tools = [c["tool"] for c in tool_calls[:-1]]
                        last_prior = prior_tools[-1] if prior_tools else None
                        already_warned = any(c["tool"] == "answer" and c.get("rejected")
                                             for c in tool_calls[:-1])
                        if (_use_curate_tool and last_prior != "curate_evidence"
                                and not already_warned):
                            tool_calls[-1]["rejected"] = "missing_curate"
                            logger.warning("[Tool Call] answer rejected — last tool was %r, "
                                           "curate_evidence must come first.", last_prior)
                            out = ("Cannot accept `answer()` yet: call `curate_evidence` "
                                   "with your FINAL evidence selection IMMEDIATELY before "
                                   "`answer()`, then call answer() again.")
                        else:
                            out = _run_answer()
                            saw_answer = True
                    else:
                        out = f"Unknown tool: {tname}"
                        logger.warning("[Tool] unknown tool: %s", tname)

                    # EVERY function_call needs a matching function_call_output,
                    # else the next Responses call errors on the dangling call.
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out,
                    })

            if saw_answer:
                retrieval_done = True
                break
            if not any_tool:
                logger.info("[Agent-GPT] no tool calls this turn — ending loop")
                break
        else:
            logger.warning("[Agent-GPT] tool-use loop hit max iterations (%d)",
                           _MAX_TOOL_ITERATIONS)

        kept_caps = [pool[eid] for eid in memory if eid in pool]

        # ----- Answering agent (shared) ------------------------------------
        ans_latency = ans_tok_in = ans_tok_out = ans_cache_r = ans_cache_w = 0
        final_reasoning, final_answer = "", "?"
        final_deduped, final_curated = [], []
        try:
            (ans_text, ans_letter, ans_latency,
             ans_tok_in, ans_tok_out, ans_cache_r, ans_cache_w,
             ans_deduped, ans_curated) = self._run_answering_agent(
                question=question, choices=choices or {}, evidence_items=kept_caps,
                before_date=before_date, before_time=before_time, target_abs=target_abs,
            )
            final_reasoning, final_answer = ans_text, ans_letter
            final_deduped, final_curated = ans_deduped, ans_curated
        except Exception as exc:
            logger.warning("[Answering Agent] call failed: %s", exc)

        question_latency = _time.time() - question_t0
        n_rounds = len(round_stats)
        retr_total_tokens = tok_in_total + tok_out_total + tok_cache_in
        retr_latency = sum(s["latency_sec"] for s in round_stats) if round_stats else 0.0
        ans_total_tokens = ans_tok_in + ans_tok_out + ans_cache_r + ans_cache_w
        combined_in = tok_in_total + ans_tok_in
        combined_out = tok_out_total + ans_tok_out
        combined_cache_r = tok_cache_in + ans_cache_r
        combined_cache_w = ans_cache_w
        combined_total = retr_total_tokens + ans_total_tokens

        logger.info(
            "[Retrieval Agent (GPT) total] rounds=%d  latency=%.2fs  in=%d  out=%d  "
            "cache_read=%d  total=%d", n_rounds, retr_latency,
            tok_in_total, tok_out_total, tok_cache_in, retr_total_tokens)
        logger.info(
            "[Question total] retrieval+answering  wall=%.2fs  in=%d  out=%d  total=%d",
            question_latency, combined_in, combined_out, combined_total)

        return {
            "activities":       [],
            "conversations":    [],
            "raw_captions":     kept_caps,
            "deduped_caps":     final_deduped,
            "curated_caps":     final_curated,
            "trace":            tool_calls,
            "answer":           final_answer,
            "answer_reasoning": final_reasoning,
            "usage": {
                "rounds":        n_rounds,
                "latency_sec":   question_latency,
                "input_tokens":  combined_in,
                "output_tokens": combined_out,
                "cache_read":    combined_cache_r,
                "cache_write":   combined_cache_w,
                "per_round":     round_stats,
                "retrieval_agent": {
                    "rounds":        n_rounds,
                    "latency_sec":   retr_latency,
                    "input_tokens":  tok_in_total,
                    "output_tokens": tok_out_total,
                    "cache_read":    tok_cache_in,
                    "cache_write":   0,
                    "total_tokens":  retr_total_tokens,
                },
                "answering_agent": {
                    "latency_sec":   ans_latency,
                    "input_tokens":  ans_tok_in,
                    "output_tokens": ans_tok_out,
                    "cache_read":    ans_cache_r,
                    "cache_write":   ans_cache_w,
                    "total_tokens":  ans_total_tokens,
                },
            },
        }

    # ------------------------------------------------------------------
    # Answering agent — separate LLM call invoked after the retrieval
    # tool-use loop terminates. Takes the curated raw captions + question +
    # choices and returns (reasoning_text, predicted_letter).
    # ------------------------------------------------------------------
    _ANSWERING_SYSTEM = (
        "You are an egocentric memory assistant. The person wearing the "
        "camera is asking about their own past experiences. You are given a "
        "curated set of 30-sec RAW captions (with timestamps) that a "
        "retrieval agent already selected as the evidence for this question.\n"
        "\n"
        "Use ONLY these captions to answer the multiple-choice question. Do "
        "not invent details that are not in the captions.\n"
        "\n"
        "Follow these two grounding steps before committing to a letter:\n"
        "  1. GROUND every element of the question in the captions. For each "
        "key noun, person, object, or action in the question, find the "
        "caption(s) that actually describe it. If a caption does not mention "
        "it, treat it as not happening.\n"
        "  2. GROUND the time qualifier (if any). Words like 'last', "
        "'yesterday', 'recently', 'first', 'before', 'after', 'the day before "
        "yesterday' must be resolved against the caption timestamps and the "
        "memory query point — NOT against general daily schedules.\n"
        "\n"
        "Output format (STRICT):\n"
        "  - A few sentences of REASONING that cite the specific captions "
        "(by their timestamp) and explain why they support the chosen answer "
        "and rule out the others.\n"
        "  - A single final line: `Answer: <letter>` where <letter> is A, B, "
        "C, or D."
    )

    _CURATION_SYSTEM = """\
        You are an evidence curator for a personal egocentric memory system.
        You are given a pool of candidate evidence items retrieved for a multiple-choice question.
        Your job is to select the most relevant evidence items for the final QA agent.

        # Selection criteria
        Keep between 5 and 15 items. Do NOT over-curate: leaving only 1 or 2 items
        risks discarding related evidence the QA agent needs to answer correctly.
        Prefer items that:
        - Directly mention the key entity / action / object / person the question asks about.
        - Help resolve time qualifiers ("last", "first", "yesterday", "recently") unambiguously.
        - Distinguish between the answer choices — favour items that support one choice over another.

        Drop items that are:
        - Redundant (same event repeated → keep the most informative one).
        - Clearly off-topic or unrelated to the question and choices.

        # Curation policy (apply based on question wording)
        - "usually" / "often" / "typically": curate a DIVERSE set spanning the full timeline.
        - "last" / "most recent" / "latest": curate the LATEST timestamps.
        - "first" / "earliest": curate the EARLIEST timestamps.
        - "before X" / "after X": curate items in the relevant time window relative to event X.

        Call `curate_evidence` once with your final selection, then stop.
        """

    _CURATE_ONLY_SCHEMA: List[Dict[str, Any]] = [
        {
            "name": "curate_evidence",
            "description": (
                "Commit your final evidence selection. Pass the IDs of the "
                "items to keep (5–15). Do not over-curate: keeping fewer than "
                "5 items risks discarding evidence the QA agent needs."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "keep": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Evidence IDs to keep.",
                    },
                },
                "required": ["keep"],
            },
        },
    ]

    def _run_evidence_curation(
        self,
        question: str,
        choices: Dict[str, str],
        evidence: List[dict],
    ) -> Tuple[List[dict], int, int, int, int]:
        """Tool-call LLM call (with thinking) that selects ≤15 items from the
        evidence pool via the curate_evidence tool.
        Returns (curated_items, tok_in, tok_out, tok_cache_r, tok_cache_w).
        Falls back to returning all items on any failure.
        """
        id_to_item: Dict[int, dict] = {i + 1: item for i, item in enumerate(evidence)}

        lines = [f"Question: {question}"]
        for ltr in ("A", "B", "C", "D"):
            txt = (choices.get(ltr) or "").strip()
            if txt:
                lines.append(f"{ltr}) {txt}")
        lines.append(f"\n--- CANDIDATE EVIDENCE ({len(evidence)} items) ---")
        for eid, item in id_to_item.items():
            date  = item.get("date", "")
            stime = item.get("start_time", "")
            etime = item.get("end_time", "")
            text  = (item.get("text") or item.get("caption", "")).replace("\n", " ")
            lines.append(f"  [E{eid}] [{date} {stime}-{etime}] {text}")
        lines.append("\nCall `curate_evidence` with the IDs to keep (≤15).")
        user_text = "\n".join(lines)

        logger.info("[Curation Agent] curating %d evidence items", len(evidence))
        _cfg = MODEL_CONFIG["curation_agent"]
        try:
            response = self.llm.create_with_tools(
                messages=[{"role": "user", "content": user_text}],
                system=self._CURATION_SYSTEM,
                tools=self._CURATE_ONLY_SCHEMA,
                enable_thinking=_cfg["enable_thinking"],
            )
        except Exception as exc:
            logger.warning("[Curation Agent] call failed: %s — truncating to %d",
                           exc, _CURATE_EVIDENCE_THRESHOLD)
            return evidence[:_CURATE_EVIDENCE_THRESHOLD], 0, 0, 0, 0

        keep_ids: Optional[List[int]] = None
        for block in getattr(response, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                txt = (getattr(block, "thinking", "") or "").strip()
                if txt:
                    logger.info("\033[90m[Curation thinking] %s\033[0m",
                                txt.replace("\n", " ")[:400])
            elif btype == "tool_use" and getattr(block, "name", None) == "curate_evidence":
                keep_ids = list((getattr(block, "input", {}) or {}).get("keep") or [])

        usage = getattr(response, "usage", None)
        tok_in  = int(getattr(usage, "input_tokens",                0) or 0) if usage else 0
        tok_out = int(getattr(usage, "output_tokens",               0) or 0) if usage else 0
        tok_cr  = int(getattr(usage, "cache_read_input_tokens",     0) or 0) if usage else 0
        tok_cw  = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0

        if keep_ids is None:
            logger.warning("[Curation Agent] no curate_evidence call — truncating to %d",
                           _CURATE_EVIDENCE_THRESHOLD)
            return evidence[:_CURATE_EVIDENCE_THRESHOLD], tok_in, tok_out, tok_cr, tok_cw

        keep_set = {int(k) for k in keep_ids}
        curated = [item for eid, item in id_to_item.items() if eid in keep_set]
        if len(curated) > _CURATE_EVIDENCE_THRESHOLD:      # hard cap even if model over-keeps
            curated = curated[:_CURATE_EVIDENCE_THRESHOLD]
        logger.info("[Curation Agent] selected %d / %d items (keep=%s)",
                    len(curated), len(evidence), sorted(keep_set))
        return curated, tok_in, tok_out, tok_cr, tok_cw

    def _run_answering_agent(
        self,
        question: str,
        choices: Dict[str, str],
        evidence_items: List[dict],
        before_date: Optional[str],
        before_time: Optional[str],
        target_abs: Optional[int],
    ) -> Tuple[str, str, float, int, int, int, int]:
        """Three-step answering pipeline:
          1. Dedup evidence_items by (date, start_time, end_time, text).
          2. If >15 items: LLM curation call (with thinking) to select ≤15.
          3. Expand to raw 30-sec captions, dedup+sort, final QA LLM call.
        Returns (reasoning_text, letter, latency_sec, in_tokens, out_tokens,
                 cache_read, cache_write) — tokens summed across all LLM calls.
        """
        from agent.search import _abs_ts as _ts
        import time as _time
        t0_total = _time.time()
        tok_in_total = tok_out_total = tok_cr_total = tok_cw_total = 0

        # ---- Step 1: Dedup by (date, start_time, end_time, text) ----
        seen_keys: set = set()
        deduped: List[dict] = []
        for item in evidence_items:
            key = (
                item.get("date", ""), item.get("start_time", ""),
                item.get("end_time", ""), (item.get("text") or "").strip(),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(item)
        logger.info("----- Answering Agent -----")
        logger.info("[Answering Agent] evidence after dedup: %d → %d",
                    len(evidence_items), len(deduped))

        # ---- Step 2: LLM curation (only when pool is large) ----
        curated = deduped
        if len(deduped) > _CURATE_EVIDENCE_THRESHOLD:
            try:
                curated, cur_in, cur_out, cur_cr, cur_cw = self._run_evidence_curation(
                    question=question, choices=choices, evidence=deduped,
                )
                tok_in_total  += cur_in
                tok_out_total += cur_out
                tok_cr_total  += cur_cr
                tok_cw_total  += cur_cw
            except Exception as exc:
                logger.warning("[Curation Agent] failed (%s) — skipping curation, using all %d items",
                               exc, len(deduped))
                curated = deduped

        # ---- Step 3: Expand curated items to raw 30-sec captions ----
        seen_cap_keys: set = set()
        raw_caps: List[dict] = []
        for item in curated:
            expanded = self._search.get_raw_captions(item)
            if expanded:
                for cap in expanded:
                    k = (
                        cap.get("date", ""), cap.get("start_time", ""),
                        cap.get("end_time", ""), (cap.get("text") or "").strip(),
                    )
                    if k not in seen_cap_keys:
                        seen_cap_keys.add(k)
                        raw_caps.append(cap)
            else:
                k = (
                    item.get("date", ""), item.get("start_time", ""),
                    item.get("end_time", ""), (item.get("text") or "").strip(),
                )
                if k not in seen_cap_keys:
                    seen_cap_keys.add(k)
                    raw_caps.append(item)

        raw_caps.sort(
            key=lambda c: _ts(c.get("date", "DAY1"), c.get("start_time", "0"))
        )
        logger.info("[Answering Agent] raw captions for QA: %d", len(raw_caps))

        # ---- Step 4: Final QA LLM call ----
        lines = [
            f"[Memory query point: {(before_date or '')} {(before_time or '')}]",
            "",
            f"--- CURATED EVIDENCE ({len(raw_caps)} raw 30-sec captions, chronological) ---",
        ]
        for c in raw_caps:
            date  = c.get("date", "")
            stime = c.get("start_time", "")
            etime = c.get("end_time", "")
            text  = (c.get("text") or c.get("caption", "")).replace("\n", " ")
            lines.append(f"  [{date} {stime}-{etime}] {text}")
        lines.append("")
        lines.append(f"Question: {question}")
        for ltr in ("A", "B", "C", "D"):
            txt = (choices.get(ltr) or "").strip()
            if txt:
                lines.append(f"{ltr}) {txt}")
        lines.append("")
        lines.append(
            "Reason from the captions above, citing specific timestamps. "
            "End with a single line: `Answer: <letter>`."
        )
        user = "\n".join(lines)

        _cfg = MODEL_CONFIG["answering_agent"]
        t0_qa = _time.time()
        try:
            response = self.llm.create_with_tools(
                messages=[{"role": "user", "content": user}],
                system=self._ANSWERING_SYSTEM,
                tools=None,
                enable_thinking=_cfg["enable_thinking"],
            )
        except AttributeError:
            text = self.llm.generate([
                {"role": "system", "content": self._ANSWERING_SYSTEM},
                {"role": "user",   "content": user},
            ])
            latency = _time.time() - t0_total
            text_for_parse = re.sub(r"<think>.*?</think>", "", text,
                                    flags=re.DOTALL).strip()
            letter = self._extract_answer_letter(text_for_parse)
            logger.info("[Answering Agent] %.2fs  letter=%s  reasoning_chars=%d  "
                        "tokens=(unknown — non-Claude backend)",
                        latency, letter, len(text_for_parse))
            logger.info("[Answering Agent reasoning]\n%s", text_for_parse)
            return (text_for_parse, letter, latency,
                    tok_in_total, tok_out_total, tok_cr_total, tok_cw_total,
                    deduped, curated)
        qa_latency = _time.time() - t0_qa

        thinking_text = ""
        answer_text   = ""
        for block in getattr(response, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                thinking_text += getattr(block, "thinking", "") or ""
            elif btype == "text":
                answer_text += getattr(block, "text", "") or ""

        full = (thinking_text + "\n\n" + answer_text).strip()
        text_for_parse = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL).strip()
        letter = self._extract_answer_letter(text_for_parse)

        # Fallback: if regex couldn't find a letter, ask helper_agent to extract it.
        if letter == "?":
            letter = self._llm_extract_letter(text_for_parse)

        usage = getattr(response, "usage", None)
        qa_in  = int(getattr(usage, "input_tokens",                0) or 0) if usage else 0
        qa_out = int(getattr(usage, "output_tokens",               0) or 0) if usage else 0
        qa_cr  = int(getattr(usage, "cache_read_input_tokens",     0) or 0) if usage else 0
        qa_cw  = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        tok_in_total  += qa_in
        tok_out_total += qa_out
        tok_cr_total  += qa_cr
        tok_cw_total  += qa_cw

        total_latency = _time.time() - t0_total
        logger.info(
            "[Answering Agent] qa_latency=%.2fs  total_latency=%.2fs  "
            "in=%d  out=%d  cache_read=%d  cache_write=%d  letter=%s  reasoning_chars=%d",
            qa_latency, total_latency,
            tok_in_total, tok_out_total, tok_cr_total, tok_cw_total,
            letter, len(text_for_parse),
        )
        logger.info("[Answering Agent reasoning]\n%s", text_for_parse)
        return (text_for_parse, letter, total_latency,
                tok_in_total, tok_out_total, tok_cr_total, tok_cw_total,
                deduped, curated)

    def _llm_extract_letter(self, text: str) -> str:
        """Use self.retriever_llm (helper_agent) to extract A/B/C/D when regex fails."""
        try:
            raw = self.retriever_llm.generate([
                {"role": "system", "content":
                    "You are an answer extractor. Output a single letter A, B, C, or D only. No explanation."},
                {"role": "user", "content":
                    f"Extract the final answer choice (A, B, C, or D) from the following response:\n\n{text}"},
            ])
            letter = self._extract_answer_letter(raw)
            logger.info("[Helper Agent] letter extraction: %r → %s", raw.strip(), letter)
            return letter
        except Exception as exc:
            logger.warning("[Helper Agent] letter extraction failed: %s", exc)
            return "?"

    # Shared letter-extraction helper used by both the answering agent and any
    # downstream fallbacks.
    @staticmethod
    def _extract_answer_letter(text: str) -> str:
        if not text:
            return "?"
        m = re.search(r"Answer\s*[:\-]\s*([A-D])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        for pat in (
            r"(?:answer(?:\s+is)?|option|choice)[:\s]+([A-D])\b",
            r"\b([A-D])\s*[\)\.]",
            r"^\s*([A-D])\s*$",
        ):
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()
        letters = re.findall(r"\b([A-D])\b", text.upper())
        return letters[-1] if letters else "?"

    def _retrieve_once(
        self,
        question: str,
        before_date: Optional[str] = None,
        before_time: Optional[str] = None,
        target_time: Optional[tuple] = None,
        choices: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        One pass of the ReAct loop. Returns a dict compatible with the eval scripts:
          {
            "activities":   [...],
            "raw_captions": [...],
            "trace":        [{round, activity_query, action_query, n_activity_hits}, ...]
          }
        """
        # Branch: when the MAIN model is Claude, switch to native tool-use with
        # interleaved thinking (search_action_speech / search_activity / search_conversation
        # / curate_evidence / answer). All other backends keep the existing text-JSON
        # ReAct loop below.
        try:
            from models.claude_api import ClaudeAPI as _ClaudeAPI
        except Exception:
            _ClaudeAPI = None
        try:
            from models.vllm_client import VLLMClient as _VLLMClient
        except Exception:
            _VLLMClient = None
        _uses_tool_loop = (
            (_ClaudeAPI  is not None and isinstance(self.llm, _ClaudeAPI))
            or (_VLLMClient is not None and isinstance(self.llm, _VLLMClient))
        )

        # GPT is a different system: route it to the native OpenAI Responses
        # harness (function_call / function_call_output over an `input` list).
        # "Speaks the OpenAI Responses API" is exactly ResponsesAPIMixin —
        # openai_gpt.GPTAPI and codex_gpt.CodexGPTAPI.
        try:
            from models._openai_shared import ResponsesAPIMixin as _ResponsesAPI
        except Exception:
            _ResponsesAPI = None
        _uses_gpt_native = (
            _ResponsesAPI is not None and isinstance(self.llm, _ResponsesAPI)
        )
        if _uses_gpt_native:
            return self._retrieve_with_tools_once_gpt(
                question=question,
                before_date=before_date,
                before_time=before_time,
                target_time=target_time,
                choices=choices,
            )

        return self._retrieve_with_tools_once(
            question=question,
            before_date=before_date,
            before_time=before_time,
            target_time=target_time,
            choices=choices,
        )

        from agent.search import _abs_ts as _ts
        target_abs: Optional[int] = None
        if target_time:
            try:
                target_abs = _ts(target_time[0], target_time[1])
            except Exception:
                pass

        system_msg = _SEARCH_SYSTEM
        conversation: List[dict] = [{"role": "system", "content": system_msg}]

        rounds: List[SearchRound] = []
        seen_queries: set = set()
        # Working evidence memory: up to _EVIDENCE_MEMORY_K captions carried
        # across rounds. Each round: rerank → top-K new → curate (memory + new
        # → keep up to _EVIDENCE_MEMORY_K) → use as the caption context for the
        # next decision.
        evidence_memory: List[dict] = []
        err_count = 0

        for round_num in range(1, self.max_rounds + 1):
            user_msg = self._build_round_user_msg(
                question, before_date or "", before_time or "", rounds, round_num,
                evidence_memory=evidence_memory,
                target_abs=target_abs,
            )
            conversation.append({"role": "user", "content": user_msg})

            # Log caption timestamps going into this decision: the evidence
            # memory (chronological), with GT / Near-GT coloring.
            from agent.search import _abs_ts as _ts
            memory_sorted = sorted(
                evidence_memory,
                key=lambda c: _ts(c.get("date", "DAY1"), c.get("start_time", "0"))
                              if c.get("date") else 0,
            )

            _RED   = "\033[91m"
            _BLUE  = "\033[94m"
            _RESET = "\033[0m"
            _NEAR_GT_SEC = 5 * 60  # ±5 minutes

            def _label(cap: dict) -> str:
                base = f"{cap.get('date','')} {cap.get('start_time','')}-{cap.get('end_time','')}"
                if target_abs is None:
                    return base
                try:
                    cs = _ts(cap.get("date", "DAY1"), cap.get("start_time", "0"))
                    ce = _ts(cap.get("date", "DAY1"), cap.get("end_time", "0"))
                except Exception:
                    return base
                if cs <= target_abs <= ce:
                    return f"{_RED}{base} *** GT ***{_RESET}"
                dist = min(abs(target_abs - cs), abs(target_abs - ce))
                if dist <= _NEAR_GT_SEC:
                    return f"{_BLUE}{base} ** Next to GT **{_RESET}"
                return base

            acc_ts = [_label(c) for c in memory_sorted]
            logger.info("[Agent] Round %d memory captions (%d/%d): %s",
                        round_num, len(acc_ts), self._EVIDENCE_MEMORY_K,
                        " | ".join(acc_ts) if acc_ts else "(empty)")

            # Solicit a structurally valid decision, retrying up to _MAX_INROUND_RETRIES
            # if the LLM emits malformed JSON, a bad decision field, an "answer"
            # before any search has been issued, or a "search" missing its queries.
            decision: Optional[dict] = None
            raw: str = ""
            llm_failed = False
            for retry in range(_MAX_INROUND_RETRIES + 1):
                try:
                    raw = self.llm.generate(conversation)
                except Exception as e:
                    logger.warning("LLM search round %d failed: %s", round_num, e)
                    err_count += 1
                    llm_failed = True
                    break

                self._log_thinking(raw)
                # Always parse the raw output via a non-thinking LLM call.
                # The first LLM call may have produced reasoning + JSON in any
                # format (markdown-fenced, embedded in <think>, etc.); the
                # non-thinking parser normalizes it to the strict schema.
                parsed = self._parse_decision_via_llm(raw)
                ok, feedback = self._validate_decision(parsed, has_prior_rounds=bool(rounds))
                if ok:
                    decision = parsed
                    conversation.append({"role": "assistant", "content": raw})
                    break

                logger.info("[Agent] Round %d invalid decision (retry %d/%d): %s",
                            round_num, retry + 1, _MAX_INROUND_RETRIES, feedback.splitlines()[0])
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({"role": "user",      "content": feedback})

            if llm_failed:
                if err_count >= 3:
                    break
                conversation.append({"role": "assistant", "content": '{"decision": "answer"}'})
                continue

            if decision is None:
                # Retries exhausted. Force a search using the raw question as fallback query
                # so we never silently abandon the question with zero evidence.
                logger.warning("[Agent] Round %d retries exhausted — forcing search with the question as query",
                               round_num)
                decision = {
                    "decision": "search",
                    "query":    question[:80],
                }

            # Read the query (new key `query`, with legacy fallbacks for older outputs).
            query_text = (
                decision.get("query")
                or decision.get("action_query")
                or decision.get("activity_query")
                or ""
            ).strip()
            # Optional: a concrete time query extracted from the user question.
            # The agent emits it (a point/interval, ideally dataset format); a
            # NON-THINKING normalizer resolves it to a canonical string, which we
            # parse into an absolute target window. That window drives the
            # similarity time-decay at search time and biases the caption rerank.
            raw_time_query = (
                decision.get("time_query")
                or decision.get("time_qualifier")   # legacy key tolerance
                or ""
            ).strip()
            canonical_tq = self._normalize_time_query(
                question, raw_time_query, before_date, before_time)
            from agent.search import _parse_target_window
            target_window = _parse_target_window(canonical_tq)

            logger.info("[Agent] Round %d | decision=%s query=%s",
                        round_num, decision.get("decision", "?"), query_text or "-")
            logger.info("[Agent] Round %d | time_query: raw=%r canonical=%r window=%s",
                        round_num, raw_time_query, canonical_tq, target_window)

            if decision.get("decision") != "search":
                logger.info("[Agent] Round %d | decision=answer — stopping search", round_num)
                break

            # Avoid repeating the exact query
            if query_text in seen_queries:
                logger.info("[Agent] Repeated query skipped: %r", query_text)
                conversation.append({
                    "role": "user",
                    "content": "You already issued that exact query. Try a different query, or decide to answer.",
                })
                continue
            seen_queries.add(query_text)

            # For downstream code that still names two variables; both point at the same query now.
            activity_query = query_text
            action_query   = query_text

            def _run(level: str, query: str, top_k: int, display_k: int,
                     decay_lambda: Optional[float] = None) -> List[dict]:
                fetch_k = max(top_k, display_k)
                logger.info("[Agent] Searching level=%s query=%r top_k=%d (display_k=%d)",
                            level, query, top_k, display_k)
                hits = self._search.search(
                    query=query, level=level, top_k=fetch_k,
                    before_date=before_date, before_time=before_time,
                    target_window=target_window, decay_lambda=decay_lambda,
                )
                logger.info("[Agent] Showing top %d %s hits (using top %d downstream)",
                            len(hits), level, top_k)
                for rank, h in enumerate(hits, start=1):
                    marker = " <- used" if rank <= top_k else ""
                    final = h.get("score", 0)
                    sim   = h.get("base_score", final)
                    if decay_lambda is None:
                        decay_str = "off"
                    else:
                        decay_str = "%.3f" % ((final / sim) if abs(sim) > 1e-9 else 1.0)
                    line = "%s #%02d [%s %s-%s] sim=%.3f λ^Δt=%s score=%.3f  %s%s" % (
                        level, rank, h["date"], h["start_time"], h["end_time"],
                        sim, decay_str, final, h["text"], marker)
                    logger.info(self._mark_gt(line, h["date"], h["start_time"],
                                              h["end_time"], target_abs))
                return hits[:top_k]

            # Action search — atomic 30-sec hits (time-decay uses lambda_action).
            action_hits = _run("action", action_query, top_k=10, display_k=50,
                               decay_lambda=self.lambda_action)
            # Activity search — SAME query, top-3 (5-min) activities, show top-10
            # (time-decay uses lambda_activity).
            activity_hits = _run("activity", action_query, top_k=3, display_k=10,
                                 decay_lambda=self.lambda_activity)
            # Conversation search — SAME query, top-3 conversation topics, show
            # top-10 (time-decay uses lambda_conversation). Empty if the DAG has
            # no conversation layer (run build_conversation.py → build_dag.py).
            if self._search._nodes.get("conversation"):
                conversation_hits = _run("conversation", action_query,
                                         top_k=3, display_k=10,
                                         decay_lambda=self.lambda_conversation)
            else:
                conversation_hits = []

            # Captions of the top activity + conversation hits → caption pool.
            activity_caps: List[dict] = []
            for act in activity_hits:
                activity_caps.extend(self._search.get_raw_captions(act))
            for conv in conversation_hits:
                activity_caps.extend(self._search.get_raw_captions(conv))

            raw_caps = self._collect_raw_captions(
                action_hits, query=action_query, target_abs=target_abs,
                extra_caps=activity_caps,
                time_qualifier=canonical_tq,        # canonical time signal for rerank
                target_window=target_window,        # decay window for action pool
                decay_lambda=self.lambda_action,
                choices=choices,                    # MCQ options for support/refute ranking
            )
            logger.info("[Agent] Collected %d raw captions (action pool + %d activity captions)",
                        len(raw_caps), len(activity_caps))

            # Curate evidence memory: feed (current memory + new top-K) to the
            # agent and ask it to keep up to _EVIDENCE_MEMORY_K. The returned
            # list replaces the working memory for the next round.
            evidence_memory = self._curate_evidence_memory(
                question        = question,
                current_memory  = evidence_memory,
                new_candidates  = raw_caps,
                max_keep        = self._EVIDENCE_MEMORY_K,
                target_abs      = target_abs,
            )

            rounds.append(SearchRound(
                round_num=round_num,
                activity_query=activity_query, action_query=action_query,
                activity_hits=activity_hits,
                action_hits=action_hits,
                conversation_hits=conversation_hits,
                raw_captions=raw_caps,   # this round's reranked top-K (pre-curation)
            ))

        # ------------------------------------------------------------------
        # Collect evidence by level for the caller
        # ------------------------------------------------------------------
        activities: List[dict] = []
        seen_act_ids: set = set()
        conversations: List[dict] = []
        seen_conv_ids: set = set()

        for r in rounds:
            for h in r.activity_hits:
                if h["id"] not in seen_act_ids:
                    seen_act_ids.add(h["id"])
                    activities.append(h)
            for h in r.conversation_hits:
                if h["id"] not in seen_conv_ids:
                    seen_conv_ids.add(h["id"])
                    conversations.append(h)

        # The curated evidence memory IS the caption set the agent reasoned over.
        # Return it (sorted chronologically) rather than accumulating per-round
        # contributions, since the curation step already dropped what the agent
        # didn't want to keep.
        from agent.search import _abs_ts as _ts
        all_raw_captions: List[dict] = sorted(
            evidence_memory,
            key=lambda c: _ts(c.get("date", "DAY1"), c.get("start_time", "0"))
                          if c.get("date") else 0,
        )

        trace = [
            {
                "round":              r.round_num,
                "activity_query":     r.activity_query,
                "action_query":       r.action_query,
                "n_activity_hits":    len(r.activity_hits),
                "n_conversation_hits": len(r.conversation_hits),
            }
            for r in rounds
        ]

        return {
            "activities":    activities,
            "conversations": conversations,
            "raw_captions":  all_raw_captions,
            "trace":         trace,
        }
