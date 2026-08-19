"""The retrieval agent: question -> evidence -> answer.

    EpisodicSearch  FAISS search over a built dag.json, with time decay and
                    graph traversal down to the raw 30-second captions.
    EpisodicAgent   a tool-use loop on top of it. It calls search_action_speech /
                    search_activity / search_conversation to gather evidence,
                    curate_evidence to prune it, and answer() to hand the curated
                    captions to a separate answering agent.

Three LLM roles, configured per backend in agent.py's MODEL_CONFIG block:
retrieval agent, curation agent, answering agent.

`memory/` builds the DAG this reads; every prompt sent from here lives in
prompt/agent/*.txt.
"""

from agent.agent import EpisodicAgent, AgentResult, SearchRound
from agent.search import EpisodicSearch

__all__ = ["EpisodicAgent", "AgentResult", "SearchRound", "EpisodicSearch"]
