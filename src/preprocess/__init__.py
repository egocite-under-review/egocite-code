"""Preprocessing: the raw EgoLife release -> the caption list the memory build reads.

    preprocess_egolife.py   DenseCaption SRTs -> translated jsonl -> per-clip sync files
    caption_egolife.py      sync files -> output/captions/{source}/{person}/*.json

Prompts live in prompt/caption/*.txt; paths come from config.py.
"""
