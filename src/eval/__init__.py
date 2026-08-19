"""Benchmark harnesses — run the agent over a question set and score it.

    eval_egolifeqa.py   EgoLifeQA  (Ego-QA-4.4K / manual-2.9K), A/B/C/D
    eval_egomem.py      EgoMem     (LifeDialBench), single/multi-event, detail, time
    eval_egor1.py       Ego-R1-Bench, --benchmark manual|gemini

All three take --person and --source {densecaption,vlm}, build an EpisodicAgent
over output/memory/{source}/{person}/dag.json, and write

    output/logs/{bench}/{source}/{name}.log
    output/results/{bench}/{source}/{name}.json

Question sets are read from config.DATASET_DIR; the QA system prompt is
prompt/eval/qa_system.txt.
"""
