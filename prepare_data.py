"""Rebuild train-2.jsonl (the fine-tuning file read by FINAL.ipynb).

The notebook fine-tunes on 1,524 rows: prefinetuningdata-2.xlsx (1,793 rows)
minus the 269 rows held out in test.xlsx. This script reconstructs that file
from the two spreadsheets, using the same system prompt and user-prompt
format as the evaluation code in the notebook.

NOTE: this is a reconstruction; the original train-2.jsonl was not saved.

Usage:  python prepare_data.py
"""
import json
import pandas as pd

TRAIN_XLSX = "prefinetuningdata-2.xlsx"
HELDOUT_XLSX = "test.xlsx"
OUT = "train-2.jsonl"

SYSTEM_PROMPT = (
    "You are a medical imaging assistant. You will be given a question about "
    "a radiology image and a list of clinically relevant entities that were "
    "automatically extracted from the image's findings using an NER model. "
    "The entity list may occasionally be incomplete or contain minor errors, "
    "so use it as supporting context rather than as a fully certain ground truth. "
    "If no entities are available, answer using your own medical knowledge and "
    "the question alone.\n\n"
    "Answer with a single word or short phrase only, matching the concise style "
    "shown in training (e.g. 'yes', 'no', 'cardiovascular'). Do not explain your "
    "reasoning or add extra commentary."
)

def build_user_prompt(question, entities):
    entities = entities if isinstance(entities, str) and entities.strip() else "None available"
    return f"Question: {question}\nExtracted entities: {entities}"

train = pd.read_excel(TRAIN_XLSX)
held = pd.read_excel(HELDOUT_XLSX)

key = lambda d: list(zip(d["question"], d["answer"], d["Classified_Keywords"].astype(str)))
held_keys = set(key(held))
keep = [k not in held_keys for k in key(train)]
fit = train[keep].reset_index(drop=True)
print(f"{len(train)} rows - {len(held)} held out = {len(fit)} fine-tuning rows")

with open(OUT, "w", encoding="utf-8") as f:
    for _, r in fit.iterrows():
        rec = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(r["question"], r["Classified_Keywords"])},
            {"role": "assistant", "content": str(r["answer"])},
        ]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"wrote {OUT}")
