"""Caricamento e tokenizzazione dei dati per il classificatore di contenuti sensibili.

Due task sullo stesso encoder:
  - "pii":    token classification BIO sui 20 tipi di dato personale di ai4privacy (open-pii-masking-500k)
  - "safety": multi-label a livello di testo: [unsafe] + 23 categorie Aegis 2.0 (Nemotron-Safety-Guard-Dataset-v3)
Solo italiano e inglese.
"""
import ast
import json
import os
import random

from huggingface_hub import hf_hub_download

PII_LABELS = [
    "GIVENNAME", "SURNAME", "TITLE", "SEX", "GENDER", "AGE", "DATE", "TIME",
    "CITY", "STREET", "BUILDINGNUM", "ZIPCODE", "TELEPHONENUM", "EMAIL",
    "IDCARDNUM", "PASSPORTNUM", "DRIVERLICENSENUM", "TAXNUM", "SOCIALNUM", "CREDITCARDNUMBER",
]
BIO_LABELS = ["O"] + [f"{p}-{l}" for l in PII_LABELS for p in ("B", "I")]

SAFETY_CATEGORIES = [
    "Criminal Planning/Confessions", "Needs Caution", "Hate/Identity Hate", "Violence", "Harassment",
    "Controlled/Regulated Substances", "PII/Privacy", "Profanity", "Sexual", "Immoral/Unethical",
    "Guns and Illegal Weapons", "Illegal Activity", "Suicide and Self Harm", "Unauthorized Advice",
    "Manipulation", "Political/Misinformation/Conspiracy", "Sexual (minor)", "Fraud/Deception", "Threat",
    "Other", "Malware", "Copyright/Trademark/Plagiarism", "High Risk Gov Decision Making",
]
SAFETY_LABELS = ["unsafe"] + SAFETY_CATEGORIES  # colonna 0 = binario, poi le categorie


def _hf(repo, path, revision):
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision)


def load_pii(cfg, split):
    """split: 'train' | 'val' | 'test'. val = campione fisso estratto dal train ufficiale, test = validation ufficiale."""
    path = "data/validation/test.jsonl" if split == "test" else "data/train/train.jsonl"
    f = _hf(cfg["pii_repo"], path, cfg["pii_revision"])
    rows = []
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            if r["language"] not in cfg["languages"]:
                continue
            mask = r["privacy_mask"]  # a volte lista JSON, a volte stringa con repr Python
            mask = ast.literal_eval(mask) if isinstance(mask, str) else mask
            spans = [(m["start"], m["end"], m["label"]) for m in mask]
            rows.append({"text": r["source_text"], "spans": spans, "lang": r["language"], "uid": r["uid"]})
    if split != "test":
        idx = list(range(len(rows)))
        random.Random(cfg.get("data_seed", cfg["seed"])).shuffle(idx)  # split val fisso tra i seed
        val_idx = set(idx[: cfg["pii_val_size"]])
        rows = [r for i, r in enumerate(rows) if (i in val_idx) == (split == "val")]
    return rows


def load_safety(cfg, split):
    """split: 'train' | 'val' | 'test' (split ufficiali train/valid/test)."""
    fname = {"train": "train", "val": "valid", "test": "test"}[split]
    rows, seen, stats = [], set(), {"redacted": 0, "dup": 0}
    for lang in cfg["languages"]:
        f = _hf(cfg["safety_repo"], f"{lang}/{fname}.jsonl", cfg["safety_revision"])
        for line in open(f):
            r = json.loads(line)
            if r["prompt"] == "REDACTED":
                stats["redacted"] += 1
                continue
            resp = r.get("response") or None
            key = (r["prompt"], resp)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            unsafe = "unsafe" in (r["prompt_label"], r.get("response_label"))
            # Le categorie valgono solo se il contenuto e' unsafe (in EN ci sono 'safe' con 'Needs Caution', in IT no).
            cats = [c.strip() for c in r["violated_categories"].split(",") if c.strip()] if unsafe else []
            y = [0.0] * len(SAFETY_LABELS)
            y[0] = float(unsafe)
            for c in cats:
                y[1 + SAFETY_CATEGORIES.index(c)] = 1.0
            rows.append({"prompt": r["prompt"], "response": resp, "y": y, "lang": lang})
    return rows, stats


def tokenize_pii(rows, tok, max_len):
    out = []
    for r in rows:
        enc = tok(r["text"], truncation=True, max_length=max_len, return_offsets_mapping=True)
        labels, last_span = [], None
        for (s, e) in enc["offset_mapping"]:
            if s == e:  # token speciali
                labels.append(-100)
                last_span = None
                continue
            span = next((sp for sp in r["spans"] if s < sp[1] and e > sp[0]), None)
            if span is None:
                labels.append(0)
            else:
                prefix = "I" if span == last_span else "B"
                labels.append(BIO_LABELS.index(f"{prefix}-{span[2]}"))
            last_span = span
        out.append({"input_ids": enc["input_ids"], "labels": labels, "lang": r["lang"]})
    return out


def tokenize_safety(rows, tok, max_len):
    out = []
    for r in rows:
        if r["response"]:
            enc = tok(r["prompt"], r["response"], truncation="longest_first", max_length=max_len)
        else:
            enc = tok(r["prompt"], truncation=True, max_length=max_len)
        out.append({"input_ids": enc["input_ids"], "labels": r["y"], "lang": r["lang"]})
    return out


def make_batches(examples, task, max_tokens, max_bs, rng):
    """Batch per lunghezza simile: mescola, ordina a blocchi, riempie fino a max_tokens (con padding)."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    chunk = max_bs * 50
    batches = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: len(examples[i]["input_ids"]))
        cur, cur_max = [], 0
        for i in part:
            L = len(examples[i]["input_ids"])
            if cur and (max(cur_max, L) * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
                batches.append((task, cur))
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, L)
        if cur:
            batches.append((task, cur))
    return batches


def collate(examples, ids, task, pad_id):
    import torch
    items = [examples[i] for i in ids]
    L = max(len(x["input_ids"]) for x in items)
    input_ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), L), dtype=torch.long)
    for j, x in enumerate(items):
        n = len(x["input_ids"])
        input_ids[j, :n] = torch.tensor(x["input_ids"])
        mask[j, :n] = 1
    if task == "pii":
        labels = torch.full((len(items), L), -100, dtype=torch.long)
        for j, x in enumerate(items):
            labels[j, :len(x["labels"])] = torch.tensor(x["labels"])
    else:
        labels = torch.tensor([x["labels"] for x in items], dtype=torch.float)
    return input_ids, mask, labels
