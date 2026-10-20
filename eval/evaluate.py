"""Metriche ufficiali del classificatore di contenuti sensibili (non modificare senza chiedere:
la metrica deve restare confrontabile tra i run).

- PII:    F1 a livello di entita' (match esatto di tipo + primo/ultimo token), micro e macro sui 20 tipi.
- Safety: F1 binario 'unsafe' + F1 per categoria (soglia 0.5 oppure soglie tarate sul val), micro e macro.
- Score principale: media di macro-F1 entita' PII e macro-F1 categorie safety (entrambe a grana fine).

Uso da riga di comando (valuta un checkpoint sul test set):
    python eval/evaluate.py --ckpt /runs/<run>/checkpoints/best --split test
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sensitive.data import BIO_LABELS, PII_LABELS, SAFETY_LABELS  # noqa: E402


def _entities(seq):
    """seq: lista di id BIO (gia' senza -100). Ritorna set di (tipo, inizio, fine)."""
    ents, cur = set(), None
    for i, l in enumerate(seq + [0]):
        name = BIO_LABELS[l]
        if name == "O" or name.startswith("B-") or (cur and name[2:] != cur[0]):
            if cur:
                ents.add((cur[0], cur[1], i - 1))
            cur = (name[2:], i) if name != "O" else None
        # I-X che continua l'entita' corrente: niente da fare
    return ents


def pii_metrics(gold_seqs, pred_seqs, langs=None):
    """gold_seqs/pred_seqs: liste di liste di id BIO allineate (token speciali gia' rimossi)."""
    tp = {t: 0 for t in PII_LABELS}; fp = dict(tp); fn = dict(tp)
    by_lang = {}
    for k, (g, p) in enumerate(zip(gold_seqs, pred_seqs)):
        G, P = _entities(list(g)), _entities(list(p))
        lt = by_lang.setdefault(langs[k] if langs else "all", [0, 0, 0])
        for e in P & G: tp[e[0]] += 1; lt[0] += 1
        for e in P - G: fp[e[0]] += 1; lt[1] += 1
        for e in G - P: fn[e[0]] += 1; lt[2] += 1
    f1 = lambda a, b, c: (2 * a / (2 * a + b + c)) if a + b + c else 0.0
    per = {t: {"p": tp[t] / max(tp[t] + fp[t], 1), "r": tp[t] / max(tp[t] + fn[t], 1),
               "f1": f1(tp[t], fp[t], fn[t]), "support": tp[t] + fn[t]} for t in PII_LABELS}
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    return {
        "pii/entity_f1_micro": f1(T, FP, FN),
        "pii/entity_f1_macro": float(np.mean([v["f1"] for v in per.values() if v["support"] > 0])),
        "pii/precision_micro": T / max(T + FP, 1),
        "pii/recall_micro": T / max(T + FN, 1),
        **{f"pii/f1_micro_{l}": f1(*v) for l, v in by_lang.items()},
        "pii/per_type": per,
    }


def _f1_bin(y, p):
    tp = float(((y == 1) & (p == 1)).sum()); fp = float(((y == 0) & (p == 1)).sum()); fn = float(((y == 1) & (p == 0)).sum())
    return (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0, tp, fp, fn


def tune_thresholds(y, prob, grid=np.arange(0.05, 0.96, 0.05)):
    """Soglia per colonna che massimizza F1 (da calcolare sul val, applicare al test)."""
    return [float(max(grid, key=lambda t: _f1_bin(y[:, j], (prob[:, j] >= t).astype(int))[0])) for j in range(y.shape[1])]


def safety_metrics(y, prob, langs=None, thresholds=None):
    """y, prob: array [N, 1+C]; colonna 0 = unsafe, poi categorie."""
    y = np.asarray(y); prob = np.asarray(prob)
    th = np.asarray(thresholds if thresholds is not None else [0.5] * y.shape[1])
    pred = (prob >= th).astype(int)
    from sklearn.metrics import roc_auc_score
    f1u, tpu, fpu, fnu = _f1_bin(y[:, 0], pred[:, 0])
    per, TP, FP, FN = {}, 0, 0, 0
    for j, name in enumerate(SAFETY_LABELS[1:], start=1):
        f, tp, fp, fn = _f1_bin(y[:, j], pred[:, j])
        per[name] = {"p": tp / max(tp + fp, 1), "r": tp / max(tp + fn, 1), "f1": f, "support": int(y[:, j].sum())}
        TP += tp; FP += fp; FN += fn
    out = {
        "safety/unsafe_f1": f1u,
        "safety/unsafe_precision": tpu / max(tpu + fpu, 1),
        "safety/unsafe_recall": tpu / max(tpu + fnu, 1),
        "safety/unsafe_auroc": float(roc_auc_score(y[:, 0], prob[:, 0])) if 0 < y[:, 0].sum() < len(y) else float("nan"),
        "safety/cat_f1_micro": (2 * TP / (2 * TP + FP + FN)) if TP + FP + FN else 0.0,
        "safety/cat_f1_macro": float(np.mean([v["f1"] for v in per.values() if v["support"] > 0])),
        "safety/per_category": per,
    }
    if langs is not None:
        langs = np.asarray(langs)
        for l in sorted(set(langs)):
            m = langs == l
            out[f"safety/unsafe_f1_{l}"] = _f1_bin(y[m, 0], pred[m, 0])[0]
            out[f"safety/cat_f1_macro_{l}"] = float(np.mean([_f1_bin(y[m, j], pred[m, j])[0]
                                                             for j in range(1, y.shape[1]) if y[m, j].sum() > 0]))
    return out


def main_score(m):
    return 0.5 * (m["pii/entity_f1_macro"] + m["safety/cat_f1_macro"])


# ---------------------------------------------------------------- inferenza + CLI

def predict(model, examples, task, device, pad_id, max_tokens=32768, max_bs=256):
    import random
    import torch
    from sensitive.data import collate, make_batches
    model.eval()
    batches = make_batches(examples, task, max_tokens, max_bs, random.Random(0))
    preds = [None] * len(examples)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _, ids in batches:
            x, m, _ = collate(examples, ids, task, pad_id)
            out = model(x.to(device), m.to(device), task).float()
            if task == "pii":
                am = out.argmax(-1).cpu().numpy()
                for j, i in enumerate(ids):
                    lab = examples[i]["labels"]
                    preds[i] = [int(am[j, t]) for t in range(len(lab)) if lab[t] != -100]
            else:
                pr = torch.sigmoid(out).cpu().numpy()
                for j, i in enumerate(ids):
                    preds[i] = pr[j]
    model.train()
    return preds


def evaluate(model, pii_ex, safety_ex, device, pad_id, thresholds=None):
    pp = predict(model, pii_ex, "pii", device, pad_id)
    gold = [[l for l in e["labels"] if l != -100] for e in pii_ex]
    m = pii_metrics(gold, pp, [e["lang"] for e in pii_ex])
    sp = np.stack(predict(model, safety_ex, "safety", device, pad_id))
    y = np.array([e["labels"] for e in safety_ex])
    m.update(safety_metrics(y, sp, [e["lang"] for e in safety_ex], thresholds))
    m["score"] = main_score(m)
    return m, y, sp


def _cli():
    import torch
    import yaml
    from transformers import AutoTokenizer
    from sensitive.data import load_pii, load_safety, tokenize_pii, tokenize_safety
    from sensitive.model import SensitiveClassifier
    from safetensors.torch import load_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="cartella con model.safetensors, config.yaml, tokenizer")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default=None, help="file json dei risultati (default: <ckpt>/eval_<split>.json)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(a.ckpt, "config.yaml")))
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = SensitiveClassifier(cfg["encoder"], len(BIO_LABELS), len(SAFETY_LABELS), cfg["dropout"], cfg["attn"])
    model.load_state_dict(load_file(os.path.join(a.ckpt, "model.safetensors")))
    model.to("cuda")
    pii = tokenize_pii(load_pii(cfg, a.split), tok, cfg["max_len"])
    saf = tokenize_safety(load_safety(cfg, a.split)[0], tok, cfg["max_len"])
    res = {}
    m, _, _ = evaluate(model, pii, saf, "cuda", tok.pad_token_id)
    res["threshold_0.5"] = m
    th_path = os.path.join(a.ckpt, "thresholds_val.json")
    if os.path.exists(th_path):
        m2, _, _ = evaluate(model, pii, saf, "cuda", tok.pad_token_id, json.load(open(th_path)))
        res["thresholds_val"] = m2
    out = a.out or os.path.join(a.ckpt, f"eval_{a.split}.json")
    json.dump(res, open(out, "w"), indent=1)
    for k, mm in res.items():
        print(k, {kk: round(v, 4) for kk, v in mm.items() if isinstance(v, float)})
    print("salvato in", out)


if __name__ == "__main__":
    _cli()
