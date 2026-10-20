"""Analisi aggiuntive del checkpoint best sul test (solo inferenza, non modifica eval/evaluate.py).

    python analysis/analyze_best.py --run_dir /runs/20261016-1910_sensitive-xlmr-large

Scrive <run_dir>/analysis/analysis_test.json con:
  - safety per sottoinsieme: tag del dataset (generic / adapted / jailbreaking) e formato (solo prompt / prompt+risposta)
  - PII: confusione tra tipi per entita' con confini esatti ma tipo sbagliato; F1 per tipo e lingua
  - velocita' di inferenza (testi/s) per task
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.evaluate import _entities, pii_metrics, predict, safety_metrics  # noqa: E402
from sensitive.data import (BIO_LABELS, SAFETY_LABELS, load_pii, load_safety, tokenize_pii,  # noqa: E402
                            tokenize_safety, _hf)
from sensitive.model import SensitiveClassifier  # noqa: E402


def safety_tags(cfg):
    """Ricostruisce tag per riga con la stessa logica di filtro/deduplica di load_safety."""
    tags, seen = [], set()
    for lang in cfg["languages"]:
        for line in open(_hf(cfg["safety_repo"], f"{lang}/test.jsonl", cfg["safety_revision"])):
            r = json.loads(line)
            if r["prompt"] == "REDACTED":
                continue
            key = (r["prompt"], r.get("response") or None)
            if key in seen:
                continue
            seen.add(key)
            tags.append(r["tag"])
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    a = ap.parse_args()
    ck = os.path.join(a.run_dir, "checkpoints", "best")
    cfg = yaml.safe_load(open(os.path.join(ck, "config.yaml")))
    th = json.load(open(os.path.join(ck, "thresholds_val.json")))
    tok = AutoTokenizer.from_pretrained(ck)
    model = SensitiveClassifier(cfg["encoder"], len(BIO_LABELS), len(SAFETY_LABELS), cfg["dropout"], cfg["attn"])
    model.load_state_dict(load_file(os.path.join(ck, "model.safetensors")))
    model.to("cuda").eval()
    res = {}

    # ---- safety per sottoinsieme
    rows, _ = load_safety(cfg, "test")
    tags = safety_tags(cfg)
    assert len(tags) == len(rows)
    ex = tokenize_safety(rows, tok, cfg["max_len"])
    torch.cuda.synchronize(); t = time.time()
    prob = np.stack(predict(model, ex, "safety", "cuda", tok.pad_token_id))
    torch.cuda.synchronize(); res["speed_safety_texts_per_s"] = len(ex) / (time.time() - t)
    y = np.array([e["labels"] for e in ex])
    fmt = np.array(["prompt+response" if r["response"] else "prompt" for r in rows])
    tags = np.array(tags); langs = np.array([r["lang"] for r in rows])
    sub = {}
    for name, mask in [*[(f"tag={v}", tags == v) for v in sorted(set(tags))],
                       *[(f"format={v}", fmt == v) for v in sorted(set(fmt))]]:
        m = safety_metrics(y[mask], prob[mask], langs[mask], th)
        sub[name] = {"n": int(mask.sum()), "unsafe_frac": float(y[mask, 0].mean()),
                     **{k: v for k, v in m.items() if isinstance(v, float)}}
    res["safety_subsets"] = sub
    # coerenza: categorie predette quando unsafe non e' predetto e viceversa
    pred = prob >= np.array(th)
    res["safety_consistency"] = {
        "pred_unsafe_without_category": int((pred[:, 0] & ~pred[:, 1:].any(1)).sum()),
        "pred_category_without_unsafe": int((~pred[:, 0] & pred[:, 1:].any(1)).sum()),
        "n": int(len(pred)),
    }

    # ---- PII
    prow = load_pii(cfg, "test")
    pex = tokenize_pii(prow, tok, cfg["max_len"])
    torch.cuda.synchronize(); t = time.time()
    pp = predict(model, pex, "pii", "cuda", tok.pad_token_id)
    torch.cuda.synchronize(); res["speed_pii_texts_per_s"] = len(pex) / (time.time() - t)
    gold = [[l for l in e["labels"] if l != -100] for e in pex]
    conf = collections.Counter()
    boundary_err = 0
    for g, p in zip(gold, pp):
        G, P = _entities(g), _entities(p)
        gspan = {(s, e): t for t, s, e in G}
        for t_, s, e in P - G:
            if (s, e) in gspan:
                conf[(gspan[(s, e)], t_)] += 1  # confini giusti, tipo sbagliato
            elif any(t2 == t_ and not (e < s2 or s > e2) for t2, s2, e2 in G):
                boundary_err += 1  # stesso tipo, confini diversi
    res["pii_type_confusions_top"] = [{"gold": g, "pred": p, "n": n} for (g, p), n in conf.most_common(15)]
    res["pii_type_confusions_total"] = int(sum(conf.values()))
    res["pii_boundary_errors_same_type"] = boundary_err
    for lang in cfg["languages"]:
        idx = [i for i, e in enumerate(pex) if e["lang"] == lang]
        m = pii_metrics([gold[i] for i in idx], [pp[i] for i in idx])
        res[f"pii_{lang}"] = {"f1_micro": m["pii/entity_f1_micro"], "f1_macro": m["pii/entity_f1_macro"],
                              "per_type_f1": {k: v["f1"] for k, v in m["pii/per_type"].items()}}
    res["gpu"] = torch.cuda.get_device_name(0)

    out = os.path.join(a.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "analysis_test.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("pii_e") and not k.startswith("pii_i")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
