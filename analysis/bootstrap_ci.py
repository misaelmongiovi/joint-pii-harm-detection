"""Intervalli di confidenza bootstrap (95%, ricampionamento dei documenti del test) per le metriche principali.

    python analysis/bootstrap_ci.py --run_dir /runs/20261016-1910_sensitive-xlmr-large [--n 1000]

Usa il checkpoint best e le soglie tarate sul val. Scrive <run_dir>/analysis/bootstrap_test.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.evaluate import _entities, predict  # noqa: E402
from sensitive.data import (BIO_LABELS, PII_LABELS, SAFETY_LABELS, load_pii, load_safety,  # noqa: E402
                            tokenize_pii, tokenize_safety)
from sensitive.model import SensitiveClassifier  # noqa: E402


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-9), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    ck = os.path.join(a.run_dir, "checkpoints", "best")
    cfg = yaml.safe_load(open(os.path.join(ck, "config.yaml")))
    th = np.array(json.load(open(os.path.join(ck, "thresholds_val.json"))))
    tok = AutoTokenizer.from_pretrained(ck)
    model = SensitiveClassifier(cfg["encoder"], len(BIO_LABELS), len(SAFETY_LABELS), cfg["dropout"], cfg["attn"])
    model.load_state_dict(load_file(os.path.join(ck, "model.safetensors")))
    model.to("cuda").eval()
    rng = np.random.default_rng(0)

    # PII: conteggi tp/fp/fn per documento e per tipo
    pex = tokenize_pii(load_pii(cfg, "test"), tok, cfg["max_len"])
    pp = predict(model, pex, "pii", "cuda", tok.pad_token_id)
    T = len(PII_LABELS)
    C = np.zeros((len(pex), 3, T))
    for i, (e, p) in enumerate(zip(pex, pp)):
        G, P = _entities([l for l in e["labels"] if l != -100]), _entities(p)
        for x in P & G: C[i, 0, PII_LABELS.index(x[0])] += 1
        for x in P - G: C[i, 1, PII_LABELS.index(x[0])] += 1
        for x in G - P: C[i, 2, PII_LABELS.index(x[0])] += 1

    def pii_stats(idx):
        s = C[idx].sum(0)
        return float(f1(s[0].sum(), s[1].sum(), s[2].sum())), float(f1(s[0], s[1], s[2]).mean())

    # Safety
    sex = tokenize_safety(load_safety(cfg, "test")[0], tok, cfg["max_len"])
    prob = np.stack(predict(model, sex, "safety", "cuda", tok.pad_token_id))
    y = np.array([e["labels"] for e in sex]).astype(int)
    pred = (prob >= th).astype(int)

    def saf_stats(idx):
        yy, pr = y[idx], pred[idx]
        tp = (yy & pr).sum(0); fp = ((1 - yy) & pr).sum(0); fn = (yy & (1 - pr)).sum(0)
        f = f1(tp, fp, fn)
        sup = yy[:, 1:].sum(0) > 0
        return float(f[0]), float(f[1:][sup].mean()), float(f1(tp[1:].sum(), fp[1:].sum(), fn[1:].sum()))

    point = {"pii_f1_micro": pii_stats(np.arange(len(C)))[0], "pii_f1_macro": pii_stats(np.arange(len(C)))[1]}
    point.update(dict(zip(["unsafe_f1", "cat_f1_macro", "cat_f1_micro"], saf_stats(np.arange(len(y))))))
    boot = {k: [] for k in point}
    for _ in range(a.n):
        pi = rng.integers(0, len(C), len(C)); si = rng.integers(0, len(y), len(y))
        mi, ma = pii_stats(pi); u, cm, cmi = saf_stats(si)
        for k, v in zip(boot, [mi, ma, u, cm, cmi]):
            boot[k].append(v)
    res = {k: {"point": point[k], "ci95": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]}
           for k in point}
    res["n_boot"] = a.n
    out = os.path.join(a.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "bootstrap_test.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
