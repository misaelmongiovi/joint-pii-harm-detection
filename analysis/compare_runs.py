"""Confronto modello congiunto (piu' seed) vs baseline single-task sul test, con bootstrap appaiato.

    python analysis/compare_runs.py --joint RUN42 RUN43 RUN44 --pii_only RUN_PII --safety_only RUN_SAF \
        --out /runs/<...>/comparison_test.json

Per ogni run usa checkpoints/best e (per safety) le soglie tarate sul val. Le predizioni sul test vengono salvate in
<run>/analysis/test_preds.npz e riusate. Il bootstrap ricampiona gli stessi documenti per tutti i modelli
(1000 ricampionamenti): delta = media dei seed congiunti - single-task.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.evaluate import _entities, predict  # noqa: E402
from sensitive.data import (BIO_LABELS, PII_LABELS, SAFETY_LABELS, load_pii, load_safety,  # noqa: E402
                            tokenize_pii, tokenize_safety)
from sensitive.model import SensitiveClassifier  # noqa: E402

_CACHE = {}


def test_data(cfg, tok, task):
    if task not in _CACHE:
        if task == "pii":
            _CACHE[task] = tokenize_pii(load_pii(cfg, "test"), tok, cfg["max_len"])
        else:
            _CACHE[task] = tokenize_safety(load_safety(cfg, "test")[0], tok, cfg["max_len"])
    return _CACHE[task]


def run_preds(rd):
    """Ritorna dict con C (conteggi PII per documento) e/o prob, th (safety)."""
    ck = os.path.join(rd, "checkpoints", "best")
    cache = os.path.join(rd, "analysis", "test_preds.npz")
    cfg = yaml.safe_load(open(os.path.join(ck, "config.yaml")))
    tasks = cfg.get("tasks", ["pii", "safety"])
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}, tasks
    tok = AutoTokenizer.from_pretrained(ck)
    model = SensitiveClassifier(cfg["encoder"], len(BIO_LABELS), len(SAFETY_LABELS), cfg["dropout"], cfg["attn"])
    model.load_state_dict(load_file(os.path.join(ck, "model.safetensors")))
    model.to("cuda").eval()
    out = {}
    if "pii" in tasks:
        ex = test_data(cfg, tok, "pii")
        pp = predict(model, ex, "pii", "cuda", tok.pad_token_id)
        C = np.zeros((len(ex), 3, len(PII_LABELS)), dtype=np.int32)
        for i, (e, p) in enumerate(zip(ex, pp)):
            G, P = _entities([l for l in e["labels"] if l != -100]), _entities(p)
            for x in P & G: C[i, 0, PII_LABELS.index(x[0])] += 1
            for x in P - G: C[i, 1, PII_LABELS.index(x[0])] += 1
            for x in G - P: C[i, 2, PII_LABELS.index(x[0])] += 1
        out["C"] = C
        out["pii_lang"] = np.array([e["lang"] == "it" for e in ex])
    if "safety" in tasks:
        ex = test_data(cfg, tok, "safety")
        out["prob"] = np.stack(predict(model, ex, "safety", "cuda", tok.pad_token_id))
        out["y"] = np.array([e["labels"] for e in ex]).astype(int)
        out["th"] = np.array(json.load(open(os.path.join(ck, "thresholds_val.json"))))
        out["saf_lang"] = np.array([e["lang"] == "it" for e in ex])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    del model
    return out, tasks


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-9), 0.0)


def pii_m(C, idx):
    s = C[idx].sum(0)
    return {"pii_f1_micro": float(f1(s[0].sum(), s[1].sum(), s[2].sum())), "pii_f1_macro": float(f1(s[0], s[1], s[2]).mean())}


def saf_m(y, prob, th, idx, prefix=""):
    yy = y[idx]
    out = {}
    for name, t in [("cal", th), ("05", np.full(y.shape[1], 0.5))]:
        pr = (prob[idx] >= t).astype(int)
        tp = (yy & pr).sum(0); fp = ((1 - yy) & pr).sum(0); fn = (yy & (1 - pr)).sum(0)
        f = f1(tp, fp, fn); sup = yy[:, 1:].sum(0) > 0
        out[f"unsafe_f1_{name}"] = float(f[0])
        out[f"cat_f1_macro_{name}"] = float(f[1:][sup].mean())
        out[f"cat_f1_micro_{name}"] = float(f1(tp[1:].sum(), fp[1:].sum(), fn[1:].sum()))
    return out


def all_metrics(p, tasks, pidx=None, sidx=None):
    m = {}
    if "pii" in tasks:
        m.update(pii_m(p["C"], np.arange(len(p["C"])) if pidx is None else pidx))
    if "safety" in tasks:
        m.update(saf_m(p["y"], p["prob"], p["th"], np.arange(len(p["y"])) if sidx is None else sidx))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", nargs="+", required=True)
    ap.add_argument("--pii_only", required=True)
    ap.add_argument("--safety_only", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()

    runs = {f"joint_{i}": r for i, r in enumerate(a.joint)}
    runs["pii_only"] = a.pii_only
    runs["safety_only"] = a.safety_only
    P, T = {}, {}
    for k, r in runs.items():
        P[k], T[k] = run_preds(r)
        print("predizioni pronte:", k, r, flush=True)

    res = {"runs": runs, "point": {}, "per_language": {}}
    for k in runs:
        res["point"][k] = all_metrics(P[k], T[k])
        pl = {}
        if "pii" in T[k]:
            for lang, flag in [("en", False), ("it", True)]:
                pl[f"pii_f1_micro_{lang}"] = pii_m(P[k]["C"], np.where(P[k]["pii_lang"] == flag)[0])["pii_f1_micro"]
                pl[f"pii_f1_macro_{lang}"] = pii_m(P[k]["C"], np.where(P[k]["pii_lang"] == flag)[0])["pii_f1_macro"]
        if "safety" in T[k]:
            y, pr = P[k]["y"], P[k]["prob"]
            res["point"][k]["unsafe_auroc"] = float(roc_auc_score(y[:, 0], pr[:, 0]))
            for lang, flag in [("en", False), ("it", True)]:
                idx = np.where(P[k]["saf_lang"] == flag)[0]
                mm = saf_m(y, pr, P[k]["th"], idx)
                pl[f"unsafe_f1_cal_{lang}"] = mm["unsafe_f1_cal"]
                pl[f"cat_f1_macro_cal_{lang}"] = mm["cat_f1_macro_cal"]
        res["per_language"][k] = pl

    jk = [k for k in runs if k.startswith("joint_")]
    keys = sorted(set().union(*[res["point"][k] for k in jk]))
    res["joint_mean_std"] = {m: {"mean": float(np.mean([res["point"][k][m] for k in jk])),
                                 "std": float(np.std([res["point"][k][m] for k in jk], ddof=1)) if len(jk) > 1 else 0.0}
                             for m in keys}
    lkeys = sorted(set().union(*[res["per_language"][k] for k in jk]))
    res["joint_mean_std_per_language"] = {m: {"mean": float(np.mean([res["per_language"][k][m] for k in jk])),
                                              "std": float(np.std([res["per_language"][k][m] for k in jk], ddof=1))}
                                          for m in lkeys}

    # bootstrap appaiato: delta = media seed congiunti - single-task (stessi documenti ricampionati)
    rng = np.random.default_rng(0)
    nP, nS = len(P["pii_only"]["C"]), len(P["safety_only"]["y"])
    deltas = {}
    for _ in range(a.n):
        pi = rng.integers(0, nP, nP); si = rng.integers(0, nS, nS)
        single = {**pii_m(P["pii_only"]["C"], pi),
                  **saf_m(P["safety_only"]["y"], P["safety_only"]["prob"], P["safety_only"]["th"], si)}
        joint = [all_metrics(P[k], T[k], pi, si) for k in jk]
        for m in single:
            jm = np.mean([j[m] for j in joint])
            deltas.setdefault(m, []).append(jm - single[m])
            deltas.setdefault(m + "__seed0", []).append(joint[0][m] - single[m])
    res["paired_bootstrap_delta_joint_minus_single"] = {
        m: {"point": None, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "p_joint_better": float(np.mean(np.array(v) > 0))} for m, v in deltas.items()}
    for m, d in res["paired_bootstrap_delta_joint_minus_single"].items():
        base = m.replace("__seed0", "")
        single = res["point"]["pii_only" if base.startswith("pii") else "safety_only"][base]
        joint = res["point"]["joint_0"][base] if m.endswith("__seed0") else res["joint_mean_std"][base]["mean"]
        d["point"] = joint - single
    res["n_boot"] = a.n
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("point", "joint_mean_std")}, indent=1))
    for m, d in res["paired_bootstrap_delta_joint_minus_single"].items():
        print(f"delta {m:28s} {d['point']:+.4f}  CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  P(joint>single)={d['p_joint_better']:.3f}")


if __name__ == "__main__":
    main()
