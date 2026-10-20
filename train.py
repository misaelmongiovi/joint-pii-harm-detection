"""Training multi-task del classificatore di contenuti sensibili (PII a grana fine + categorie di danno).

    python train.py --config configs/sensitive_xlmr_large.yaml --run_dir /runs/<AAAAMMGG-HHMM_nome>
    python train.py ... --max_steps 50 --eval_every 25     # prova breve

Riprende automaticamente da <run_dir>/checkpoints/last/state.pt se esiste.
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from eval.evaluate import evaluate, pii_metrics, predict, safety_metrics, tune_thresholds
from sensitive.data import (BIO_LABELS, SAFETY_LABELS, collate, load_pii, load_safety, make_batches,
                            tokenize_pii, tokenize_safety)
from sensitive.model import SensitiveClassifier

HERE = os.path.dirname(os.path.abspath(__file__))


ALL_TASKS = ["pii", "safety"]
MACRO_KEY = {"pii": "pii/entity_f1_macro", "safety": "safety/cat_f1_macro"}


def epoch_batches(cfg, pii_tr, saf_tr, epoch):
    rng = random.Random(cfg["seed"] * 1000 + epoch)
    tasks = cfg.get("tasks", ALL_TASKS)
    b = make_batches(pii_tr, "pii", cfg["max_tokens"], cfg["max_bs"], rng) if "pii" in tasks else []
    b += make_batches(saf_tr, "safety", cfg["max_tokens"], cfg["max_bs"], rng) if "safety" in tasks else []
    rng.shuffle(b)
    return b


def eval_tasks(model, tasks, pii_ex, saf_ex, dev, pad_id, thresholds=None):
    """Metriche ufficiali (eval/evaluate.py) sui soli task addestrati; score = media delle macro-F1 dei task.
    Con entrambi i task coincide esattamente con evaluate()/main_score."""
    if sorted(tasks) == ALL_TASKS:
        return evaluate(model, pii_ex, saf_ex, dev, pad_id, thresholds)
    m, y, sp = {}, None, None
    if "pii" in tasks:
        pp = predict(model, pii_ex, "pii", dev, pad_id)
        m.update(pii_metrics([[l for l in e["labels"] if l != -100] for e in pii_ex], pp, [e["lang"] for e in pii_ex]))
    if "safety" in tasks:
        sp = np.stack(predict(model, saf_ex, "safety", dev, pad_id))
        y = np.array([e["labels"] for e in saf_ex])
        m.update(safety_metrics(y, sp, [e["lang"] for e in saf_ex], thresholds))
    m["score"] = float(np.mean([m[MACRO_KEY[t]] for t in tasks]))
    return m, y, sp


def save_best(model, tok, cfg, metrics, path):
    tmp = path + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(tmp, "model.safetensors"))
    tok.save_pretrained(tmp)
    yaml.safe_dump(cfg, open(os.path.join(tmp, "config.yaml"), "w"), sort_keys=False)
    json.dump({"bio_labels": BIO_LABELS, "safety_labels": SAFETY_LABELS}, open(os.path.join(tmp, "labels.json"), "w"), indent=1)
    json.dump(metrics, open(os.path.join(tmp, "val_metrics.json"), "w"), indent=1)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def scalars(m):
    return {k: v for k, v in m.items() if isinstance(v, (int, float))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="limite di step (prova breve)")
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--val_limit", type=int, default=None, help="usa solo N esempi di val per task (prova breve)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.eval_every:
        cfg["eval_every"] = a.eval_every
    rd = a.run_dir
    ck = os.path.join(rd, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(rd, "config.yaml"), "w"), sort_keys=False)
    code_dir = os.path.join(rd, "code")
    if not os.path.exists(code_dir):  # copia del codice usato (al primo avvio, non sui resume)
        for sub in ["train.py", "sensitive", "eval", "configs", "requirements.txt"]:
            src = os.path.join(HERE, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(code_dir, sub), ignore=shutil.ignore_patterns("__pycache__"))
            elif os.path.exists(src):
                os.makedirs(code_dir, exist_ok=True)
                shutil.copy(src, code_dir)

    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"

    t0 = time.time()
    tasks = cfg.get("tasks", ALL_TASKS)
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    pii_tr, pii_va, saf_tr, saf_va, saf_stats = [], [], [], [], {}
    if "pii" in tasks:
        pii_tr = tokenize_pii(load_pii(cfg, "train"), tok, cfg["max_len"])
        pii_va = tokenize_pii(load_pii(cfg, "val"), tok, cfg["max_len"])
    if "safety" in tasks:
        saf_rows, saf_stats = load_safety(cfg, "train")
        saf_tr = tokenize_safety(saf_rows, tok, cfg["max_len"])
        saf_va = tokenize_safety(load_safety(cfg, "val")[0], tok, cfg["max_len"])
    if a.val_limit:
        pii_va, saf_va = pii_va[:a.val_limit], saf_va[:a.val_limit]
    print(f"dati: pii train {len(pii_tr)} val {len(pii_va)} | safety train {len(saf_tr)} val {len(saf_va)} "
          f"(scartati {saf_stats}) | {time.time() - t0:.0f}s", flush=True)

    model = SensitiveClassifier(cfg["encoder"], len(BIO_LABELS), len(SAFETY_LABELS), cfg["dropout"], cfg["attn"]).to(dev)
    if cfg.get("grad_checkpointing"):
        model.encoder.gradient_checkpointing_enable()
    enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": cfg["lr"]}, {"params": head_p, "lr": cfg["head_lr"]}],
                            weight_decay=cfg["weight_decay"], betas=(0.9, 0.98), eps=1e-6, fused=True)
    steps_per_epoch = len(epoch_batches(cfg, pii_tr, saf_tr, 0))
    total = steps_per_epoch * cfg["epochs"]
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    print(f"step per epoca {steps_per_epoch}, step totali {total}", flush=True)

    step, epoch, bi, best = 0, 0, 0, -1.0
    last = os.path.join(ck, "last", "state.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, epoch, bi, best = st["step"], st["epoch"], st["batch_idx"], st["best"]
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state(st["rng_cuda"])
        print(f"ripreso da step {step} (epoca {epoch}, batch {bi}, best {best:.4f})", flush=True)

    tb = SummaryWriter(os.path.join(rd, "tb"), purge_step=step if step else None)
    mlog = open(os.path.join(rd, "metrics.jsonl"), "a")

    def log(d):
        d = {"step": step, "time": time.time(), **d}
        mlog.write(json.dumps(d) + "\n"); mlog.flush()
        for k, v in d.items():
            if k not in ("step", "time") and isinstance(v, (int, float)) and math.isfinite(v):
                tb.add_scalar(k, v, step)

    last_eval = [-1]

    def do_eval():
        nonlocal best
        if last_eval[0] == step:  # gia' valutato a questo step (es. fine run = fine epoca)
            return
        last_eval[0] = step
        m, _, _ = eval_tasks(model, tasks, pii_va, saf_va, dev, tok.pad_token_id)
        log({f"val/{k}": v for k, v in scalars(m).items()})
        msg = f"[eval step {step}] score {m['score']:.4f}"
        if "pii" in tasks:
            msg += f" | pii F1 micro {m['pii/entity_f1_micro']:.4f} macro {m['pii/entity_f1_macro']:.4f}"
        if "safety" in tasks:
            msg += (f" | unsafe F1 {m['safety/unsafe_f1']:.4f} cat F1 micro {m['safety/cat_f1_micro']:.4f}"
                    f" macro {m['safety/cat_f1_macro']:.4f}")
        print(msg, flush=True)
        if m["score"] > best:
            best = m["score"]
            save_best(model, tok, cfg, {"step": step, **m}, os.path.join(ck, "best"))
            print(f"  nuovo best ({best:.4f}) salvato", flush=True)

    def save_last():
        os.makedirs(os.path.dirname(last), exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "epoch": epoch, "batch_idx": bi, "best": best,
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)

    model.train()
    acc = {"pii": [], "safety": []}
    ntok, tlast = 0, time.time()
    done = step >= total
    while not done and epoch < cfg["epochs"]:
        batches = epoch_batches(cfg, pii_tr, saf_tr, epoch)
        while bi < len(batches):
            task, ids = batches[bi]
            data = pii_tr if task == "pii" else saf_tr
            x, m, y = collate(data, ids, task, tok.pad_token_id)
            x, m, y = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, m, task)
            loss = SensitiveClassifier.loss(out, y, task) * cfg["loss_weight"][task]
            if not torch.isfinite(loss):
                save_last()
                raise RuntimeError(f"loss non finita allo step {step} (task {task}): run fermato")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; bi += 1
            acc[task].append(loss.item()); ntok += int(m.sum())

            if step % cfg["log_every"] == 0:
                dt = time.time() - tlast
                log({"train/loss_pii": float(np.mean(acc["pii"])) if acc["pii"] else float("nan"),
                     "train/loss_safety": float(np.mean(acc["safety"])) if acc["safety"] else float("nan"),
                     "train/grad_norm": float(gn), "train/lr": sched.get_last_lr()[0], "train/epoch": epoch + bi / len(batches),
                     "perf/tokens_per_s": ntok / dt, "perf/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                print(f"step {step}/{total} ep {epoch} loss pii {np.mean(acc['pii']) if acc['pii'] else float('nan'):.4f} "
                      f"safety {np.mean(acc['safety']) if acc['safety'] else float('nan'):.4f} gn {float(gn):.2f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {ntok / dt:.0f} tok/s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                      flush=True)
                acc = {"pii": [], "safety": []}; ntok, tlast = 0, time.time()
            if step % cfg["eval_every"] == 0:
                do_eval()
            if step % cfg["save_every"] == 0:
                save_last()
            if step >= total:
                done = True
                break
        if not done:
            epoch += 1; bi = 0
            do_eval(); save_last()

    do_eval()
    save_last()

    # Soglie per colonna tarate sul val con il modello migliore, poi valutazione finale sul test.
    model.load_state_dict(load_file(os.path.join(ck, "best", "model.safetensors")))
    th = None
    if "safety" in tasks:
        _, yv, pv = eval_tasks(model, tasks, pii_va, saf_va, dev, tok.pad_token_id)
        th = tune_thresholds(yv, pv)
        json.dump(th, open(os.path.join(ck, "best", "thresholds_val.json"), "w"))
    pii_te = tokenize_pii(load_pii(cfg, "test"), tok, cfg["max_len"]) if "pii" in tasks else []
    saf_te = tokenize_safety(load_safety(cfg, "test")[0], tok, cfg["max_len"]) if "safety" in tasks else []
    res = {"threshold_0.5": eval_tasks(model, tasks, pii_te, saf_te, dev, tok.pad_token_id)[0],
           "thresholds_val": eval_tasks(model, tasks, pii_te, saf_te, dev, tok.pad_token_id, th)[0],
           "best_step": json.load(open(os.path.join(ck, "best", "val_metrics.json")))["step"]}
    json.dump(res, open(os.path.join(rd, "test_results.json"), "w"), indent=1)
    for k in ("threshold_0.5", "thresholds_val"):
        log({f"test_{k}/{kk}": v for kk, v in scalars(res[k]).items()})
        print(f"TEST ({k}): " + " ".join(f"{kk}={v:.4f}" for kk, v in scalars(res[k]).items()), flush=True)
    tb.close()
    print("fine", flush=True)


if __name__ == "__main__":
    main()
