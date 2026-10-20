#!/bin/bash
# Coda dei baseline per il paper (avviata il 2026-10-17)
exec /workspace/scripts/run_queue.sh \
  "sensitive-pii-only|configs/sensitive_xlmr_large_pii_only.yaml|Baseline single-task: solo la testa PII, stessa ricetta (3 epoche, stessi iperparametri, seed 42). Serve a capire se l'addestramento congiunto peggiora o migliora il task PII." \
  "sensitive-safety-only|configs/sensitive_xlmr_large_safety_only.yaml|Baseline single-task: solo la testa safety (unsafe + 23 categorie), stessa ricetta e seed 42. Serve a capire se l'addestramento congiunto peggiora o migliora il task safety." \
  "sensitive-joint-seed43|configs/sensitive_xlmr_large_seed43.yaml|Modello congiunto con seed 43 (split di validazione PII fisso, data_seed 42): stima della variabilita' tra seed del run 20261016-1910." \
  "sensitive-joint-seed44|configs/sensitive_xlmr_large_seed44.yaml|Modello congiunto con seed 44 (split di validazione PII fisso, data_seed 42): stima della variabilita' tra seed del run 20261016-1910."
