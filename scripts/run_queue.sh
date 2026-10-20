#!/bin/bash
# Lancia i training in sequenza, ciascuno nella propria sessione tmux (un solo training pesante alla volta).
# Uso: scripts/run_queue.sh "<nome>|<config>|<obiettivo>" ...
# Lo stato della coda e' in /runs/queue.log
set -u
for spec in "$@"; do
  IFS='|' read -r name cfg goal <<< "$spec"
  rd=/runs/$(date +%Y%m%d-%H%M)_$name
  mkdir -p "$rd"
  cmd="cd /workspace && python train.py --config $cfg --run_dir $rd 2>&1 | tee -a $rd/train.log"
  cat > "$rd/README.md" <<EOR
# $name

**Stato:** in corso (avviato $(date '+%Y-%m-%d %H:%M'))

## Obiettivo
$goal

## Comando
\`\`\`bash
tmux new -d -s $name "$cmd"
\`\`\`
Config: \`$cfg\` (copiata in \`config.yaml\`); codice in \`code/\`. Run di riferimento: \`/runs/20261016-1910_sensitive-xlmr-large\`.

## Risultato
_(da completare a fine run)_
EOR
  echo "$(date '+%F %T') START $name -> $rd" >> /runs/queue.log
  tmux new -d -s "$name" "$cmd"
  while tmux has-session -t "$name" 2>/dev/null; do sleep 30; done
  if grep -q '^fine' "$rd/train.log"; then st=OK; else st=FAILED; fi
  echo "$(date '+%F %T') END $name $st" >> /runs/queue.log
done
echo "$(date '+%F %T') QUEUE DONE" >> /runs/queue.log
