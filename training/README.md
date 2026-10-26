# training/ — traccia "modelli piccoli" (fase 1)

Codice usato per: riduzione del vocabolario di GLiClass multilang edge (`skejja_vocab.py`),
modello e formato degli input (`skejja_glic.py`), fine-tuning e distillazione (`train_skejja.py`),
export e quantizzazione ONNX (`skejja_onnx.py`), misure (`run_bench.py`, `skejja_bench_child.py`),
tabelle (`skejja_report.py`).

Nota: questi script sono stati eseguiti dentro Claude Science e possono contenere percorsi
dell'ambiente originale. Renderli eseguibili da riga di comando (argomenti, percorsi relativi a
`bench/data/`) è un compito della fase 2. Metodo completo: `docs/methods_modelli_piccoli.md`.
I checkpoint PyTorch non sono stati salvati; i pesi esatti sono nei bundle ONNX fp32 in `releases/`.

## Fase 2

- `rebuild_checkpoints.py`: ricostruisce i checkpoint PyTorch della fase 1 (zero-shot dal GLiClass originale, fine-tuned dal bundle ONNX fp32) e li verifica.
- `translate_ext.py`: traduzioni OPUS-MT delle frasi esterne (cache in `bench/data/ext/mt_cache.jsonl`).
- `train_skejja2.py`: fine-tuning Choice + Noul multi-sorgente con negativi fratelli e parafrasi; selezione su `bench/data/dev_noul_ext.jsonl`.
- `eval_noul.py`: misure Noul per tipo di negativo, test originale ed esteso, intervalli bootstrap; Choice come controllo.
- `export_bundle.py`: export ONNX fp32 / int8 solo pesi / int8 dinamico, calibrazione e verifica contro torch.
Tutti si lanciano da riga di comando dalla radice del repository; comandi completi in `docs/fase2_noul.md`.

## Addestramento su GPU (workstation RTX 3090)

`train_skejja2.py --device cuda` (oppure `auto`) addestra sulla GPU; il valore predefinito resta `cpu`.
Tempi misurati, dal campo `sec` dei log salvati in `train_meta.json` ([tempi_addestramento.json](../results/fase2/tempi_addestramento.json), 14 ottobre 2026; a parità di script, 320 token e batch 32):

| Giro | Macchina | Passi | Tempo | Secondi per passo |
|---|---|---:|---:|---:|
| run_F | workstation RTX 3090 | 5.086 | 1.202 s (20 min) | 0,236 |
| run_Fh | workstation RTX 3090 | 3.829 | 751 s | 0,196 |
| prova del 14 ottobre | Mac, 10 thread | 120 | 114 s | 0,948 |

**La workstation è circa 4 volte più veloce del Mac, non 20** come stimato a occhio il 13 ottobre: quella stima non era appoggiata a nessun file ed è stata sostituita da questi numeri. Il giro sul Mac usa solo il pool di intenzioni, con testi più corti di quelli di ampiezza, quindi il rapporto a parità di dati è almeno 4. I tempi sono di solo addestramento (preparazione dei dati esclusa) e comprendono le valutazioni periodiche.
Con torch 2.14 su CUDA alcuni kernel (rotary di mmBERT) sono compilati da Triton al primo uso: nell'ambiente serve un compilatore C (`conda install -c conda-forge c-compiler`).
`export_bundle.py` con onnxruntime 1.31 richiede anche `onnx-ir` (quantizzatore MatMulNBits).

## Dati di ampiezza (giro F, 14 ottobre 2026)

`train_skejja2.py` accetta richieste ed etichette di un insegnante (`bench/teacher_label.py`, `bench/teacher_generate.py`):
`--broad_requests` (JSONL di richieste), `--broad_labels` (JSONL di etichette, campo `modello`), `--broad_teacher`
(usa solo quell'insegnante), `--n_broad_choice` / `--n_broad_noul` (-1 = tutti), `--broad_dev_frac` (quota di testi
per lo sviluppo, per `text_id`), `--broad_min_cop` (copertura minima), `--kd_broad`. Le righe `uso = solo_metro`
sono scartate, e un assert verifica che nessun loro testo entri. Lo sviluppo di ampiezza entra nella media di
selezione. Comando del giro F (sulla workstation, file decompressi):

```
python training/train_skejja2.py --init <run_E/best> --teacher <run_C/best> --kd_w 0.5 --kd_agent 0 --kd_broad 0 \
  --lr 2e-5 --head_lr 6e-5 --p_desc_native 0.5 --agent_dir bench/data/agent --n_agent 12000 --max_len 320 \
  --device cuda --threads 8 \
  --broad_requests teacher_trial_in.jsonl requests_synth.jsonl \
  --broad_labels labels_trial_qwen9b.jsonl labels_synth_qwen9b.jsonl --broad_teacher Qwen/Qwen3.5-9B --out <run_F>
```

I file sono in `bench/data/broad/`: `requests_synth.jsonl` e `labels_synth_qwen9b.jsonl` vanno prima decompressi
(`.gz`); `teacher_trial_in.jsonl` non è in git (contiene i testi del metro) e si rigenera con
`python bench/teacher_label.py prepara --requests bench/data/broad/requests_trial.jsonl --out bench/data/broad/teacher_trial_in.jsonl`.

Argomenti completi in [results/fase2/train_meta_run_F.json](../results/fase2/train_meta_run_F.json); risultati in
[docs/fase2_ampiezza.md](../docs/fase2_ampiezza.md), §6.

## Formato S1MB e tetti per fonte (giri G e H, 15–16 ottobre 2026)

Opzioni aggiunte per il giro G: `--max_len 1024`, `--token_budget` (tetto di token per i batch di ampiezza con
molte opzioni), `--n_broad_score` e `--lambda_score` (primitiva Score: entropia incrociata sui livelli più errore
sul valore atteso normalizzato, la metrica di S1MB).

Opzioni aggiunte per il giro H, dopo il verdetto del giro G (`docs/fase2_giro_G.md`, §6–7):

- `--broad_caps 'tipo/origine=n,...'`: tetti per fonte, applicati prima dei tetti per tipo. Le chiavi sono quelle
  di `broad_stats` nel log (`noul/circa`, `choice/casa`, ...; origine assente = `naturale`). Il giro G aveva un
  tetto unico sull'unione dei dati, e le fonti nuove avevano ridotto quelle del giro F a circa un terzo;
- `--bnoul_budget N`: le sì/no di ampiezza vanno in batch propri con un tetto di N token, invece che nei batch sì/no
  da `--bs` con padding sul più lungo;
- il tetto di token conta ora i token dopo il padding, e gli esempi sono ordinati per lunghezza dentro blocchi
  casuali di 100 batch. Nel giro G il tetto sommava le lunghezze vere, quindi un esempio lungo in mezzo a 31 corti
  portava tutto il batch a 1.024 token;
- il teacher Skejja gira solo dove la sua uscita entra nella perdita.

Effetto sul tempo, sulla RTX 3090 con 1.024 token: giro G 8,3 s per passo nei primi 100 passi (4.000 passi in 10,8
ore); giro H 0,42 s per passo (6.404 passi in 2.716 s, valutazioni comprese). Comando del giro H nel §7 del
documento; argomenti completi in [results/fase2/train_meta_run_H.json](../results/fase2/train_meta_run_H.json).
