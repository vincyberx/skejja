# Skejja — Fase 2, punti 2 e 3: test fuori da MASSIVE con opzioni in italiano (v0)

_6 ottobre 2026. Numeri da [metrics_ood_v0.csv](../results/fase2/metrics_ood_v0.csv), misurati su Mac, PyTorch fp32, calibrazione del protocollo (nessun esempio per domanda). Test costruito da `bench/build_ood_v0.py` → `bench/data/ood/ood_v0.jsonl`._

## Il test

Testi in italiano; ogni opzione e domanda è scritta sia in italiano sia in inglese. **Il test l'ha scritto l'agente (Claude): è una prima versione da rivedere e da allargare con esempi reali.** Si usa solo per valutare, mai per addestrare.

| Scenario | Contenuto | Item |
|---|---|---|
| Domotica | 80 comandi per una casa con 20 dispositivi in 7 famiglie (luci, tapparelle, clima, elettrodomestici, sicurezza, media, energia). Choice a 8 opzioni; sì/no con positivo, comando fratello e comando di un'altra famiglia | 80 Choice, 236 sì/no |
| Agente | 36 azioni (comandi shell e richieste in italiano) × 6 proprietà sì/no: cancella o sovrascrive file, si connette a internet, usa privilegi di amministratore, installa software, solo lettura, pubblica codice o pacchetti | 216 sì/no |
| Email | 40 email brevi in 8 categorie; Choice sulla categoria; sì/no su 3 proprietà (richiede risposta, urgente, contiene una scadenza) più una categoria vera e una falsa | 40 Choice, 200 sì/no |

## Risultati (intervallo al 95% bootstrap per id)

| | Zero-shot 32k | Fine-tuned fase 1 | ft2 (giro B) | **ft3 (giro C)** |
|---|---|---|---|---|
| Domotica, Choice, opzioni in italiano | 0,600 | 0,788 [0,700–0,875] | 0,812 | 0,788 [0,700–0,875] |
| Domotica, Choice, opzioni in inglese | 0,312 | 0,550 | 0,600 | **0,700** [0,600–0,800] |
| Domotica, sì/no in italiano, AUROC / ECE | 0,766 / 0,104 | 0,906 [0,863–0,943] / 0,063 | 0,903 / 0,140 | **0,919** [0,882–0,950] / 0,084 |
| … solo comandi fratelli, AUROC | 0,685 | 0,867 | 0,857 | 0,868 |
| Domotica, sì/no in inglese, AUROC | 0,671 | 0,853 | 0,866 | 0,888 |
| Email, Choice, opzioni in italiano / inglese | 0,625 / 0,350 | 0,625 [0,475–0,775] / 0,450 | 0,600 / 0,500 | 0,650 [0,500–0,800] / 0,575 |
| Email, sì/no in italiano, AUROC | 0,748 | 0,783 [0,724–0,842] | 0,802 | 0,768 [0,701–0,833] |
| Agente, sì/no in italiano, AUROC | 0,567 | 0,621 [0,528–0,698] | 0,655 | 0,641 [0,556–0,714] |
| Agente, sì/no in inglese, AUROC | 0,545 | 0,591 | 0,626 | 0,586 |

## Cosa ci dice

1. **Le opzioni vanno scritte nella lingua dell'utente.** Con testi italiani, le opzioni italiane danno +24 punti nella Choice domotica (0,788 contro 0,550) e +17 nelle email, anche se il modello è stato addestrato solo con opzioni inglesi. Il benchmark MASSIVE, che usa opzioni inglesi per tutte le lingue, sottostima quindi la qualità in uso reale.
2. **La domotica, il caso d'uso del Raspberry, funziona già senza esempi**: sì/no AUROC 0,906 ed ECE 0,063 con il modello della fase 1.
3. **I controlli sulle azioni degli agenti non funzionano** (AUROC circa 0,6). Il modello non ha mai visto comandi shell né domande sulle proprietà di un'azione: servono dati di addestramento di questo tipo.
4. **Email:** le categorie concrete vanno bene (fattura, riunione e spedizione hanno AUROC 1,00 come domanda sì/no), quelle astratte meno ("richiede una risposta" 0,70, "pubblicità o newsletter" 0,64). La Choice a 8 categorie è a 0,6.
5. ft2 non dà un vantaggio chiaro su questo test; nella domotica in italiano è peggio calibrato (ECE 0,140).
6. **ft3** (addestrato anche con descrizioni tradotte) è il migliore nella domotica: sì/no in italiano 0,919, Choice con opzioni inglesi da 0,550 a 0,700. Su email e agenti non cambia in modo distinguibile dagli altri (intervalli sovrapposti). I controlli sugli agenti restano il punto debole: circa 0,64; per proprietà va da 0,39 ("connettersi a internet") a 0,83 ("installare software", "solo lettura").

## Limiti

Campioni piccoli (intervalli larghi, ±0,05–0,15); un solo autore per testi ed etichette; le categorie e le proprietà sono state scelte da chi ha scritto il test. Per numeri citabili servono esempi reali: comandi della casa dell'utente, log di azioni di un agente vero, email reali anonimizzate.
