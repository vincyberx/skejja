# Annotazioni

`unseen_label_audit.csv`: controllo delle etichette MASSIVE delle 359 frasi (in inglese) dei 4 ambiti esclusi presenti nel
test originale (lette da `test_choice.jsonl`, che ha gli stessi id e testi di `test_noul.jsonl` e l'intenzione vera nel
campo `gold`) e in `test_noul_unseen_ext.jsonl` (colonna `src`) (gli id valgono per tutte e 5 le lingue). Domanda posta per ogni frase:
date le descrizioni del benchmark delle intenzioni dello stesso ambito, la descrizione dell'intenzione etichettata è
l'unica corretta?

Codici: `ok` (sì), `amb` (due descrizioni plausibili o frase vaga), `desc` (l'etichetta MASSIVE non corrisponde alla
descrizione del benchmark: indicazioni stradali per transport_query, "il locale fa consegne?" per takeaway_query),
`wrong` (la descrizione di un'intenzione sorella calza meglio), `none` (nessuna descrizione calza).

Annotatore: l'agente (Claude), 6 ottobre 2026, un solo passaggio, leggendo solo testo ed etichetta (senza i punteggi del
modello). Da rivedere a campione da una persona. Uso: metriche "etichette ok" in `results/fase2/`; il benchmark v1 resta
invariato.
