"""Etichette dell'insegnante (modello aperto, Apache-2.0) per i dati di ampiezza della fase 2.

Tre passi:
  prepara  (Mac)          richieste della prova + metro di qualità + varianti di stabilità -> un solo JSONL
  etichetta (workstation) vLLM, un solo token generato, probabilità lette dai logprob -> etichette JSONL, con ripresa
  (analisi: bench/teacher_analysis.py, sul Mac -> results/fase2/teacher_trial.csv e _riepilogo.json)

Il metro di qualità usa lo split di TRAINING di quattro dataset dell'articolo su Jev (SST-2, AG News, Emotion, SMS
Spam), che hanno un'etichetta vera. Regola (docs/fase2_ampiezza.md §1, punto 4): servono SOLO a misurare
l'insegnante, non entrano mai nei dati di Skejja. Le loro righe hanno `uso = "solo_metro"` e build_broad.py le
esclude dal pool.

Formato della richiesta all'insegnante: istruzione in inglese, testo, poi domanda e opzioni nella lingua scelta
per Skejja (`lang_domanda`): l'insegnante giudica esattamente le descrizioni che vedrà Skejja. Per le Choice la
risposta è una lettera (A, B, …); per le sì/no «yes» o «no». Il testo viene prima della domanda, così vLLM riusa la
parte comune fra le richieste sullo stesso testo (prefix caching).

uso:
  python bench/teacher_label.py prepara --requests bench/data/broad/requests_trial.jsonl --out bench/data/broad/teacher_trial_in.jsonl
  python bench/teacher_label.py etichetta --requests teacher_trial_in.jsonl --model RedHatAI/Qwen3.8-27B-INT4 --out labels_qwen27b.jsonl
  python bench/teacher_analysis.py --requests bench/data/broad/teacher_trial_in.jsonl --labels a.jsonl b.jsonl --out results/fase2/teacher_trial.csv
"""
import argparse, collections, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERE = "ABCDEFGH"
LANGS = ["en", "it", "es", "de", "fr"]
SISTEMA = ("You annotate texts for a dataset. Read the text and answer the question about it. Judge only what the "
           "text itself says or shows. The question and the options may be in a language other than English; "
           "that does not matter. Reply with the answer only, nothing else.")
# metro di qualità: (repo, prefisso dei file di TRAINING, colonna testo, colonna etichetta, famiglia, tipo, mappa)
# AG News «World» -> «politica» è una corrispondenza approssimata (le notizie dal mondo sono soprattutto politica):
# l'accuratezza su AG News è un limite inferiore della qualità dell'insegnante, non un numero esatto.
METRO = {
    "sst2": ("stanfordnlp/sst2", "data/train", "sentence", "label", "sentiment", "choice",
             {0: "negativo", 1: "positivo"}),
    "agnews": ("fancyzhx/ag_news", "data/train", "text", "label", "argomento", "choice",
               {0: "politica", 1: "sport", 2: "economia", 3: "tecnologia"}),
    "emotion": ("dair-ai/emotion", "split/train", "text", "label", "emozione", "choice",
                {0: "tristezza", 1: "gioia", 2: "affetto", 3: "rabbia", 4: "paura", 5: "sorpresa"}),
    "sms": ("ucirvine/sms_spam", "plain_text/train", "sms", "label", "indesiderato", "noul", {0: 0, 1: 1}),
}


def famiglie():
    return json.load(open(os.path.join(HERE, "data", "broad", "families.json")))["famiglie"]


def richiesta(F, base, fam, tipo, lq, rng, opzioni=None, aff=None):
    r = dict(base, famiglia=fam, tipo=tipo, lang_domanda=lq, domanda=F[fam]["domanda"][lq])
    if tipo == "choice":
        if opzioni is None:
            opzioni = sorted(F[fam]["opzioni"])
            rng.shuffle(opzioni)
        r.update(opzioni=opzioni, opzioni_desc=[F[fam]["opzioni"][o][lq] for o in opzioni])
    else:
        r.update(affermazione=aff, affermazione_desc=F[fam]["affermazioni"][aff][lq])
    return r


def prepara(a):
    rng = random.Random(a.seed)
    F = famiglie()
    base = [json.loads(l) for l in open(a.requests)]
    for r in base:
        r["uso"] = "addestramento"
    out = list(base)
    # varianti di stabilità sui testi del pool
    for r in base:
        if r["tipo"] == "choice" and rng.random() < a.p_rev:
            v = dict(r, rid=r["rid"] + "#rev", uso="variante_ordine", di=r["rid"],
                     opzioni=r["opzioni"][::-1], opzioni_desc=r["opzioni_desc"][::-1])
            out.append(v)
        if rng.random() < a.p_lang:
            altra = r["lang"] if r["lang_domanda"] == "en" and r["lang"] != "en" else (
                "en" if r["lang"] != "en" else rng.choice(LANGS[1:]))
            if altra != r["lang_domanda"]:
                key = "affermazione" if r["tipo"] == "noul" else None
                v = richiesta(F, {k: r[k] for k in ("text_id", "lang", "text", "gold", "gold_da")}, r["famiglia"],
                              r["tipo"], altra, rng, opzioni=r.get("opzioni"), aff=r.get(key) if key else None)
                v.update(rid=r["rid"] + "#lang", uso="variante_lingua", di=r["rid"])
                out.append(v)
    # metro di qualità con etichetta vera (solo per misurare l'insegnante)
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi()
    for nome, (repo, prefisso, col_t, col_y, fam, tipo, mappa) in METRO.items():
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                       if f.endswith(".parquet") and f.startswith(prefisso))
        assert files, (repo, prefisso)
        righe = []
        for f in files:
            righe += pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=[col_t, col_y]).to_pylist()
        rng.shuffle(righe)
        if tipo == "noul":   # bilanciato: metà positivi
            pos = [x for x in righe if x[col_y] == 1][: a.n_metro // 2]
            neg = [x for x in righe if x[col_y] == 0][: a.n_metro - len(pos)]
            righe = pos + neg
        else:
            righe = [x for x in righe if x[col_y] in mappa][: a.n_metro]
        for i, x in enumerate(righe):
            lq = "en" if rng.random() < 0.5 else rng.choice(LANGS[1:])
            b = dict(text_id=f"metro-{nome}-{i}", lang="en", text=str(x[col_t]).strip(), gold=mappa[x[col_y]],
                     gold_da=f"metro:{nome}", uso="solo_metro")
            aff = "indesiderato" if tipo == "noul" else None
            r = richiesta(F, b, fam, tipo, lq, rng, aff=aff)
            r["rid"] = f"metro-{nome}-{i}-{fam}-{tipo}"
            out.append(r)
        print(f"metro {nome}: {len(righe)} da {files}", flush=True)
    out.sort(key=lambda r: (r["text_id"], r["rid"]))
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(file=a.out, n=len(out), per_uso=collections.Counter(r["uso"] for r in out)),
                     ensure_ascii=False, default=dict))


def corpo(r):
    """Testo da mostrare all'insegnante. Dal giro G le richieste hanno lo `stato` separato dall'istruzione
    (resa «istruzioni + stato + criteri»): qui si mostra solo lo stato, l'istruzione va nella domanda."""
    stato = r.get("stato")
    if not stato:
        return r["text"]
    if isinstance(stato, str):
        return stato
    return "\n".join(f"{k}: {v}" for k, v in stato.items() if v)


def messaggio(r):
    domanda = r.get("domanda") or r.get("istruzioni", "")
    if r["tipo"] in ("choice", "score"):
        righe = "\n".join(f"{LETTERE[i]}. {d}" for i, d in enumerate(r["opzioni_desc"]))
        coda = (f"Question: {domanda}\nOptions:\n{righe}\n\n"
                f"Answer with the letter of the best option ({', '.join(LETTERE[:len(r['opzioni_desc'])])}).")
    else:
        coda = (f"{domanda}\nStatement: {r['affermazione_desc']}\n\n"
                "Is the statement true for this text? Answer yes or no.")
    return [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"Text:\n\"\"\"\n{corpo(r)}\n\"\"\"\n\n{coda}"}]


def leggi_logprob(r, lp):
    """lp: dict token_id -> Logprob del primo token generato. Restituisce probabilità sulle risposte valide."""
    massa = collections.defaultdict(float)
    for v in lp.values():
        t = (v.decoded_token or "").strip()
        p = math.exp(v.logprob)
        if r["tipo"] in ("choice", "score"):
            t = t.upper().rstrip(".):")
            if len(t) == 1 and t in LETTERE[:len(r["opzioni"])]:
                massa[t] += p
        else:
            t = t.lower().rstrip(".!,")
            if t in ("yes", "no"):
                massa[t] += p
    cop = sum(massa.values())
    if r["tipo"] in ("choice", "score"):
        probs = {o: (massa.get(LETTERE[i], 0.0) / cop if cop > 0 else 1 / len(r["opzioni"]))
                 for i, o in enumerate(r["opzioni"])}
        return dict(p=probs, copertura=cop)
    return dict(p_yes=(massa["yes"] / cop if cop > 0 else 0.5), copertura=cop)


def etichetta(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    reqs = [json.loads(l) for l in open(a.requests)]
    if a.limit:
        reqs = reqs[: a.limit]
    fatte = set()
    if os.path.exists(a.out):
        fatte = {json.loads(l)["rid"] for l in open(a.out)}
    da_fare = [r for r in reqs if r["rid"] not in fatte]
    print(f"richieste {len(reqs)}, già fatte {len(fatte)}, da fare {len(da_fare)}", flush=True)
    if not da_fare:
        return
    tok = AutoTokenizer.from_pretrained(a.model)

    def prompt(r):
        s = tok.apply_chat_template(messaggio(r), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if "<think>" in (tok.chat_template or "") and not s.rstrip().endswith("</think>"):
            s += "<think>\n\n</think>\n\n"
        return s

    print("esempio di prompt:\n" + prompt(da_fare[0]), flush=True)
    t0 = time.time()
    # FLASH_ATTN: kernel già compilati in vLLM. FlashInfer (scelto da solo per i modelli ibridi) compila i kernel
    # al primo uso e ha bisogno di nvcc, che sulla workstation non c'è.
    # kv_cache_dtype: alcuni checkpoint quantizzati (es. RedHatAI/Qwen3.8-27B-INT4) chiedono una cache KV in FP8,
    # che la FlashAttention 2 di Ampere (sm_86) non supporta; "bfloat16" la riporta al tipo del modello.
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=True, enforce_eager=a.enforce_eager, max_num_seqs=a.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, attention_backend=a.attention_backend,
              kv_cache_dtype=a.kv_cache_dtype)
    caricamento = time.time() - t0
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    tempi = dict(modello=a.model, caricamento_s=round(caricamento, 1), blocchi=[])
    with open(a.out, "a") as f:
        for i in range(0, len(da_fare), a.chunk):
            blocco = da_fare[i:i + a.chunk]
            prompts = [prompt(r) for r in blocco]
            t1 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.time() - t1
            ntok, coperture = 0, []
            for r, o in zip(blocco, outs):
                lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
                rec = dict(rid=r["rid"], modello=a.model, top=o.outputs[0].text, n_prompt=len(o.prompt_token_ids),
                           **leggi_logprob(r, lp))
                ntok += rec["n_prompt"]
                coperture.append(rec["copertura"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i == 0 and sum(coperture) / len(coperture) < a.min_copertura:
                primi = collections.Counter(json.loads(l)["top"] for l in open(a.out)).most_common(8)
                print(f"FORMATO: copertura media {sum(coperture) / len(coperture):.3f} < {a.min_copertura}; "
                      f"primi token più frequenti {primi}. Mi fermo.", flush=True)
                raise SystemExit(3)
            tempi["blocchi"].append(dict(n=len(blocco), secondi=round(dt, 2), token_prompt=ntok))
            print(f"blocco {i // a.chunk + 1}: {len(blocco)} richieste in {dt:.1f} s "
                  f"({len(blocco) / dt:.1f} rich/s, {ntok / dt:.0f} token/s)", flush=True)
    n = sum(b["n"] for b in tempi["blocchi"]); s = sum(b["secondi"] for b in tempi["blocchi"])
    tempi.update(richieste=n, secondi_generazione=round(s, 1), richieste_al_secondo=round(n / s, 2),
                 token_prompt_al_secondo=round(sum(b["token_prompt"] for b in tempi["blocchi"]) / s, 1),
                 vllm=dict(max_model_len=a.max_model_len, gpu_util=a.gpu_util, enforce_eager=a.enforce_eager,
                           max_num_seqs=a.max_num_seqs, prefix_caching=True,
                           attention_backend=a.attention_backend, kv_cache_dtype=a.kv_cache_dtype))
    json.dump(tempi, open(os.path.splitext(a.out)[0] + "_tempi.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in tempi.items() if k != "blocchi"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepara")
    p.add_argument("--requests", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_metro", type=int, default=400)
    p.add_argument("--p_rev", type=float, default=0.15)
    p.add_argument("--p_lang", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=11)
    e = sub.add_parser("etichetta")
    e.add_argument("--requests", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--max_model_len", type=int, default=2048)
    e.add_argument("--gpu_util", type=float, default=0.94)
    e.add_argument("--max_num_seqs", type=int, default=64)
    e.add_argument("--enforce_eager", action="store_true")
    e.add_argument("--attention_backend", default="FLASH_ATTN")
    e.add_argument("--kv_cache_dtype", default="auto")
    e.add_argument("--chunk", type=int, default=1000)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--min_copertura", type=float, default=0.5,
                   help="dopo il primo blocco: se la probabilità media sulle risposte valide è sotto, ci si ferma")
    a = ap.parse_args()
    {"prepara": prepara, "etichetta": etichetta}[a.cmd](a)


if __name__ == "__main__":
    main()
