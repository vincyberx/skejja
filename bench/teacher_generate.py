"""Testi sintetici scritti dall'insegnante (Apache-2.0) per bilanciare le classi rare dei dati di ampiezza.

Perché: nella prova di etichettatura i testi naturali (Aya, mfaq, web) sono quasi tutti neutri: per Qwen3.5-9B
emozione «nessuna» nell'82% dei casi, sentiment negativo nel 4%, spam nell'1,1%. Addestrare su quelli insegnerebbe
a Skejja a dire sempre «neutro» e «no». Qui l'insegnante scrive testi brevi e realistici con una classe voluta, nelle
5 lingue; poi gli stessi testi vengono etichettati con le domande normali (bench/teacher_label.py) e la classe
voluta serve da controllo: le richieste in cui insegnante e classe voluta non concordano si scartano.

La famiglia `relazione` (coppie A/B: implicazione, contraddizione, risposta a una domanda) copre la parte di
S1MB che chiede relazioni fra due testi. È scritta da zero dall'insegnante: nessuna fonte di S1MB.

uso (workstation):
  python bench/teacher_generate.py specifiche --out synth_specs.jsonl --per_lang_scale 1.0
  python bench/teacher_generate.py genera --specs synth_specs.jsonl --model Qwen/Qwen3.5-9B --out synth_texts.jsonl
  python bench/teacher_generate.py richieste --texts synth_texts.jsonl --out requests_synth.jsonl
"""
import argparse, collections, hashlib, json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "it", "es", "de", "fr"]
NOME = {"en": "English", "it": "Italian", "es": "Spanish", "de": "German", "fr": "French"}
SOGGETTI = ("a mobile phone, a train journey, a pizza restaurant, a football match, the dentist, a summer holiday, "
            "car insurance, a laptop, a rock concert, a school meeting, the rent, the weather, a parcel delivery, a bank "
            "account, the gym, a doctor's visit, a pet cat, the garden, a recipe, the bus to work, a new job, a wedding, "
            "a hotel room, an online course, the electricity bill, a museum, a bicycle, a hairdresser, the neighbours, "
            "a birthday party, a video game, a supermarket, a flight delay, a smartwatch, a local election, a marathon, "
            "a TV series, a coffee machine, a hospital, a library, a parking fine, a mountain hike, a broken washing "
            "machine, a music festival, a used car, a cooking class, a phone contract, a job interview, a ski resort, "
            "a streaming service, a city park, a lost wallet, a tax return, a new apartment, a vaccination, a chess "
            "tournament, a camping trip, an airline, a bakery").split(", ")
LUNGHEZZE = ["one sentence", "two sentences", "two or three sentences", "three or four sentences"]
# (famiglia, classe) -> (descrizione per l'insegnante, generi, quanti per lingua a scala 1)
SPEC = {
    ("sentiment", "positivo"): ("clearly positive: the writer is satisfied, pleased or recommends it", "R", 200),
    ("sentiment", "negativo"): ("clearly negative: the writer is dissatisfied, disappointed or critical", "R", 200),
    ("sentiment", "neutro"): ("purely factual, reporting information with no opinion at all", "R", 100),
    ("emozione", "gioia"): ("written by someone who clearly feels joy or enthusiasm", "P", 100),
    ("emozione", "tristezza"): ("written by someone who clearly feels sad or disappointed", "P", 100),
    ("emozione", "rabbia"): ("written by someone who is clearly angry or irritated", "P", 100),
    ("emozione", "paura"): ("written by someone who is clearly afraid or worried", "P", 100),
    ("emozione", "sorpresa"): ("written by someone who is clearly surprised", "P", 100),
    ("emozione", "affetto"): ("written by someone who clearly feels affection or gratitude", "P", 100),
    ("emozione", "niente"): ("neutral and practical, showing no particular emotion", "P", 50),
    ("indesiderato", "si"): ("an unwanted message: unsolicited advertising, a scam, a phishing attempt asking to click a "
                             "link or send data, or a fake prize offer", "M", 200),
    ("indesiderato", "no"): ("a normal, legitimate message: from a friend, a family member, a colleague, or a genuine "
                             "notification from a service the person uses", "M", 200),
    ("sicurezza", "offese"): ("a rude reply that insults or belittles another person (no slurs, no threats)", "C", 120),
    ("sicurezza", "dati_personali"): ("a message that contains an invented person's full name together with their "
                                      "home address or phone number", "M", 120),
    ("sicurezza", "violenza"): ("a short news report about an assault, a fight or an accident in which people were "
                                "hurt", "N", 120),
    ("sicurezza", "innocuo"): ("a polite, harmless and friendly reply", "C", 120),
    **{("argomento", t): (f"about {d}", "T", 60) for t, d in [
        ("politica", "politics, government or public affairs"), ("sport", "sport, a match, a team or an athlete"),
        ("economia", "the economy, work, money or a company"), ("tecnologia", "science, technology or computers"),
        ("salute", "health, medicine or the body"), ("cultura", "culture or entertainment: music, film or books"),
        ("casa", "home, family or everyday life"), ("diritto", "law, rules, contracts or official documents")]},
    ("registro", "formale"): ("written in formal, polite language", "G", 80),
    ("registro", "informale"): ("written in informal, everyday language, like a chat between friends", "G", 80),
    ("registro", "tecnico"): ("written in technical, specialist language for experts", "G", 80),
    ("intento_del_testo", "domanda"): ("a question that someone asks", "G", 60),
    ("intento_del_testo", "ordine"): ("an instruction or an order that someone gives", "G", 60),
    ("intento_del_testo", "racconto"): ("someone reporting a fact or telling what happened", "G", 60),
    ("intento_del_testo", "opinione"): ("someone expressing a personal opinion", "G", 60),
    ("intento_del_testo", "lamentela"): ("someone complaining about something", "G", 60),
    ("intento_del_testo", "promozione"): ("an advertisement or someone trying to sell something", "G", 60),
}
GENERI = {
    "R": ["a short product review", "a short restaurant or hotel review", "a short film, book or music review",
          "a comment under a news article", "a message to customer service", "a social media post"],
    "P": ["a short message to a friend", "a social media post", "a short diary entry", "a comment on a forum"],
    "M": ["an SMS", "an email", "a chat message"],
    "C": ["an online comment", "a reply in an online discussion"],
    "N": ["a local news item"],
    "T": ["a short news snippet", "a forum post", "a short blog paragraph"],
    "G": ["a chat message", "an email", "a social media post", "a sentence said to a colleague"],
}
RELAZIONI = {   # classe -> (istruzione, etichette attese sulle affermazioni)
    "implica": ("B must follow logically from A: whenever A is true, B is also true. B must not just repeat A.",
                dict(implica=1, contraddice=0)),
    "contraddice": ("B must contradict A: A and B cannot both be true.", dict(implica=0, contraddice=1)),
    "neutra": ("B must be about the same subject as A, but neither follow from A nor contradict it.",
               dict(implica=0, contraddice=0)),
    "risponde": ("A must be a question. B must be a short passage that answers it.", dict(risponde=1)),
    "non_risponde": ("A must be a question. B must be a short passage on a related subject that does NOT answer it.",
                     dict(risponde=0)),
}
SISTEMA = ("You write realistic short texts for a dataset. Output only the requested text: no title, no quotation "
           "marks, no translation, no explanation.")


def specifiche(a):
    rng = random.Random(a.seed)
    out = []
    for lang in LANGS:
        for (fam, cl), (desc, g, n) in SPEC.items():
            for i in range(max(1, round(n * a.per_lang_scale))):
                genere = rng.choice(GENERI[g])
                ogg = rng.choice(SOGGETTI)
                lung = rng.choice(LUNGHEZZE)
                p = (f"Write {genere} in {NOME[lang]}. It must be {desc}. Subject: {ogg}. Length: {lung}. Use natural "
                     f"{NOME[lang]} as a native speaker would write it; invent names and details where needed.")
                out.append(dict(sid=f"syn-{lang}-{fam}-{cl}-{i}", lang=lang, famiglia=fam, classe=cl, genere=genere,
                                soggetto=ogg, prompt=p))
        for cl, (istr, _) in RELAZIONI.items():
            for i in range(max(1, round(150 * a.per_lang_scale))):
                ogg = rng.choice(SOGGETTI)
                p = (f"Write two short sentences A and B in {NOME[lang]} about {ogg}. {istr} Use natural {NOME[lang]}. "
                     "Output exactly two lines, in this format:\nA: <sentence>\nB: <sentence>")
                out.append(dict(sid=f"syn-{lang}-relazione-{cl}-{i}", lang=lang, famiglia="relazione", classe=cl,
                                genere="coppia", soggetto=ogg, prompt=p))
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(n=len(out), per_lingua=collections.Counter(r["lang"] for r in out),
                          per_famiglia=collections.Counter(r["famiglia"] for r in out)), default=dict))


def pulisci(r, t):
    t = re.sub(r"^\s*(here is|here's|ecco|voici|hier ist|aquí tienes)[^:\n]*:\s*", "", t, flags=re.I)
    t = t.strip().strip('"“”«»„').strip()
    if r["famiglia"] == "relazione":
        m = re.search(r"A\s*[:.)]\s*(.+?)\s*\n+\s*B\s*[:.)]\s*(.+)", t, flags=re.S)
        if not m:
            return None
        a_, b_ = m.group(1).strip(), m.group(2).strip().split("\n")[0].strip()
        if not a_ or not b_ or a_ == b_:
            return None
        return f"A: {a_}\nB: {b_}"
    t = re.sub(r"\s+\n", "\n", t)
    if not (20 <= len(t) <= 1000):
        return None
    return t


def genera(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    specs = [json.loads(l) for l in open(a.specs)]
    fatte = set()
    if os.path.exists(a.out):
        fatte = {json.loads(l)["sid"] for l in open(a.out)}
    da_fare = [s for s in specs if s["sid"] not in fatte]
    print(f"specifiche {len(specs)}, già fatte {len(fatte)}, da fare {len(da_fare)}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)

    def prompt(s):
        m = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": s["prompt"]}]
        x = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if "<think>" in (tok.chat_template or "") and not x.rstrip().endswith("</think>"):
            x += "<think>\n\n</think>\n\n"
        return x

    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util, enforce_eager=True,
              enable_prefix_caching=True, max_num_seqs=a.max_num_seqs, limit_mm_per_prompt={"image": 0, "video": 0},
              seed=0, attention_backend="FLASH_ATTN", kv_cache_dtype=a.kv_cache_dtype)
    scartati = collections.Counter()
    t0 = time.time(); n_ok = 0; ntok = 0
    with open(a.out, "a") as f:
        for i in range(0, len(da_fare), a.chunk):
            blocco = da_fare[i:i + a.chunk]
            sps = [SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, top_p=0.95,
                                  seed=int(hashlib.sha1(s["sid"].encode()).hexdigest()[:8], 16)) for s in blocco]
            outs = llm.generate([prompt(s) for s in blocco], sps, use_tqdm=False)
            for s, o in zip(blocco, outs):
                ntok += len(o.outputs[0].token_ids)
                t = pulisci(s, o.outputs[0].text)
                if t is None:
                    scartati["formato"] += 1
                    f.write(json.dumps(dict(sid=s["sid"], scartato="formato", grezzo=o.outputs[0].text[:300]),
                                       ensure_ascii=False) + "\n")
                    continue
                n_ok += 1
                f.write(json.dumps(dict(s, text=t, generatore=a.model), ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"blocco {i // a.chunk + 1}: {min(i + a.chunk, len(da_fare))}/{len(da_fare)}, {el:.0f} s, "
                  f"{ntok / el:.0f} token generati/s, scartati {dict(scartati)}", flush=True)
    json.dump(dict(modello=a.model, testi=n_ok, scartati=dict(scartati), secondi=round(time.time() - t0, 1),
                   token_generati=ntok, temperatura=a.temperature),
              open(os.path.splitext(a.out)[0] + "_tempi.json", "w"), indent=1)


def richieste(a):
    """Per ogni testo: la famiglia voluta (Choice se ne ha, e un'affermazione legata) + una famiglia a caso.
    Relazione: le affermazioni attese della coppia. Ogni richiesta porta `atteso` per il controllo."""
    sys.path.insert(0, HERE)
    F = json.load(open(os.path.join(HERE, "data", "broad", "families.json")))["famiglie"]
    rng = random.Random(a.seed)
    testi = [json.loads(l) for l in open(a.texts)]
    testi = [t for t in testi if "text" in t]
    # affermazione legata alla classe voluta e valore atteso
    LEGATE = {
        ("sentiment", "positivo"): [("positivo", 1), ("negativo", 0)], ("sentiment", "negativo"): [("negativo", 1), ("positivo", 0)],
        ("sentiment", "neutro"): [("solo_fatti", 1), ("negativo", 0)],
        ("emozione", "rabbia"): [("arrabbiato", 1)], ("emozione", "paura"): [("preoccupato", 1)],
        ("emozione", "gioia"): [("contento", 1), ("arrabbiato", 0)], ("emozione", "tristezza"): [("arrabbiato", None)],
        ("emozione", "sorpresa"): [("arrabbiato", None)], ("emozione", "affetto"): [("arrabbiato", 0)],
        ("emozione", "niente"): [("arrabbiato", 0), ("preoccupato", 0)],
        ("indesiderato", "si"): [("indesiderato", 1)], ("indesiderato", "no"): [("indesiderato", 0)],
        ("sicurezza", "offese"): [("offese", 1)], ("sicurezza", "dati_personali"): [("dati_personali", 1)],
        ("sicurezza", "violenza"): [("violenza", 1)], ("sicurezza", "innocuo"): [("offese", 0), ("violenza", 0)],
        ("argomento", "sport"): [("parla_di_sport", 1)], ("argomento", "economia"): [("parla_di_soldi", 1)],
        ("argomento", "salute"): [("parla_di_salute", 1)],
        ("registro", "formale"): [("formale", 1)], ("registro", "tecnico"): [("tecnico", 1)],
        ("registro", "informale"): [("formale", 0)],
        ("intento_del_testo", "domanda"): [("contiene_domanda", 1)], ("intento_del_testo", "ordine"): [("chiede_azione", 1)],
        ("intento_del_testo", "promozione"): [("promuove", 1)],
    }
    MAPPA_CHOICE = {("indesiderato", "si"): None, ("sicurezza", "innocuo"): None}
    con_opzioni = [k for k, v in F.items() if v["opzioni"] and k != "lingua"]
    con_aff = [k for k, v in F.items() if v["affermazioni"] and k not in ("lingua", "relazione")]
    out = []

    def ql(lang):
        return lang if (lang != "en" and rng.random() < a.p_domanda_nativa) else "en"

    def choice(t, fam, atteso):
        lq = ql(t["lang"]); opz = sorted(F[fam]["opzioni"]); rng.shuffle(opz)
        return dict(rid=f"{t['sid']}-{fam}-choice", text_id=t["sid"], lang=t["lang"], text=t["text"], famiglia=fam,
                    tipo="choice", lang_domanda=lq, domanda=F[fam]["domanda"][lq], opzioni=opz,
                    opzioni_desc=[F[fam]["opzioni"][o][lq] for o in opz], gold=None, gold_da="insegnante",
                    atteso=atteso, uso="addestramento", origine="sintetico")

    def noul(t, fam, aff, atteso):
        lq = ql(t["lang"])
        return dict(rid=f"{t['sid']}-{fam}-{aff}", text_id=t["sid"], lang=t["lang"], text=t["text"], famiglia=fam,
                    tipo="noul", lang_domanda=lq, domanda=F[fam]["domanda"][lq], affermazione=aff,
                    affermazione_desc=F[fam]["affermazioni"][aff][lq], gold=None, gold_da="insegnante",
                    atteso=atteso, uso="addestramento", origine="sintetico")

    for t in testi:
        fam, cl = t["famiglia"], t["classe"]
        if fam == "relazione":
            for aff, v in RELAZIONI[cl][1].items():
                out.append(noul(t, "relazione", aff, v))
            continue
        if F[fam]["opzioni"] and (fam, cl) not in MAPPA_CHOICE:
            out.append(choice(t, fam, cl))
        for aff, v in LEGATE.get((fam, cl), [])[:1]:
            out.append(noul(t, fam, aff, v))
        altra = rng.choice([f for f in con_opzioni if f != fam])
        out.append(choice(t, altra, None))
        fa = rng.choice([f for f in con_aff if f != fam])
        out.append(noul(t, fa, rng.choice(sorted(F[fa]["affermazioni"])), None))
    with open(a.out, "w") as f:
        for r in sorted(out, key=lambda r: (r["text_id"], r["rid"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(n=len(out), testi=len(testi), con_atteso=sum(r["atteso"] is not None for r in out),
                          per_tipo=collections.Counter(r["tipo"] for r in out)), default=dict))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("specifiche"); s.add_argument("--out", required=True)
    s.add_argument("--per_lang_scale", type=float, default=1.0); s.add_argument("--seed", type=int, default=23)
    g = sub.add_parser("genera")
    g.add_argument("--specs", required=True); g.add_argument("--model", required=True); g.add_argument("--out", required=True)
    g.add_argument("--temperature", type=float, default=0.9); g.add_argument("--max_tokens", type=int, default=160)
    g.add_argument("--max_model_len", type=int, default=1024); g.add_argument("--gpu_util", type=float, default=0.92)
    g.add_argument("--max_num_seqs", type=int, default=128); g.add_argument("--chunk", type=int, default=2000)
    g.add_argument("--kv_cache_dtype", default="auto")
    r = sub.add_parser("richieste")
    r.add_argument("--texts", required=True); r.add_argument("--out", required=True)
    r.add_argument("--p_domanda_nativa", type=float, default=0.5); r.add_argument("--seed", type=int, default=29)
    a = ap.parse_args()
    {"specifiche": specifiche, "genera": genera, "richieste": richieste}[a.cmd](a)


if __name__ == "__main__":
    main()
