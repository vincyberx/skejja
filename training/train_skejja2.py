"""Fase 2: fine-tuning Choice + Noul su dati multi-sorgente, con negativi fratelli e parafrasi delle descrizioni.

Dati (tutti da bench/data, nessun file di test):
- MASSIVE train, solo le 49 intenzioni viste (le 10 dei 4 ambiti esclusi non compaiono mai, né come frasi né come
  domande/opzioni: verificato con assert);
- bench/data/ext/ext_utterances.jsonl (CLINC150, Banking77, MINDS-14; solo intenzioni con role=train) e le
  traduzioni OPUS-MT di bench/data/ext/mt_cache.jsonl;
- descrizioni e parafrasi da bench/data/ext/descriptions.json (per MASSIVE la prima è quella del benchmark).
Esempi Noul: per ogni frase ancora un positivo (descrizione vera) e un negativo fratello (stesso gruppo), più con
probabilità --p_other un negativo di un altro gruppo compatibile (vedi `other_ok`). Le frasi out-of-scope di CLINC
danno solo negativi (domande CLINC). Esempi Choice: oro + fino a 3 fratelli + altri compatibili, 8 opzioni.
Perdite come in fase 1 (Choice: CE + 0,5·BCE + kd_w·KL; Noul: BCE + kd_w·BCE sul teacher).
Con --p_desc_native > 0 domanda e opzioni sono a volte nella lingua della frase (descrizioni tradotte con OPUS-MT,
bench/data/ext/descriptions_mt.json). Con --agent_dir si aggiungono esempi sulle proprietà delle azioni (bench/data/agent, da tldr-pages) e lo sviluppo
tldr entra nella media di selezione. Selezione: ogni --eval_every passi AUROC Noul su bench/data/dev_noul_ext.jsonl
(domini CLINC tenuti fuori; con --p_desc_native la media fra domande inglesi e nella lingua della frase); si
salva il checkpoint migliore (`best/`) e l'ultimo (`last/`).

uso: python training/train_skejja2.py --init releases/checkpoints/edge32k-zs --out releases/checkpoints/run_A \
        --teacher releases/checkpoints/edge32k-zs --kd_w 0.2
"""
import argparse, collections, json, math, os, random, shutil, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import skejja_measure as fm  # noqa: E402
import skejja_glic as fg  # noqa: E402
from skejja_measure import open_mt_cache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default=os.path.join(HERE, "..", "bench", "data"))
ap.add_argument("--teacher", default="")
ap.add_argument("--kd_w", type=float, default=0.2)
ap.add_argument("--n_noul_anchor", type=int, default=24000, help="frasi ancora Noul (ognuna dà 2-3 esempi)")
ap.add_argument("--n_choice", type=int, default=24000)
ap.add_argument("--p_other", type=float, default=0.5)
ap.add_argument("--p_para", type=float, default=0.5, help="probabilità di usare una parafrasi invece della descrizione principale")
ap.add_argument("--p_en_mt", type=float, default=0.3, help="frasi tradotte: probabilità di tenere l'inglese")
ap.add_argument("--p_desc_native", type=float, default=0.0,
                help="frasi non inglesi: probabilità di scrivere domanda/opzioni nella lingua della frase (descriptions_mt.json)")
ap.add_argument("--mix", default="massive:0.40,clinc:0.25,banking77:0.25,minds14:0.10")
ap.add_argument("--agent_dir", default="", help="bench/data/agent: esempi sulle proprietà delle azioni (tldr-pages)")
ap.add_argument("--n_agent", type=int, default=12000, help="esempi Noul sulle azioni (bilanciati per proprietà ed etichetta)")
ap.add_argument("--kd_agent", type=int, default=1, help="1 = distillazione anche sugli esempi sulle azioni (giro D); 0 = no")
ap.add_argument("--broad_requests", nargs="*", default=[],
                help="dati di ampiezza (fase 2): richieste JSONL di bench/teacher_label.py / teacher_generate.py")
ap.add_argument("--broad_labels", nargs="*", default=[], help="etichette dell'insegnante (JSONL) per quelle richieste")
ap.add_argument("--broad_teacher", default="", help="usa solo le etichette di questo insegnante (campo `modello`)")
ap.add_argument("--n_broad_choice", type=int, default=-1, help="esempi Choice di ampiezza (-1 = tutti)")
ap.add_argument("--n_broad_noul", type=int, default=-1, help="esempi sì/no di ampiezza (-1 = tutti)")
ap.add_argument("--broad_dev_frac", type=float, default=0.05, help="quota di testi (per text_id) tenuta per lo sviluppo")
ap.add_argument("--broad_min_cop", type=float, default=0.9, help="copertura minima della risposta dell'insegnante")
ap.add_argument("--n_broad_score", type=int, default=-1, help="esempi Score di ampiezza (-1 = tutti)")
ap.add_argument("--broad_caps", default="",
                help="tetti per fonte, applicati prima dei tetti per tipo: 'noul/circa=4000,choice/casa=6000'. "
                     "Chiave = tipo/origine (origine assente = naturale), le stesse chiavi di broad_stats. "
                     "Giro H: il tetto unico del giro G aveva diluito le famiglie del giro F")
ap.add_argument("--bnoul_budget", type=int, default=0,
                help=">0: le sì/no di ampiezza vanno in batch propri con questo tetto di token, invece che nei "
                     "batch sì/no da --bs con padding sul più lungo (giro G: 8-10 s per passo)")
ap.add_argument("--lambda_score", type=float, default=0.5,
                help="peso dell'errore sul valore atteso nella perdita delle Score (la metrica di S1MB)")
ap.add_argument("--token_budget", type=int, default=16384,
                help="token per batch sugli esempi di ampiezza lunghi o con molte opzioni: il batch si "
                     "accorcia quando testo e opzioni crescono, cosi la memoria resta costante")
ap.add_argument("--kd_broad", type=int, default=0, help="1 = distillazione dal teacher Skejja anche sugli esempi di ampiezza")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--head_lr", type=float, default=1e-4)
ap.add_argument("--max_len", type=int, default=192)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--eval_n", type=int, default=2500, help="item di dev_noul_ext usati per la selezione")
ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
args = ap.parse_args()

from gliclass import GLiClassModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

torch.set_num_threads(args.threads)
DEV = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
rng = random.Random(args.seed)
D = args.data

# ---------------- intenzioni, descrizioni, gruppi
J = json.load(open(os.path.join(D, "ext", "descriptions.json")))
INT = {k: v for k, v in J["intents"].items() if v["role"] == "train"}
NO_NEG = {frozenset(p) for p in J["no_negative_pairs"]}
SOURCE = lambda k: k.split(":", 1)[0]
FAMILY_BANK = {"clinc_banking", "clinc_credit", "banking77", "minds14"}
family = lambda k: "bank" if INT[k]["group"] in FAMILY_BANK else "assistant"
assert not any(SOURCE(k) == "massive" and fm.is_unseen(k.split(":", 1)[1]) for k in INT)


def other_ok(a, b):
    """b può fare da negativo 'di altro gruppo' per una frase di a? Stessa fonte e gruppo diverso, oppure fonti diverse
    di famiglie diverse (assistente vs banca): due fonti della stessa famiglia hanno intenzioni sovrapposte
    (es. MASSIVE weather_query e CLINC weather; Banking77 e MINDS-14 balance)."""
    if a == b or frozenset((a, b)) in NO_NEG or INT[a]["group"] == INT[b]["group"]:
        return False
    return SOURCE(a) == SOURCE(b) or family(a) != family(b)


GROUP = collections.defaultdict(list)
for k, v in INT.items():
    GROUP[v["group"]].append(k)
SIBS = {k: [s for s in GROUP[INT[k]["group"]] if s != k and frozenset((k, s)) not in NO_NEG] for k in INT}
OTHERS = {k: [o for o in INT if other_ok(k, o)] for k in INT}
CLINC_Q = [k for k in INT if SOURCE(k) == "clinc"]


DMT = {}
if args.p_desc_native > 0:
    DMT = json.load(open(os.path.join(D, "ext", "descriptions_mt.json")))["translations"]


def describe(k, lang="en"):
    d = INT[k]["desc"]
    s = d[0] if (len(d) == 1 or rng.random() >= args.p_para) else rng.choice(d[1:])
    return DMT[s][lang] if lang != "en" else s


def opt_lang(lang):
    """lingua di domanda/opzioni per un esempio: quella della frase con probabilità p_desc_native, altrimenti inglese"""
    return lang if (lang != "en" and lang in ("it", "es", "de", "fr") and DMT and rng.random() < args.p_desc_native) else "en"


# ---------------- frasi
mt = collections.defaultdict(dict)
for l in open_mt_cache(os.path.join(D, "ext")):
    r = json.loads(l); mt[r["src"]][r["lang"]] = r["text"]
pool = collections.defaultdict(list)   # source -> [(intent or None, {lang: text})]
for l in open(os.path.join(D, "train_utterances.jsonl")):
    r = json.loads(l)
    if fm.is_unseen(r["intent"]):
        continue
    pool["massive"].append(("massive:" + r["intent"], {r["lang"]: r["text"]}))
for l in open(os.path.join(D, "ext", "ext_utterances.jsonl")):
    r = json.loads(l)
    texts = {r["lang"]: r["text"]}
    if r["lang"] == "en" and r["source"] != "minds14":
        texts.update(mt.get(r["text"], {}))
    pool[r["source"]].append((None if r["neg_only"] else r["intent"], texts))
for s, rr in pool.items():
    assert all(i is None or i in INT for i, _ in rr), s
mix = {k: float(v) for k, v in (x.split(":") for x in args.mix.split(","))}
srcs = sorted(mix); w = np.array([mix[s] for s in srcs]); w = w / w.sum()


def pick_text(texts):
    if len(texts) == 1:
        return next(iter(texts.items()))
    if rng.random() < args.p_en_mt and "en" in texts:
        return "en", texts["en"]
    l = rng.choice(sorted(l for l in texts if l != "en"))
    return l, texts[l]


def anchor():
    s = srcs[int(np.searchsorted(np.cumsum(w), rng.random()))]
    it, texts = rng.choice(pool[s])
    lang, t = pick_text(texts)
    return s, it, t, lang


def make_noul():
    s, g, t, lang = anchor()
    if g is None:   # out-of-scope CLINC: solo negativi
        return [(fg.fmt(t, [describe(q, opt_lang(lang))]), 0) for q in rng.sample(CLINC_Q, 2)]
    ex = [(fg.fmt(t, [describe(g, opt_lang(lang))]), 1)]
    if SIBS[g]:
        ex.append((fg.fmt(t, [describe(rng.choice(SIBS[g]), opt_lang(lang))]), 0))
    if rng.random() < args.p_other or not SIBS[g]:
        ex.append((fg.fmt(t, [describe(rng.choice(OTHERS[g]), opt_lang(lang))]), 0))
    return ex


def make_choice():
    while True:
        s, g, t, lang = anchor()
        if g is not None:
            break
    ol = opt_lang(lang)
    opts = rng.sample(SIBS[g], min(3, len(SIBS[g])))
    rest = [o for o in OTHERS[g] if o not in opts]
    opts += rng.sample(rest, 8 - 1 - len(opts))
    opts.append(g); rng.shuffle(opts)
    return fg.fmt(t, [describe(o, ol) for o in opts]), opts.index(g)


noul_ex = [e for _ in range(args.n_noul_anchor) for e in make_noul()]
choice_ex = [make_choice() for _ in range(args.n_choice)]
# ---------------- azioni di un agente (proprietà sì/no, etichette per regole sui comandi tldr)
AG_PROPS = {}
agent_dev = []
if args.agent_dir:
    AG_PROPS = json.load(open(os.path.join(args.agent_dir, "props.json")))["props"]
    ag = [json.loads(l) for l in open(os.path.join(args.agent_dir, "tldr_train.jsonl"))]
    AIDX = collections.defaultdict(list)
    for r in ag:
        for k, v in r["labels"].items():
            AIDX[(k, v)].append(r)

    def prop_q(k, lang):
        qs = AG_PROPS[k][lang]
        return qs[0] if rng.random() >= args.p_para else rng.choice(qs[1:])

    def make_agent():
        k = rng.choice(sorted(AG_PROPS)); lab = int(rng.random() < 0.5)
        if not AIDX[(k, lab)]:
            lab = 1 - lab
        r = rng.choice(AIDX[(k, lab)])
        kinds = ["cmd", "both"] if k == "sudo" else ["cmd", "desc", "both"]
        kind = rng.choice(kinds)
        lang = rng.choice(sorted(r["desc"]))
        if kind == "cmd":
            text = r["cmd"]; ql = "en" if rng.random() < 0.5 else rng.choice(["it", "es", "de", "fr"])
        else:
            d = r["desc"][lang]
            text = d if kind == "desc" else f"{d}: {r['cmd']}"
            ql = lang if (lang != "en" and rng.random() < 0.6) else ("en" if lang != "en" else rng.choice(["en", "en", "it", "es", "de", "fr"]))
        return fg.fmt(text, [prop_q(k, ql)]), lab, 1   # terzo campo: esempio sulle azioni (niente distillazione)

    noul_ex += [make_agent() for _ in range(args.n_agent)]
    agent_dev = [json.loads(l) for l in open(os.path.join(args.agent_dir, "tldr_dev.jsonl"))]
# ---------------- dati di ampiezza (fase 2): etichette morbide di un insegnante aperto
# noul: (testo, p_sì, "broad"); bchoice: (testo, [p per opzione]). Le righe `solo_metro` (split di training dei
# dataset dell'articolo su Jev, usate solo per misurare l'insegnante) non entrano mai.
broad_choice, broad_noul, broad_score, broad_dev = [], [], [], []
broad_stats = collections.Counter()
if args.broad_requests:
    import hashlib
    LAB = {}
    for path in args.broad_labels:
        for l in open(path):
            x = json.loads(l)
            if args.broad_teacher and x.get("modello") != args.broad_teacher:
                continue
            LAB.setdefault(x["rid"], x)
    in_dev = lambda tid: int(hashlib.sha1(tid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < args.broad_dev_frac
    metro_testi, usati = set(), set()
    for path in args.broad_requests:
        for l in open(path):
            r = json.loads(l)
            if r.get("uso") == "solo_metro":
                metro_testi.add(r["text"].strip().lower())
                broad_stats["escluse_metro"] += 1; continue
            usati.add(r["text"].strip().lower())
            lab = LAB.get(r["rid"])
            if r.get("gold_da") == "corpus":   # etichetta vera dal corpus (lingua, pertinenza, Circa, giudizi)
                if r["tipo"] in ("choice", "score"):
                    tgt = [1.0 if str(o) == str(r["gold"]) else 0.0 for o in r["opzioni"]]
                    if sum(tgt) != 1.0:
                        broad_stats["gold_non_fra_le_opzioni"] += 1; continue
                else:
                    tgt = float(r["gold"])
            else:
                if lab is None:
                    broad_stats["senza_etichetta"] += 1; continue
                if lab["copertura"] < args.broad_min_cop:
                    broad_stats["copertura_bassa"] += 1; continue
                if r["tipo"] in ("choice", "score"):
                    tgt = [float(lab["p"][o]) for o in r["opzioni"]]
                    if r.get("atteso") is not None and r["opzioni"][int(np.argmax(tgt))] != r["atteso"]:
                        broad_stats["disaccordo_con_classe_voluta"] += 1; continue
                else:
                    tgt = float(lab["p_yes"])
                    if r.get("atteso") is not None and int(tgt >= 0.5) != int(r["atteso"]):
                        broad_stats["disaccordo_con_classe_voluta"] += 1; continue
            labels = r["opzioni_desc"] if r["tipo"] in ("choice", "score") else [r["affermazione_desc"]]
            ex = (fg.fmt(r["text"], labels), tgt)
            if r["tipo"] == "score":     # i valori dei livelli viaggiano con l'esempio: servono alla perdita
                ex = (ex[0], ex[1], [float(v) for v in r["valori"]])
            chiave = f"{r['tipo']}/{r.get('origine') or 'naturale'}"
            if in_dev(r["text_id"]):
                broad_dev.append((r["tipo"], ex)); broad_stats["sviluppo"] += 1
            elif r["tipo"] == "choice":
                broad_choice.append((chiave, ex))
            elif r["tipo"] == "score":
                broad_score.append((chiave, ex))
            else:
                broad_noul.append((chiave, (ex[0], ex[1], "broad")))
            broad_stats[chiave] += 1
    # tetti per fonte (giro H): stesso ordine casuale per tutti, poi si tengono i primi n di ogni chiave
    tetti = {k.strip(): int(v) for k, v in (x.split("=") for x in args.broad_caps.split(",") if x.strip())}
    ignoti = set(tetti) - {k for lst in (broad_choice, broad_noul, broad_score) for k, _ in lst}
    assert not ignoti, f"chiavi di --broad_caps senza esempi: {sorted(ignoti)}"
    broad_tenuti = collections.Counter()
    def applica_tetti(lst):
        rng.shuffle(lst); visti, out = collections.Counter(), []
        for k, e in lst:
            if k in tetti and visti[k] >= tetti[k]:
                continue
            visti[k] += 1; out.append(e)
        broad_tenuti.update(visti)
        return out
    broad_choice, broad_noul, broad_score = map(applica_tetti, (broad_choice, broad_noul, broad_score))
    rng.shuffle(broad_choice); rng.shuffle(broad_noul)
    if args.n_broad_choice >= 0:
        broad_choice = broad_choice[:args.n_broad_choice]
    if args.n_broad_noul >= 0:
        broad_noul = broad_noul[:args.n_broad_noul]
    rng.shuffle(broad_score)
    if args.n_broad_score >= 0:
        broad_score = broad_score[:args.n_broad_score]
    assert not (metro_testi & usati), f"{len(metro_testi & usati)} testi del metro fra i dati di addestramento"
    if args.bnoul_budget <= 0:
        noul_ex += broad_noul
    print(json.dumps(dict(broad=dict(broad_stats), tenuti=dict(broad_tenuti), broad_choice=len(broad_choice),
                          broad_noul=len(broad_noul), broad_score=len(broad_score), broad_dev=len(broad_dev))), flush=True)
rng.shuffle(noul_ex)
# nessuna descrizione delle 10 intenzioni mai viste deve comparire negli input
bench_desc = json.load(open(os.path.join(D, "intent_descriptions.json")))
unseen_desc = {bench_desc[i] for i in bench_desc if fm.is_unseen(i)}
all_labels = set()
for x in [e[0] for e in noul_ex] + [c[0] for c in choice_ex] + [c[0] for c in broad_choice] + [d[1][0] for d in broad_dev]:
    all_labels.update(p for p in x.split("<<SEP>>")[0].split("<<LABEL>>") if p)
assert not (all_labels & unseen_desc), all_labels & unseen_desc

# ---------------- modello
tok = AutoTokenizer.from_pretrained(args.init)
model = GLiClassModel.from_pretrained(args.init, dtype=torch.float32).to(DEV)
teacher = None
if args.teacher and args.kd_w > 0:
    teacher = GLiClassModel.from_pretrained(args.teacher, dtype=torch.float32).to(DEV).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)


def batches(exs, bs):
    idx = list(range(len(exs))); rng.shuffle(idx)
    return [[exs[i] for i in idx[j:j + bs]] for j in range(0, len(idx), bs)]


cb = [("choice", b) for b in batches(choice_ex, args.bs)]
nb = [("noul", b) for b in batches(noul_ex, args.bs)]
plan = []
ratio = len(nb) / max(1, len(cb)); ci = 0
for i, b in enumerate(nb):
    plan.append(b)
    while ci < len(cb) and ci < (i + 1) / ratio:
        plan.append(cb[ci]); ci += 1
plan += cb[ci:]
def batch_a_budget(exs, bs_max, budget):
    """Batch con lo stesso numero di opzioni e un tetto di token complessivi: con ingressi da 1.024 token e
    fino a 50 opzioni un batch da 32 non entrerebbe in memoria, e accorciarlo è meglio che troncare il testo.
    Giro H: il tetto conta i token DOPO il padding (n_esempi × il più lungo), e dentro blocchi casuali di
    100 batch gli esempi sono ordinati per lunghezza; nel giro G il tetto sommava le lunghezze vere, quindi un
    esempio lungo in mezzo a 31 corti portava tutto il batch a 1.024 token."""
    lun = [min(len(e[0]) // 4 + 8 * (len(e[1]) if isinstance(e[1], (list, tuple)) else 1), args.max_len) for e in exs]
    idx = list(range(len(exs))); rng.shuffle(idx)
    out, blocco = [], 100 * bs_max
    for j in range(0, len(idx), blocco):
        corrente, lmax = [], 0
        for i in sorted(idx[j:j + blocco], key=lambda i: lun[i]):
            if corrente and (len(corrente) >= bs_max or (len(corrente) + 1) * max(lmax, lun[i]) > budget):
                out.append(corrente); corrente, lmax = [], 0
            corrente.append(exs[i]); lmax = max(lmax, lun[i])
        if corrente:
            out.append(corrente)
    return out


if broad_choice:   # Choice di ampiezza: un batch ha sempre lo stesso numero di opzioni
    byk = collections.defaultdict(list)
    for e in broad_choice:
        byk[len(e[1])].append(e)
    bcb = [("bchoice", b) for k in sorted(byk) for b in batch_a_budget(byk[k], args.bs, args.token_budget)]
    rng.shuffle(bcb)
    merged, passo, j = [], len(plan) / max(1, len(bcb)), 0
    for i, p_ in enumerate(plan):
        merged.append(p_)
        while j < len(bcb) and j < (i + 1) / passo:
            merged.append(bcb[j]); j += 1
    plan = merged + bcb[j:]
if broad_noul and args.bnoul_budget > 0:   # sì/no di ampiezza in batch propri, tetto di token come le Choice
    bnb = [("bnoul", b) for b in batch_a_budget(broad_noul, args.bs, args.bnoul_budget)]
    rng.shuffle(bnb)
    merged, passo, j = [], len(plan) / max(1, len(bnb)), 0
    for i, p_ in enumerate(plan):
        merged.append(p_)
        while j < len(bnb) and j < (i + 1) / passo:
            merged.append(bnb[j]); j += 1
    plan = merged + bnb[j:]
if broad_score:    # Score: stessa regola delle Choice, un batch ha sempre la stessa scala
    byk = collections.defaultdict(list)
    for e in broad_score:
        byk[(len(e[1]), tuple(e[2]))].append(e)
    bsb = [("score", b) for k in sorted(byk, key=lambda x: x[0]) for b in
           batch_a_budget(byk[k], args.bs, args.token_budget)]
    rng.shuffle(bsb)
    merged, passo, j = [], len(plan) / max(1, len(bsb)), 0
    for i, p_ in enumerate(plan):
        merged.append(p_)
        while j < len(bsb) and j < (i + 1) / passo:
            merged.append(bsb[j]); j += 1
    plan = merged + bsb[j:]
if args.max_steps > 0:
    plan = plan[:args.max_steps]
total = len(plan)

dev = [json.loads(l) for l in open(os.path.join(D, "dev_noul_ext.jsonl"))]
drng = random.Random(1234); dev = drng.sample(dev, min(args.eval_n, len(dev)))
dev_in = fg.noul_inputs(dev); dev_y = np.array([r["label"] for r in dev])
dev_same = np.array([r["neg_type"] in ("pos", "neg_same") for r in dev])


dev_in_nat = None
if DMT:
    dev_in_nat = [fg.fmt(r["text"], [fg.noul_label(r["question"]) if r["lang"] == "en" else DMT[fg.noul_label(r["question"])][r["lang"]]])
                  for r in dev]


ag_in, ag_y = [], []
for r in agent_dev:
    for k, v in sorted(r["labels"].items()):
        for ql in ("en", "it"):
            ag_in.append(fg.fmt(r["cmd"], [AG_PROPS[k][ql][0]])); ag_y.append(v)
ag_y = np.array(ag_y)


bdev_noul = [ex for tipo, ex in broad_dev if tipo == "noul"]
bdev_choice = collections.defaultdict(list)
bdev_score = collections.defaultdict(list)
for tipo, ex in broad_dev:
    if tipo == "choice":
        bdev_choice[len(ex[1])].append(ex)
    elif tipo == "score":
        bdev_score[(len(ex[1]), tuple(ex[2]))].append(ex)


def eval_broad():
    """Sviluppo di ampiezza: accordo con l'insegnante (AUROC sì/no sull'etichetta dura dell'insegnante, accuratezza
    Choice sulla sua opzione più probabile), su testi tenuti fuori per text_id."""
    out = {}
    if bdev_noul:
        s = fg.score_torch(model, tok, [t for t, _ in bdev_noul], 1, batch_size=64, max_length=args.max_len)[:, 0]
        yb = np.array([p >= 0.5 for _, p in bdev_noul], int)
        if 0 < yb.sum() < len(yb):
            out["dev_broad_noul_auroc"] = float(roc_auc_score(yb, s))
    ok = []
    for k, items in bdev_choice.items():
        z = fg.score_torch(model, tok, [t for t, _ in items], k, batch_size=64, max_length=args.max_len)
        ok += list(z.argmax(1) == np.array([int(np.argmax(p)) for _, p in items]))
    if ok:
        out["dev_broad_choice_acc"] = float(np.mean(ok))
    mae = []
    for (k, valori), items in bdev_score.items():
        z = fg.score_torch(model, tok, [t for t, _, _ in items], k, batch_size=32, max_length=args.max_len)
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        v = np.array(valori, float)
        ampiezza = (v.max() - v.min()) or 1.0
        y = np.array([t_[1] for t_ in items], float)
        mae += list(np.abs(p @ v - y @ v) / ampiezza)
    if mae:
        # stessa metrica di S1MB (errore sul valore atteso normalizzato), girata in «più alto è meglio»
        out["dev_broad_score_mae"] = float(np.mean(mae))
        out["dev_broad_score"] = float(1.0 - np.mean(mae))
    return out


def evaluate():
    model.eval()
    s = fg.score_torch(model, tok, dev_in, 1, batch_size=64, max_length=args.max_len)[:, 0]
    out = dict(dev_auroc_en=float(roc_auc_score(dev_y, s)),
               dev_auroc_same=float(roc_auc_score(dev_y[dev_same], s[dev_same])))
    if dev_in_nat is not None:
        sn = fg.score_torch(model, tok, dev_in_nat, 1, batch_size=64, max_length=args.max_len)[:, 0]
        out["dev_auroc_native"] = float(roc_auc_score(dev_y, sn))
        out["dev_auroc"] = (out["dev_auroc_en"] + out["dev_auroc_native"]) / 2
    else:
        out["dev_auroc"] = out["dev_auroc_en"]
    if ag_in:
        sa = fg.score_torch(model, tok, ag_in, 1, batch_size=64, max_length=args.max_len)[:, 0]
        out["dev_auroc_agent"] = float(roc_auc_score(ag_y, sa))
        parts = [out["dev_auroc_en"], out.get("dev_auroc_native", out["dev_auroc_en"]), out["dev_auroc_agent"]]
        out["dev_auroc"] = float(np.mean(parts))
    if broad_dev:   # la media di selezione include anche l'ampiezza (media di AUROC sì/no e accuratezza Choice)
        out.update(eval_broad())
        bparts = [out[k] for k in ("dev_broad_noul_auroc", "dev_broad_choice_acc", "dev_broad_score") if k in out]
        if bparts:
            base = [out["dev_auroc_en"], out.get("dev_auroc_native", out["dev_auroc_en"])]
            if "dev_auroc_agent" in out:
                base.append(out["dev_auroc_agent"])
            out["dev_broad"] = float(np.mean(bparts))
            out["dev_auroc"] = float(np.mean(base + [out["dev_broad"]]))
    model.train()
    return out


enc_params = [p_ for n, p_ in model.named_parameters() if "encoder_model" in n]
head_params = [p_ for n, p_ in model.named_parameters() if "encoder_model" not in n]
opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
warm = max(1, int(0.06 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))


def save(dirname, extra):
    out = os.path.join(args.out, dirname)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(os.path.join(args.init, f), os.path.join(out, f))
    json.dump(extra, open(os.path.join(out, "train_meta.json"), "w"), indent=1)


meta = dict(args=vars(args), n_noul_examples=len(noul_ex), n_choice_examples=len(choice_ex),
            noul_pos_frac=float(np.mean([e[1] for e in noul_ex])), steps=total,
            n_train_intents=len(INT), pool_sizes={k: len(v) for k, v in pool.items()},
            n_broad_choice=len(broad_choice), n_broad_noul=len(broad_noul), n_broad_score=len(broad_score),
            n_broad_dev=len(broad_dev),
            broad_stats=dict(broad_stats), broad_tenuti=dict(broad_tenuti) if args.broad_requests else {},
            plan_counts=dict(collections.Counter(t for t, _ in plan)))
print(json.dumps({k: v for k, v in meta.items() if k != "args"}), flush=True)
log = []; best = -1.0; t0 = time.time(); run = {"choice": [], "noul": [], "bchoice": [], "bnoul": [], "score": []}
rec = dict(step=0, **evaluate()); log.append(rec); print(json.dumps(rec), flush=True)
model.train()
for step, (task, b) in enumerate(plan):
    texts = [e[0] for e in b]
    if task in ("bchoice", "score"):   # bersagli morbidi, K opzioni (o livelli) uguali in tutto il batch
        K = len(b[0][1]); y = torch.tensor([e[1] for e in b], dtype=torch.float32, device=DEV)
    else:
        K = 8 if task == "choice" else 1
        y = torch.tensor([e[1] for e in b], dtype=torch.float32 if task == "bnoul" else None, device=DEV)
    enc_in = {k_: v_.to(DEV) for k_, v_ in tok(texts, truncation=True, max_length=args.max_len, padding="longest", return_tensors="pt").items()}
    lg = model(input_ids=enc_in["input_ids"], attention_mask=enc_in["attention_mask"], max_num_classes=K).logits[:, :K].float()
    # il teacher Skejja serve solo dove la sua uscita entra nella perdita: prima girava su ogni batch, anche su
    # quelli di ampiezza (kd_broad=0) e sulle Score, dove veniva buttata (stessa perdita, meno calcolo)
    serve_teacher = teacher is not None and (task in ("choice", "noul") or (args.kd_broad and task in ("bchoice", "bnoul")))
    if serve_teacher:
        with torch.no_grad():
            tl = teacher(input_ids=enc_in["input_ids"], attention_mask=enc_in["attention_mask"], max_num_classes=K).logits[:, :K].float()
    if task == "choice":
        loss = F.cross_entropy(lg, y) + 0.5 * F.binary_cross_entropy_with_logits(lg, F.one_hot(y, K).float())
        if serve_teacher:
            loss = loss + args.kd_w * F.kl_div(F.log_softmax(lg, -1), F.softmax(tl, -1), reduction="batchmean")
    elif task == "bchoice":
        loss = F.cross_entropy(lg, y) + 0.5 * F.binary_cross_entropy_with_logits(lg, y)
        if serve_teacher:
            loss = loss + args.kd_w * F.kl_div(F.log_softmax(lg, -1), F.softmax(tl, -1), reduction="batchmean")
    elif task == "bnoul":
        # sì/no di ampiezza in batch propri: la stessa perdita che avevano dentro i batch sì/no (BCE sul p_sì
        # dell'insegnante o sull'etichetta del corpus), distillazione solo con --kd_broad 1
        s = lg[:, 0]
        loss = F.binary_cross_entropy_with_logits(s, y)
        if serve_teacher:
            loss = loss + args.kd_w * F.binary_cross_entropy_with_logits(s, torch.sigmoid(tl[:, 0]))
    elif task == "score":
        # Score = Choice sui livelli, più l'errore sul VALORE ATTESO, che è quello che misura S1MB
        # (normalized_expected_score_mae): sbagliare di un livello non pesa come sbagliare di quattro.
        val = torch.tensor(b[0][2], dtype=torch.float32, device=DEV)
        ampiezza = float(val.max() - val.min()) or 1.0
        p = F.softmax(lg, -1)
        atteso_p, atteso_y = (p * val).sum(-1), (y * val).sum(-1)
        loss = (F.cross_entropy(lg, y)
                + args.lambda_score * ((atteso_p - atteso_y).abs() / ampiezza).mean())
    else:
        s = lg[:, 0]
        loss = F.binary_cross_entropy_with_logits(s, y.float())
        if serve_teacher:
            # niente distillazione dal teacher Skejja sugli esempi sulle azioni (se kd_agent=0) e di ampiezza (se kd_broad=0)
            senza_kd = lambda e: len(e) > 2 and ((e[2] == "broad" and not args.kd_broad) or
                                                 (e[2] != "broad" and bool(e[2]) and not args.kd_agent))
            keep = torch.tensor([0.0 if senza_kd(e) else 1.0 for e in b], device=DEV)
            if keep.sum() == len(b):
                loss = loss + args.kd_w * F.binary_cross_entropy_with_logits(s, torch.sigmoid(tl[:, 0]))
            elif keep.sum() > 0:
                kd = F.binary_cross_entropy_with_logits(s, torch.sigmoid(tl[:, 0]), reduction="none")
                loss = loss + args.kd_w * (kd * keep).sum() / keep.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    run[task].append(float(loss))
    if (step + 1) % 100 == 0 or step + 1 == total:
        el = time.time() - t0
        rec = dict(step=step + 1, total=total, sec=round(el, 1), eta_min=round(el / (step + 1) * (total - step - 1) / 60, 1),
                   loss_choice=float(np.mean(run["choice"][-50:])) if run["choice"] else None,
                   loss_noul=float(np.mean(run["noul"][-50:])) if run["noul"] else None,
                   loss_bchoice=float(np.mean(run["bchoice"][-50:])) if run["bchoice"] else None,
                   loss_bnoul=float(np.mean(run["bnoul"][-50:])) if run["bnoul"] else None,
                   loss_score=float(np.mean(run["score"][-50:])) if run["score"] else None)
        if (step + 1) % args.eval_every == 0 or step + 1 == total:
            rec.update(evaluate())
            if rec["dev_auroc"] > best:
                best = rec["dev_auroc"]; save("best", dict(meta, log=log + [rec], best_step=step + 1))
        log.append(rec); print(json.dumps(rec), flush=True)
model.eval()
meta.update(train_sec=round(time.time() - t0, 1), log=log, best_dev_auroc=best)
save("last", meta)
print("DONE", json.dumps({k: v for k, v in meta.items() if k not in ("log", "args")}))
