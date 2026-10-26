"""Richieste del giro G nel formato «istruzioni + stato + criteri», dal pool di `bench/build_giroG.py`.

Il formato è quello che l'adattatore usa su S1MB (`instructions-then-state-in-text, criteria-as-labels-v1`):
l'istruzione va in testa al testo, lo stato dopo, i criteri diventano le opzioni. Le richieste hanno gli stessi
campi del giro F (`rid`, `text_id`, `text`, `tipo`, `opzioni`, `opzioni_desc`, `affermazione_desc`, `atteso`,
`gold`, `gold_da`, `uso`, `origine`), più tre cose nuove:

- `tipo = "score"`, con `opzioni_desc` = descrizioni dei livelli in ordine e `valori` = valore numerico di
  ciascun livello (la primitiva Score è una Choice sui livelli, §1.3 di docs/fase2_giro_G.md);
- `istruzioni` e `stato` separati, oltre al `text` già reso, per poter rifare la resa senza rigenerare;
- Choice con molte opzioni (fino a `--max_opzioni`, predefinito 50).

Dove l'etichetta esiste già nel corpus (`gold_da = "corpus"`) l'insegnante non serve: pertinenza di mfaq,
GermanQuAD, PIAF e MIRACL, classi di Circa, giudizi di HelpSteer3 e UltraFeedback, intenzioni delle fonti di
casa. L'insegnante serve per la pertinenza graduata 0-4 e per le famiglie del giro F applicate ai testi lunghi.

uso:
  python bench/build_giroG_richieste.py --pool bench/data/broad/pool_giroG.jsonl \\
      --out bench/data/broad --name giroG [--max_token 1024] [--max_opzioni 50]
"""

import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_broad import LANGS  # noqa: E402

# ~4 caratteri per token: taglio di sicurezza prima del tokenizer vero, che vive nel trainer.
CHAR_PER_TOKEN = 4

ISTRUZIONI = {
    "pertinenza": {
        "en": "Decide whether the document below answers the question.",
        "it": "Decidi se il documento qui sotto risponde alla domanda.",
        "es": "Decide si el documento de abajo responde a la pregunta.",
        "de": "Entscheide, ob das Dokument unten die Frage beantwortet.",
        "fr": "Décide si le document ci-dessous répond à la question.",
    },
    "pertinenza_grado": {
        "en": "Rate how well the document below answers the question.",
        "it": "Valuta quanto bene il documento qui sotto risponde alla domanda.",
        "es": "Valora en qué medida el documento de abajo responde a la pregunta.",
        "de": "Bewerte, wie gut das Dokument unten die Frage beantwortet.",
        "fr": "Évalue dans quelle mesure le document ci-dessous répond à la question.",
    },
    "relazione": {
        "en": "Read the question and the reply, then decide what the reply means.",
        "it": "Leggi la domanda e la risposta, poi decidi che cosa significa la risposta.",
        "es": "Lee la pregunta y la respuesta, luego decide qué significa la respuesta.",
        "de": "Lies die Frage und die Antwort und entscheide, was die Antwort bedeutet.",
        "fr": "Lis la question et la réponse, puis décide ce que signifie la réponse.",
    },
    "giudizio": {
        "en": "Judge the quality of the answer given to the request below.",
        "it": "Giudica la qualità della risposta data alla richiesta qui sotto.",
        "es": "Juzga la calidad de la respuesta dada a la petición de abajo.",
        "de": "Beurteile die Qualität der Antwort auf die Anfrage unten.",
        "fr": "Juge la qualité de la réponse donnée à la demande ci-dessous.",
    },
    "intento": {
        "en": "Pick the intent that matches what the user is asking for.",
        "it": "Scegli l'intenzione che corrisponde a quello che l'utente chiede.",
        "es": "Elige la intención que corresponde a lo que pide el usuario.",
        "de": "Wähle die Absicht, die dem entspricht, was der Nutzer möchte.",
        "fr": "Choisis l'intention qui correspond à ce que demande l'utilisateur.",
    },
}

ETICHETTE_SI_NO = {
    "en": ("The document answers the question.", "yes/no"),
    "it": ("Il documento risponde alla domanda.", "sì/no"),
    "es": ("El documento responde a la pregunta.", "sí/no"),
    "de": ("Das Dokument beantwortet die Frage.", "ja/nein"),
    "fr": ("Le document répond à la question.", "oui/non"),
}

# Scala della pertinenza graduata: i cinque livelli 0-4, la scala prevalente di S1MB.
LIVELLI_PERTINENZA = {
    "en": ["the document is about something else",
           "the document is on topic but does not address the question",
           "the document touches the question without answering it",
           "the document answers the question in part",
           "the document answers the question fully"],
    "it": ["il documento parla d'altro",
           "il documento è in tema ma non affronta la domanda",
           "il documento sfiora la domanda senza rispondere",
           "il documento risponde in parte alla domanda",
           "il documento risponde pienamente alla domanda"],
    "es": ["el documento trata de otra cosa",
           "el documento está en tema pero no aborda la pregunta",
           "el documento roza la pregunta sin responderla",
           "el documento responde en parte a la pregunta",
           "el documento responde plenamente a la pregunta"],
    "de": ["das Dokument handelt von etwas anderem",
           "das Dokument ist zum Thema, behandelt die Frage aber nicht",
           "das Dokument streift die Frage, ohne sie zu beantworten",
           "das Dokument beantwortet die Frage teilweise",
           "das Dokument beantwortet die Frage vollständig"],
    "fr": ["le document parle d'autre chose",
           "le document est dans le sujet mais n'aborde pas la question",
           "le document effleure la question sans y répondre",
           "le document répond en partie à la question",
           "le document répond pleinement à la question"],
}

LIVELLI_GIUDIZIO = {
    "en": ["unusable: wrong or off-topic", "poor: partly wrong or unhelpful", "acceptable: correct but thin",
           "good: correct and useful", "excellent: complete, correct and clear"],
    "it": ["inutilizzabile: sbagliata o fuori tema", "scarsa: in parte sbagliata o poco utile",
           "accettabile: corretta ma essenziale", "buona: corretta e utile",
           "ottima: completa, corretta e chiara"],
    "es": ["inservible: incorrecta o fuera de tema", "pobre: en parte incorrecta o poco útil",
           "aceptable: correcta pero escasa", "buena: correcta y útil",
           "excelente: completa, correcta y clara"],
    "de": ["unbrauchbar: falsch oder am Thema vorbei", "schwach: teils falsch oder wenig hilfreich",
           "annehmbar: richtig, aber dünn", "gut: richtig und hilfreich",
           "ausgezeichnet: vollständig, richtig und klar"],
    "fr": ["inutilisable : fausse ou hors sujet", "faible : en partie fausse ou peu utile",
           "acceptable : correcte mais sommaire", "bonne : correcte et utile",
           "excellente : complète, correcte et claire"],
}


def taglia(testo, max_char):
    testo = (testo or "").strip()
    return testo if len(testo) <= max_char else testo[:max_char].rsplit(" ", 1)[0] + " […]"


def rendi(istruzione, pezzi, max_char):
    """Resa «istruzioni + stato»: l'istruzione in testa, poi lo stato, come fa l'adattatore S1MB."""
    corpo = "\n".join(f"{k}: {v}" for k, v in pezzi if v)
    return f"{istruzione}\n\n{taglia(corpo, max_char - len(istruzione) - 2)}"


def etichette_intenzioni(data_dir):
    """Descrizioni delle intenzioni delle fonti di casa, meno i 4 ambiti esclusi di MASSIVE."""
    sys.path.insert(0, os.path.join(HERE))
    import importlib
    fm = importlib.import_module("skejja_measure")
    desc = json.load(open(os.path.join(data_dir, "intent_descriptions.json")))
    return {i: d for i, d in desc.items() if not fm.is_unseen(i)}


def richieste_pertinenza(riga, lang, rng, max_char, teacher_grado):
    """Dalla stessa coppia: una sì/no con etichetta del corpus e, a campione, una Score graduata 0-4."""
    out = []
    dom, pos, negs = riga["text"], riga["positivo"], riga.get("negativi") or []
    aff, _ = ETICHETTE_SI_NO[lang]
    istr = ISTRUZIONI["pertinenza"][lang]
    for k, (doc, gold) in enumerate([(pos, 1.0)] + [(n, 0.0) for n in negs[:2]]):
        out.append(dict(rid=f"{riga['id']}-pert-{k}", text_id=riga["id"], lang=lang, tipo="noul",
                        famiglia="pertinenza", origine=riga["source"], istruzioni=istr,
                        stato={"question": dom, "document": doc},
                        text=rendi(istr, [("question", dom), ("document", doc)], max_char),
                        affermazione=aff, affermazione_desc=aff, gold=gold, gold_da="corpus",
                        negativo_da=riga.get("negativi_da") if gold == 0.0 else None))
    if teacher_grado and negs:
        istr_g = ISTRUZIONI["pertinenza_grado"][lang]
        doc = rng.choice([pos] + negs)
        liv = LIVELLI_PERTINENZA[lang]
        out.append(dict(rid=f"{riga['id']}-pert-grado", text_id=riga["id"], lang=lang, tipo="score",
                        famiglia="pertinenza_grado", origine=riga["source"], istruzioni=istr_g,
                        stato={"question": dom, "document": doc},
                        text=rendi(istr_g, [("question", dom), ("document", doc)], max_char),
                        opzioni=[str(i) for i in range(5)], opzioni_desc=liv, valori=[0, 1, 2, 3, 4],
                        gold=None, gold_da="insegnante",
                        atteso="4" if doc == pos else None))
    return out


def richieste_relazione(riga, max_char):
    """Circa: la classe vera del corpus diventa sia una Choice sia una sì/no."""
    classi = {
        "Yes": "the reply means yes",
        "No": "the reply means no",
        "In the middle, neither yes nor no": "the reply is in the middle, neither yes nor no",
        "Yes, subject to some conditions": "the reply means yes, but only under some conditions",
        "Other": "the reply does not answer the question",
    }
    gold = riga.get("gold")
    if gold not in classi:
        return []
    istr = ISTRUZIONI["relazione"]["en"]
    pezzi = [("context", riga.get("contesto")), ("question", riga["text"]), ("reply", riga["secondo_testo"])]
    testo = rendi(istr, pezzi, max_char)
    chiavi = list(classi)
    out = [dict(rid=f"{riga['id']}-rel-choice", text_id=riga["id"], lang="en", tipo="choice",
                famiglia="relazione_risposta", origine=riga["source"], istruzioni=istr,
                stato={k: v for k, v in pezzi if v}, text=testo, opzioni=chiavi,
                opzioni_desc=[classi[k] for k in chiavi], gold=gold, gold_da="corpus")]
    for k in (gold, *[c for c in chiavi if c != gold][:1]):
        out.append(dict(rid=f"{riga['id']}-rel-noul-{chiavi.index(k)}", text_id=riga["id"], lang="en",
                        tipo="noul", famiglia="relazione_risposta", origine=riga["source"], istruzioni=istr,
                        stato={kk: v for kk, v in pezzi if v}, text=testo,
                        affermazione=classi[k], affermazione_desc=classi[k],
                        gold=1.0 if k == gold else 0.0, gold_da="corpus"))
    return out


LIVELLI_PREFERENZA = {
    "en": ["answer A is much better", "answer A is better", "answer A is slightly better",
           "the two answers are equally good", "answer B is slightly better", "answer B is better",
           "answer B is much better"],
    "it": ["la risposta A è molto migliore", "la risposta A è migliore", "la risposta A è un po' migliore",
           "le due risposte valgono uguale", "la risposta B è un po' migliore", "la risposta B è migliore",
           "la risposta B è molto migliore"],
    "es": ["la respuesta A es mucho mejor", "la respuesta A es mejor", "la respuesta A es algo mejor",
           "las dos respuestas valen igual", "la respuesta B es algo mejor", "la respuesta B es mejor",
           "la respuesta B es mucho mejor"],
    "de": ["Antwort A ist viel besser", "Antwort A ist besser", "Antwort A ist etwas besser",
           "beide Antworten sind gleich gut", "Antwort B ist etwas besser", "Antwort B ist besser",
           "Antwort B ist viel besser"],
    "fr": ["la réponse A est bien meilleure", "la réponse A est meilleure", "la réponse A est un peu meilleure",
           "les deux réponses se valent", "la réponse B est un peu meilleure", "la réponse B est meilleure",
           "la réponse B est bien meilleure"],
}
ISTRUZIONI["preferenza"] = {
    "en": "Compare the two answers to the request below and say which one is better, and by how much.",
    "it": "Confronta le due risposte alla richiesta qui sotto e di' quale è migliore, e di quanto.",
    "es": "Compara las dos respuestas a la petición de abajo y di cuál es mejor, y por cuánto.",
    "de": "Vergleiche die beiden Antworten auf die Anfrage unten und sage, welche besser ist und wie deutlich.",
    "fr": "Compare les deux réponses à la demande ci-dessous et dis laquelle est meilleure, et de combien.",
}


def richieste_giudizio(riga, lang, max_char):
    """Giudizi ordinali con l'etichetta del corpus: preferenza fra due risposte (HelpSteer3, 7 livelli
    da -3 a +3) e qualità di una risposta sola (UltraFeedback, 1-10 riportato sui cinque livelli 0-4)."""
    voto, scala = riga.get("gold"), riga.get("scala", "")
    if voto is None:
        return []
    if "-3..3" in scala:
        livelli = LIVELLI_PREFERENZA[lang]
        valori = [-3, -2, -1, 0, 1, 2, 3]
        if int(voto) not in valori:
            return []
        istr = ISTRUZIONI["preferenza"][lang]
        pezzi = [("request", riga["text"]), ("answer A", riga.get("risposta_a")),
                 ("answer B", riga.get("risposta_b"))]
        gold = str(int(voto))
        opzioni = [str(v) for v in valori]
    elif "1-10" in scala:
        livelli = LIVELLI_GIUDIZIO[lang]
        valori = [0, 1, 2, 3, 4]
        istr = ISTRUZIONI["giudizio"][lang]
        pezzi = [("request", riga["text"]), ("answer", riga.get("secondo_testo"))]
        gold = str(min(4, max(0, int(round((float(voto) - 1) / 9 * 4)))))
        opzioni = [str(v) for v in valori]
    else:
        return []
    if not all(v for _, v in pezzi):
        return []
    return [dict(rid=f"{riga['id']}-giudizio", text_id=riga["id"], lang=lang, tipo="score",
                 famiglia="preferenza" if "-3..3" in scala else "giudizio", origine=riga["source"],
                 istruzioni=istr, stato={k: v for k, v in pezzi}, text=rendi(istr, pezzi, max_char),
                 opzioni=opzioni, opzioni_desc=livelli, valori=valori, gold=gold, gold_da="corpus",
                 nota_scala=f"voto originale {voto} ({scala})")]


def richieste_lungo(riga, famiglie, rng, max_char, n_famiglie=2):
    """Famiglie del giro F (argomento, sentiment, registro...) applicate ai testi lunghi, nel formato nuovo:
    istruzione in testa, testo come stato, opzioni come criteri. Le etichette le dà l'insegnante."""
    lang = riga["lang"]
    scelte = rng.sample(sorted(famiglie), min(n_famiglie, len(famiglie)))
    out = []
    for nome in scelte:
        fam = famiglie[nome]
        domanda = fam["domanda"].get(lang) or fam["domanda"]["en"]
        chiavi = sorted(fam.get("opzioni") or {})
        if fam.get("tipo") in ("choice", "entrambi") and chiavi:
            desc = [fam["opzioni"][k].get(lang) or fam["opzioni"][k]["en"] for k in chiavi]
            out.append(dict(rid=f"{riga['id']}-{nome}-choice", text_id=riga["id"], lang=lang, tipo="choice",
                            famiglia=nome, origine=f"lungo/{riga['source']}", istruzioni=domanda,
                            stato={"document": riga["text"]},
                            text=rendi(domanda, [("document", riga["text"])], max_char),
                            opzioni=chiavi, opzioni_desc=desc, gold=None, gold_da="insegnante"))
        aff = fam.get("affermazioni") or {}
        if aff:
            k = rng.choice(sorted(aff))
            testo_aff = aff[k].get(lang) or aff[k]["en"]
            out.append(dict(rid=f"{riga['id']}-{nome}-{k}", text_id=riga["id"], lang=lang, tipo="noul",
                            famiglia=nome, origine=f"lungo/{riga['source']}", istruzioni=domanda,
                            stato={"document": riga["text"]},
                            text=rendi(domanda, [("document", riga["text"])], max_char),
                            affermazione=k, affermazione_desc=testo_aff, gold=None, gold_da="insegnante"))
    return out


def richieste_molte_opzioni(data_dir, rng, max_opzioni, n, max_char):
    """Choice con 11-50 opzioni sulle intenzioni delle fonti di casa: etichetta vera, niente insegnante."""
    desc = etichette_intenzioni(data_dir)
    frasi = []
    path = os.path.join(data_dir, "train_utterances.jsonl")
    for l in open(path):
        r = json.loads(l)
        if r.get("intent") in desc and r.get("text"):
            frasi.append(r)
    rng.shuffle(frasi)
    intenzioni = sorted(desc)
    out = []
    for r in frasi[:n]:
        k = rng.randint(11, max_opzioni)
        altre = [i for i in intenzioni if i != r["intent"]]
        scelte = rng.sample(altre, min(k - 1, len(altre))) + [r["intent"]]
        rng.shuffle(scelte)
        istr = ISTRUZIONI["intento"].get(r.get("lang", "en"), ISTRUZIONI["intento"]["en"])
        out.append(dict(rid=f"intenti-{len(out)}", text_id=f"intenti-{r.get('id', len(out))}",
                        lang=r.get("lang", "en"), tipo="choice", famiglia="intento_molte_opzioni",
                        origine="casa", istruzioni=istr, stato={"message": r["text"]},
                        text=rendi(istr, [("message", r["text"])], max_char),
                        opzioni=scelte, opzioni_desc=[desc[i] for i in scelte],
                        gold=r["intent"], gold_da="corpus", n_opzioni=len(scelte)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "broad"))
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--name", default="giroG")
    ap.add_argument("--max_token", type=int, default=1024, help="budget di lunghezza del giro G")
    ap.add_argument("--max_opzioni", type=int, default=50)
    ap.add_argument("--n_molte_opzioni", type=int, default=12000)
    ap.add_argument("--n_famiglie_lungo", type=int, default=2, help="famiglie del giro F per ogni testo lungo")
    ap.add_argument("--p_grado", type=float, default=0.35, help="quota di coppie con anche la Score graduata")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    max_char = a.max_token * CHAR_PER_TOKEN
    famiglie = json.load(open(os.path.join(a.out, "families.json")))["famiglie"]
    righe = [json.loads(l) for l in open(a.pool)]
    fuori = []
    for r in righe:
        lang = r["lang"] if r["lang"] in LANGS else "en"
        if r["gruppo"] == "pertinenza":
            fuori += richieste_pertinenza(r, lang, rng, max_char, rng.random() < a.p_grado)
        elif r["gruppo"] == "relazione":
            fuori += richieste_relazione(r, max_char)
        elif r["gruppo"] == "score":
            fuori += richieste_giudizio(r, lang, max_char)
        elif r["gruppo"] == "lungo":
            fuori += richieste_lungo(r, famiglie, rng, max_char, a.n_famiglie_lungo)
    fuori += richieste_molte_opzioni(a.data, rng, a.max_opzioni, a.n_molte_opzioni, max_char)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"richieste_{a.name}.jsonl")
    with open(path, "w") as f:
        for r in fuori:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lung = sorted(len(r["text"]) // CHAR_PER_TOKEN for r in fuori)
    riepilogo = dict(
        file=os.path.relpath(path, os.path.dirname(HERE)), n=len(fuori),
        per_tipo=dict(collections.Counter(r["tipo"] for r in fuori)),
        per_famiglia=dict(collections.Counter(r["famiglia"] for r in fuori)),
        per_lingua=dict(collections.Counter(r["lang"] for r in fuori)),
        con_etichetta_del_corpus=sum(1 for r in fuori if r["gold_da"] == "corpus"),
        da_etichettare_con_insegnante=sum(1 for r in fuori if r["gold_da"] == "insegnante"),
        opzioni=dict(massimo=max((r.get("n_opzioni") or len(r.get("opzioni") or [1])) for r in fuori),
                     con_oltre_10=sum(1 for r in fuori if len(r.get("opzioni") or []) > 10)),
        token_stimati=dict(mediana=lung[len(lung) // 2], p90=lung[int(len(lung) * 0.9)], massimo=lung[-1],
                           nota=f"stima a {CHAR_PER_TOKEN} caratteri per token, il taglio vero è nel trainer"),
        budget=dict(max_token=a.max_token, max_opzioni=a.max_opzioni), seed=a.seed)
    json.dump(riepilogo, open(os.path.join(a.out, f"richieste_{a.name}_riepilogo.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(riepilogo, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
