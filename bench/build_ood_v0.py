"""Test fuori da MASSIVE, v0 (fase 2, punti 2 e 3): domotica, controlli sulle azioni di un agente, smistamento email.

Testi in italiano, ogni opzione/domanda scritta in italiano E in inglese (per misurare l'effetto della lingua delle
opzioni). Scritto dall'agente (Claude) il 6 ottobre 2026: è un primo test da rivedere e allargare con esempi reali
dell'utente. Solo per valutazione: mai usato in addestramento.

Formato (bench/data/ood/ood_v0.jsonl), una riga per item:
- Choice: {task:"choice", scenario, id, text, options_it, options_en, gold_idx}
- Noul:   {task:"noul", scenario, id, text, label_it, label_en, label (0/1), neg_type}
  (label_* = testo dell'etichetta passato al modello, come la descrizione X in "Does the user want to X?")

uso: python bench/build_ood_v0.py --out bench/data/ood
"""
import argparse, collections, json, os
import numpy as np

SEED = 51

# ---------------- domotica: comandi della casa (id: (italiano, inglese, famiglia))
CMD = {
    "luce_sogg_on": ("accendere la luce del soggiorno", "turn on the living room light", "luce"),
    "luce_sogg_off": ("spegnere la luce del soggiorno", "turn off the living room light", "luce"),
    "luce_cam_on": ("accendere la luce della camera da letto", "turn on the bedroom light", "luce"),
    "luce_cam_off": ("spegnere la luce della camera da letto", "turn off the bedroom light", "luce"),
    "luce_dim": ("abbassare l'intensità delle luci", "dim the lights", "luce"),
    "tapp_su": ("alzare le tapparelle", "raise the shutters", "tapparelle"),
    "tapp_giu": ("abbassare le tapparelle", "lower the shutters", "tapparelle"),
    "term_su": ("alzare la temperatura del riscaldamento", "turn the heating up", "clima"),
    "term_giu": ("abbassare la temperatura del riscaldamento", "turn the heating down", "clima"),
    "clima_on": ("accendere il condizionatore", "turn on the air conditioner", "clima"),
    "lav_start": ("avviare la lavatrice", "start the washing machine", "elettrodomestici"),
    "lav_stato": ("sapere quanto manca alla fine del lavaggio", "check how long the washing machine has left", "elettrodomestici"),
    "robot": ("avviare il robot aspirapolvere", "start the robot vacuum", "elettrodomestici"),
    "cancello_apri": ("aprire il cancello", "open the gate", "sicurezza"),
    "allarme_on": ("attivare l'allarme", "arm the alarm", "sicurezza"),
    "allarme_off": ("disattivare l'allarme", "disarm the alarm", "sicurezza"),
    "porta_stato": ("controllare se la porta d'ingresso è chiusa", "check whether the front door is locked", "sicurezza"),
    "musica_on": ("mettere della musica", "play some music", "media"),
    "tv_off": ("spegnere la televisione", "turn off the TV", "media"),
    "consumi": ("sapere quanta energia sta consumando la casa", "check the home's energy consumption", "energia"),
}
HOME = [
    ("luce_sogg_on", "accendi la luce in salotto"), ("luce_sogg_on", "fai luce in soggiorno che non vedo niente"),
    ("luce_sogg_on", "illumina il soggiorno"), ("luce_sogg_on", "salotto, luci accese per favore"),
    ("luce_sogg_off", "spegni la luce del salotto"), ("luce_sogg_off", "in soggiorno puoi spegnere?"),
    ("luce_sogg_off", "salotto luci off"), ("luce_sogg_off", "ho finito di leggere, spegni pure la lampada vicino al divano"),
    ("luce_cam_on", "accendi la luce in camera"), ("luce_cam_on", "accendimi l'abat-jour in camera da letto"),
    ("luce_cam_on", "luce in camera da letto"), ("luce_cam_on", "mi sono svegliato, accendi la luce della stanza da letto"),
    ("luce_cam_off", "spegni la luce della camera"), ("luce_cam_off", "vado a dormire, spegni in camera da letto"),
    ("luce_cam_off", "buio in camera per favore"), ("luce_cam_off", "la luce della camera da letto è rimasta accesa, spegnila"),
    ("luce_dim", "abbassa un po' le luci"), ("luce_dim", "luci soffuse per la cena"),
    ("luce_dim", "metti le luci al trenta per cento"), ("luce_dim", "questa luce è troppo forte, attenuala"),
    ("tapp_su", "alza le tapparelle"), ("tapp_su", "tira su le persiane della cucina"),
    ("tapp_su", "fai entrare un po' di sole"), ("tapp_su", "apri le serrande"),
    ("tapp_giu", "abbassa le tapparelle"), ("tapp_giu", "chiudi le persiane che entra troppo sole"),
    ("tapp_giu", "giù le serrande in camera"), ("tapp_giu", "abbassa tutto, sta arrivando il temporale"),
    ("term_su", "alza il riscaldamento di due gradi"), ("term_su", "ho freddo"),
    ("term_su", "metti il termostato a ventidue"), ("term_su", "scalda un po' la casa"),
    ("term_giu", "abbassa il riscaldamento"), ("term_giu", "fa troppo caldo con questi termosifoni"),
    ("term_giu", "porta il termostato a diciotto gradi stanotte"), ("term_giu", "riduci la temperatura di un grado"),
    ("clima_on", "accendi l'aria condizionata"), ("clima_on", "si muore di caldo, accendi il climatizzatore"),
    ("clima_on", "metti il condizionatore in camera"), ("clima_on", "rinfresca il soggiorno"),
    ("lav_start", "fai partire la lavatrice"), ("lav_start", "avvia il lavaggio a quaranta gradi"),
    ("lav_start", "lavatrice, programma cotone, vai"), ("lav_start", "ho caricato i panni, puoi avviare la lavatrice?"),
    ("lav_stato", "quanto manca alla lavatrice?"), ("lav_stato", "la lavatrice ha finito?"),
    ("lav_stato", "a che punto è il lavaggio?"), ("lav_stato", "posso già stendere i panni?"),
    ("robot", "fai partire il robot aspirapolvere"), ("robot", "pulisci il pavimento della cucina"),
    ("robot", "manda il roomba in soggiorno"), ("robot", "ci sono briciole ovunque, passa l'aspirapolvere"),
    ("cancello_apri", "apri il cancello"), ("cancello_apri", "sta arrivando il corriere, apri il cancello"),
    ("cancello_apri", "apri il passo carraio"), ("cancello_apri", "fammi entrare con la macchina"),
    ("allarme_on", "attiva l'allarme"), ("allarme_on", "esco, inserisci l'antifurto"),
    ("allarme_on", "allarme perimetrale on"), ("allarme_on", "metti l'allarme per la notte"),
    ("allarme_off", "disattiva l'allarme"), ("allarme_off", "sono a casa, togli l'antifurto"),
    ("allarme_off", "spegni l'allarme che sta suonando"), ("allarme_off", "disinserisci il sistema di sicurezza"),
    ("porta_stato", "ho chiuso la porta di casa?"), ("porta_stato", "la porta d'ingresso è chiusa a chiave?"),
    ("porta_stato", "controlla la serratura della porta principale"), ("porta_stato", "sono uscito di fretta, la porta è chiusa?"),
    ("musica_on", "metti un po' di musica"), ("musica_on", "fammi sentire qualcosa di jazz"),
    ("musica_on", "suona la mia playlist preferita"), ("musica_on", "musica in cucina"),
    ("tv_off", "spegni la tv"), ("tv_off", "la televisione in salotto è ancora accesa, spegnila"),
    ("tv_off", "basta tv per stasera"), ("tv_off", "spegni il televisore"),
    ("consumi", "quanto sta consumando la casa adesso?"), ("consumi", "quanta corrente stiamo usando?"),
    ("consumi", "dimmi i consumi elettrici di oggi"), ("consumi", "quanti kilowatt stiamo assorbendo?"),
]

# ---------------- agente: proprietà sì/no delle azioni
PROPS = [
    ("del", "cancellare o sovrascrivere file", "delete or overwrite files"),
    ("net", "connettersi a internet", "connect to the internet"),
    ("sudo", "usare privilegi di amministratore", "use administrator privileges"),
    ("inst", "installare software", "install software"),
    ("ro", "solo leggere informazioni senza modificare nulla", "only read information without changing anything"),
    ("push", "pubblicare codice o pacchetti su un servizio remoto", "publish code or packages to a remote service"),
]
# (testo dell'azione, del, net, sudo, inst, ro, push)
AGENT = [
    ("rm -rf build/ dist/", 1, 0, 0, 0, 0, 0),
    ("ls -la ~/Documenti", 0, 0, 0, 0, 1, 0),
    ("sudo apt-get install -y nginx", 0, 1, 1, 1, 0, 0),
    ("git push --force origin main", 0, 1, 0, 0, 0, 1),
    ("cat /etc/hosts", 0, 0, 0, 0, 1, 0),
    ("curl -X POST https://api.example.com/upload -F file=@clienti.csv", 0, 1, 0, 0, 0, 0),
    ("pip install requests", 0, 1, 0, 1, 0, 0),
    ("sudo rm /var/log/syslog", 1, 0, 1, 0, 0, 0),
    ("grep -r 'password' src/", 0, 0, 0, 0, 1, 0),
    ("Leggi il file config.yaml e dimmi su che porta ascolta il server", 0, 0, 0, 0, 1, 0),
    ("Invia per email il report trimestrale a tutto il team", 0, 1, 0, 0, 0, 0),
    ("Svuota il cestino", 1, 0, 0, 0, 0, 0),
    ("git commit -am 'fix' && git push", 0, 1, 0, 0, 0, 1),
    ("docker pull postgres:16", 0, 1, 0, 1, 0, 0),
    ("sudo systemctl restart ssh", 0, 0, 1, 0, 0, 0),
    ("cp nuovo.txt vecchio.txt", 1, 0, 0, 0, 0, 0),
    ("df -h", 0, 0, 0, 0, 1, 0),
    ("Carica la cartella delle foto su Google Drive", 0, 1, 0, 0, 0, 0),
    ("npm install -g typescript", 0, 1, 0, 1, 0, 0),
    ("find . -name '*.tmp' -delete", 1, 0, 0, 0, 0, 0),
    ("ping -c 3 8.8.8.8", 0, 1, 0, 0, 1, 0),
    ("brew upgrade", 0, 1, 0, 1, 0, 0),
    ("Apri il file bilancio.xlsx e calcola il totale della colonna B senza salvare", 0, 0, 0, 0, 1, 0),
    ("dd if=/dev/zero of=/dev/sdb bs=1M", 1, 0, 0, 0, 0, 0),
    ("sudo chmod -R 777 /var/www", 0, 0, 1, 0, 0, 0),
    ("scp backup.tar.gz utente@server.example.com:/backup/", 0, 1, 0, 0, 0, 0),
    ("git log --oneline | head", 0, 0, 0, 0, 1, 0),
    ("Pubblica la nuova versione del pacchetto su PyPI", 0, 1, 0, 0, 0, 1),
    ("Elimina tutti i file nella cartella Download", 1, 0, 0, 0, 0, 0),
    ("Scarica l'ultima versione di Firefox e installala", 0, 1, 0, 1, 0, 0),
    ("ps aux | grep python", 0, 0, 0, 0, 1, 0),
    ("sudo reboot", 0, 0, 1, 0, 0, 0),
    ("Rinomina tutte le foto con la data di scatto", 0, 0, 0, 0, 0, 0),
    ("git reset --hard HEAD~3", 1, 0, 0, 0, 0, 0),
    ("Controlla quanti utenti si sono registrati oggi nel database, senza modificarlo", 0, 0, 0, 0, 1, 0),
    ("sudo pip install --upgrade numpy", 0, 1, 1, 1, 0, 0),
]

# ---------------- email: categorie e proprietà
CAT = {
    "fattura": ("una fattura o un pagamento", "an invoice or a payment"),
    "riunione": ("una richiesta di riunione o appuntamento", "a meeting or appointment request"),
    "promo": ("pubblicità o newsletter", "an advertisement or newsletter"),
    "assistenza": ("una richiesta di assistenza tecnica", "a technical support request"),
    "spedizione": ("lo stato di una spedizione o di un ordine", "the status of a shipment or order"),
    "candidatura": ("una candidatura di lavoro", "a job application"),
    "phishing": ("un tentativo di truffa o phishing", "a scam or phishing attempt"),
    "personale": ("un messaggio personale di un amico o familiare", "a personal message from a friend or family member"),
}
EPROPS = [
    ("reply", "richiedere una risposta", "require a reply"),
    ("urgent", "essere urgente", "be urgent"),
    ("deadline", "contenere una scadenza", "contain a deadline"),
]
# (categoria, testo, reply, urgent, deadline)
EMAIL = [
    ("fattura", "In allegato la fattura n. 245 di settembre, importo 1.200 euro, da saldare entro il 31 ottobre.", 0, 0, 1),
    ("fattura", "Le ricordiamo che il pagamento della fattura 118 risulta scaduto da 15 giorni. La preghiamo di provvedere al più presto.", 1, 1, 0),
    ("fattura", "Ricevuta del pagamento: abbiamo ricevuto il suo bonifico di 89,90 euro. Grazie.", 0, 0, 0),
    ("fattura", "Buongiorno, potete mandarmi di nuovo la fattura di agosto? Non la trovo.", 1, 0, 0),
    ("fattura", "Il suo abbonamento verrà rinnovato il 5 novembre con un addebito di 9,99 euro sulla carta registrata.", 0, 0, 1),
    ("riunione", "Ciao, riusciamo a vederci giovedì alle 15 per parlare del progetto? Fammi sapere.", 1, 0, 0),
    ("riunione", "Invito: revisione trimestrale, lunedì 14 alle 10:00, sala B. Conferma la tua presenza.", 1, 0, 0),
    ("riunione", "Serve una call oggi entro le 12 per decidere sul fornitore, è importante.", 1, 1, 1),
    ("riunione", "Ti propongo tre date per l'incontro con il cliente: 3, 4 o 7 novembre. Quale preferisci?", 1, 0, 0),
    ("riunione", "Promemoria: domani visita dal dentista alle 9:30.", 0, 0, 0),
    ("promo", "Solo questo weekend: sconto del 30% su tutta la collezione autunno. Scopri le offerte!", 0, 0, 1),
    ("promo", "La newsletter di ottobre: le novità del mese, gli eventi in città e le ricette di stagione.", 0, 0, 0),
    ("promo", "Black Friday in anticipo: spedizione gratuita su tutti gli ordini fino a mezzanotte.", 0, 0, 1),
    ("promo", "Ti sei perso il nostro webinar? Ecco la registrazione e le slide.", 0, 0, 0),
    ("promo", "Nuovi arrivi in negozio: vieni a trovarci, ti aspettiamo con un omaggio.", 0, 0, 0),
    ("assistenza", "Da stamattina il gestionale non si apre e mi dà errore 500. Potete aiutarmi? Siamo fermi.", 1, 1, 0),
    ("assistenza", "Non riesco a collegare la stampante al nuovo portatile, come devo fare?", 1, 0, 0),
    ("assistenza", "La VPN si disconnette ogni dieci minuti, è un problema noto?", 1, 0, 0),
    ("assistenza", "Ho dimenticato la password del portale clienti e il link di recupero non arriva.", 1, 0, 0),
    ("assistenza", "Il sito dell'e-commerce è giù e stiamo perdendo ordini, intervenite subito per favore.", 1, 1, 0),
    ("spedizione", "Il tuo ordine 5582 è stato spedito e arriverà tra 2 e 3 giorni lavorativi.", 0, 0, 0),
    ("spedizione", "Il corriere non ha trovato nessuno: il pacco è in giacenza fino al 20 ottobre, poi tornerà al mittente.", 1, 0, 1),
    ("spedizione", "Ho ordinato due settimane fa e non mi è ancora arrivato niente. Dov'è il mio pacco?", 1, 0, 0),
    ("spedizione", "Il tuo reso è stato ricevuto: il rimborso sarà accreditato entro 5 giorni.", 0, 0, 0),
    ("spedizione", "L'ordine 7781 è in ritardo per un problema del fornitore, ci scusiamo per il disagio.", 0, 0, 0),
    ("candidatura", "Gentile responsabile, invio il mio curriculum per la posizione di sviluppatore backend.", 1, 0, 0),
    ("candidatura", "Buongiorno, ho visto il vostro annuncio per un magazziniere e vorrei candidarmi. Allego CV.", 1, 0, 0),
    ("candidatura", "Mi chiamo Giulia, sono una designer con cinque anni di esperienza e vorrei far parte del vostro team.", 1, 0, 0),
    ("candidatura", "Allego lettera di presentazione e portfolio per lo stage in marketing con scadenza il 30 ottobre.", 1, 0, 1),
    ("candidatura", "Vorrei sapere se avete ricevuto la mia candidatura inviata la settimana scorsa.", 1, 0, 0),
    ("phishing", "Il suo conto è stato sospeso. Clicchi qui entro 24 ore per riattivarlo inserendo i dati della carta.", 0, 1, 1),
    ("phishing", "Congratulazioni! Ha vinto un iPhone. Per ritirarlo paghi solo 2 euro di spese di spedizione.", 0, 0, 0),
    ("phishing", "Avviso di sicurezza: abbiamo rilevato un accesso sospetto. Verifichi la sua identità a questo link.", 0, 1, 0),
    ("phishing", "Pacco bloccato in dogana: paghi 1,99 euro con questo link per lo sblocco.", 0, 1, 0),
    ("phishing", "Sono il direttore, mi serve subito un bonifico riservato a questo IBAN, non dirlo a nessuno.", 1, 1, 0),
    ("personale", "Ciao! Sabato facciamo una grigliata da noi, vieni? Porta pure qualcuno.", 1, 0, 0),
    ("personale", "Mamma qui: ricordati di chiamare la nonna per il suo compleanno domani.", 0, 0, 1),
    ("personale", "Ti mando le foto del matrimonio di Luca, guarda che belle!", 0, 0, 0),
    ("personale", "Sono arrivato a Lisbona, tutto bene. Ci sentiamo stasera.", 0, 0, 0),
    ("personale", "Mi presti la tua tenda per il campeggio del prossimo weekend? Fammi sapere entro giovedì.", 1, 0, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/data/ood")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED); items = []
    fam = collections.defaultdict(list)
    for k, v in CMD.items():
        fam[v[2]].append(k)
    keys = sorted(CMD)
    for i, (g, t) in enumerate(HOME):
        same = [k for k in fam[CMD[g][2]] if k != g]
        hard = list(rng.choice(same, size=min(3, len(same)), replace=False)) if same else []
        other = [k for k in keys if CMD[k][2] != CMD[g][2]]
        opts = [g] + hard + list(rng.choice(other, size=8 - 1 - len(hard), replace=False)); rng.shuffle(opts)
        items.append(dict(task="choice", scenario="domotica", id=f"home-{i}", text=t, options_it=[CMD[o][0] for o in opts],
                          options_en=[CMD[o][1] for o in opts], gold_idx=int(opts.index(g)), gold=g))
        items.append(dict(task="noul", scenario="domotica", id=f"home-{i}", text=t, label_it=CMD[g][0], label_en=CMD[g][1],
                          label=1, neg_type="pos", q=g))
        if same:
            n = str(rng.choice(same))
            items.append(dict(task="noul", scenario="domotica", id=f"home-{i}", text=t, label_it=CMD[n][0], label_en=CMD[n][1],
                              label=0, neg_type="neg_same", q=n))
        n = str(rng.choice(other))
        items.append(dict(task="noul", scenario="domotica", id=f"home-{i}", text=t, label_it=CMD[n][0], label_en=CMD[n][1],
                          label=0, neg_type="neg_other", q=n))
    for i, row in enumerate(AGENT):
        for (pk, pit, pen), lab in zip(PROPS, row[1:]):
            items.append(dict(task="noul", scenario="agente", id=f"agent-{i}", text=row[0], label_it=pit, label_en=pen,
                              label=int(lab), neg_type="prop", q=pk))
    cats = sorted(CAT)
    for i, (c, t, *props) in enumerate(EMAIL):
        opts = list(rng.permutation(cats))
        items.append(dict(task="choice", scenario="email", id=f"email-{i}", text=t, options_it=[CAT[o][0] for o in opts],
                          options_en=[CAT[o][1] for o in opts], gold_idx=int(opts.index(c)), gold=c))
        for (pk, pit, pen), lab in zip(EPROPS, props):
            items.append(dict(task="noul", scenario="email", id=f"email-{i}", text=t, label_it=pit, label_en=pen,
                              label=int(lab), neg_type="prop", q=pk))
        n = str(rng.choice([x for x in cats if x != c]))
        items.append(dict(task="noul", scenario="email", id=f"email-{i}", text=t, label_it=CAT[c][0], label_en=CAT[c][1],
                          label=1, neg_type="pos", q=c))
        items.append(dict(task="noul", scenario="email", id=f"email-{i}", text=t, label_it=CAT[n][0], label_en=CAT[n][1],
                          label=0, neg_type="neg_cat", q=n))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ood_v0.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = collections.Counter((it["scenario"], it["task"]) for it in items)
    print(len(items), dict(c))


if __name__ == "__main__":
    main()
