"""Label-conditioned fine-tuning of a (trimmed) GLiClass uni-encoder for Skejja Choice + Noul. CPU only.
Training data: MASSIVE train utterances of SEEN intents only (skejja_measure.HELD_OUT_SCENARIOS excluded, and their
intents are never used as options). Options/questions use intent_descriptions.json (English) for every language.
Losses: Choice = CE over the K=8 option logits + 0.5*BCE(one-hot) [+ kd_w * KL(teacher||student)];
        Noul   = BCE on the single label logit [+ kd_w * BCE(student, sigmoid(teacher))].
"""
import argparse, json, math, os, random, shutil, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--desc", required=True)
ap.add_argument("--keep_layers", default="")
ap.add_argument("--teacher", default="")
ap.add_argument("--kd_w", type=float, default=0.0)
ap.add_argument("--gold_w", type=float, default=1.0)
ap.add_argument("--n_choice", type=int, default=20000)
ap.add_argument("--n_noul", type=int, default=20000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--head_lr", type=float, default=1e-4)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--wd", default="")
args = ap.parse_args()

sys.path.insert(0, args.wd or os.path.dirname(os.path.abspath(__file__)))
import skejja_measure as fm
import skejja_glic as fg
from gliclass import GLiClassModel
from transformers import AutoTokenizer

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
rng = random.Random(args.seed)

desc = json.load(open(args.desc))
seen = sorted(i for i in desc if not fm.is_unseen(i))
by_scen = {}
for i in seen:
    by_scen.setdefault(fm.scenario(i), []).append(i)
utts = [json.loads(l) for l in open(args.train)]
n_all = len(utts)
utts = [u for u in utts if not fm.is_unseen(u["intent"])]
assert all(u["intent"] in seen for u in utts)


def make_choice(u):
    g = u["intent"]
    same = [i for i in by_scen[fm.scenario(g)] if i != g]
    opts = rng.sample(same, min(3, len(same)))
    rest = [i for i in seen if i != g and i not in opts]
    opts += rng.sample(rest, 8 - 1 - len(opts))
    opts.append(g); rng.shuffle(opts)
    return fg.fmt(u["text"], [desc[o] for o in opts]), opts.index(g), opts


def make_noul(u):
    g = u["intent"]
    if rng.random() < 0.5:
        return fg.fmt(u["text"], [desc[g]]), 1, g
    same = [i for i in by_scen[fm.scenario(g)] if i != g]
    if same and rng.random() < 0.5:
        neg = rng.choice(same)
    else:
        neg = rng.choice([i for i in seen if i != g])
    return fg.fmt(u["text"], [desc[neg]]), 0, neg


ch_idx = rng.sample(range(len(utts)), min(args.n_choice, len(utts)))
no_idx = rng.sample(range(len(utts)), min(args.n_noul, len(utts)))
choice_ex = [make_choice(utts[i]) for i in ch_idx]
noul_ex = [make_noul(utts[i]) for i in no_idx]
used_opts = set(o for ex in choice_ex for o in ex[2]) | set(ex[2] for ex in noul_ex)
assert not any(fm.is_unseen(o) for o in used_opts), "unseen intent leaked into training options"

tok = AutoTokenizer.from_pretrained(args.init)
model = GLiClassModel.from_pretrained(args.init, dtype=torch.float32)
if args.keep_layers:
    keep = [int(x) for x in args.keep_layers.split(",")]
    assert keep[0] == 0
    enc = model.model.encoder_model; types = list(enc.config.layer_types)
    enc.layers = torch.nn.ModuleList([enc.layers[j] for j in keep])
    for i, l in enumerate(enc.layers):
        l.layer_idx = i; l.attn.layer_idx = i
    for c in (enc.config, model.config.encoder_config):
        c.num_hidden_layers = len(keep); c.layer_types = [types[j] for j in keep]
teacher = None
if args.teacher and args.kd_w > 0:
    teacher = GLiClassModel.from_pretrained(args.teacher, dtype=torch.float32).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    ttok = AutoTokenizer.from_pretrained(args.teacher)

# batches: alternate choice / noul
def batches(exs, bs):
    idx = list(range(len(exs))); rng.shuffle(idx)
    return [[exs[i] for i in idx[j:j + bs]] for j in range(0, len(idx), bs)]
cb = [("choice", b) for b in batches(choice_ex, args.bs)]
nb = [("noul", b) for b in batches(noul_ex, args.bs)]
plan = []
for i in range(max(len(cb), len(nb))):
    if i < len(cb): plan.append(cb[i])
    if i < len(nb): plan.append(nb[i])
if args.max_steps > 0:
    plan = plan[:args.max_steps]
total = len(plan)

enc_params = [p_ for n, p_ in model.named_parameters() if "encoder_model" in n]
head_params = [p_ for n, p_ in model.named_parameters() if "encoder_model" not in n]
opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
warm = max(1, int(0.06 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

model.train()
log = []; t0 = time.time(); run = {"choice": [], "noul": []}
for step, (task, b) in enumerate(plan):
    texts = [e[0] for e in b]; y = torch.tensor([e[1] for e in b])
    K = 8 if task == "choice" else 1
    enc_in = tok(texts, truncation=True, max_length=512, padding="longest", return_tensors="pt")
    lg = model(input_ids=enc_in["input_ids"], attention_mask=enc_in["attention_mask"], max_num_classes=K).logits[:, :K].float()
    if teacher is not None:
        with torch.no_grad():
            te = ttok(texts, truncation=True, max_length=512, padding="longest", return_tensors="pt")
            tl = teacher(input_ids=te["input_ids"], attention_mask=te["attention_mask"], max_num_classes=K).logits[:, :K].float()
    if task == "choice":
        loss = args.gold_w * (F.cross_entropy(lg, y) + 0.5 * F.binary_cross_entropy_with_logits(lg, F.one_hot(y, K).float()))
        if teacher is not None:
            loss = loss + args.kd_w * F.kl_div(F.log_softmax(lg, -1), F.softmax(tl, -1), reduction="batchmean")
    else:
        s = lg[:, 0]
        loss = args.gold_w * F.binary_cross_entropy_with_logits(s, y.float())
        if teacher is not None:
            loss = loss + args.kd_w * F.binary_cross_entropy_with_logits(s, torch.sigmoid(tl[:, 0]))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    run[task].append(float(loss))
    if (step + 1) % 50 == 0 or step + 1 == total:
        el = time.time() - t0
        rec = dict(step=step + 1, total=total, sec=round(el, 1), eta_min=round(el / (step + 1) * (total - step - 1) / 60, 1),
                   loss_choice=float(np.mean(run["choice"][-25:])) if run["choice"] else None,
                   loss_noul=float(np.mean(run["noul"][-25:])) if run["noul"] else None)
        log.append(rec); print(json.dumps(rec), flush=True)

model.eval()
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(args.init, f), os.path.join(args.out, f))
meta = dict(args=vars(args), n_train_rows_total=n_all, n_seen_utterances=len(utts), n_seen_intents=len(seen),
            n_choice_examples=len(choice_ex), n_noul_examples=len(noul_ex),
            noul_pos_frac=float(np.mean([e[1] for e in noul_ex])), steps=total, batch_size=args.bs,
            epochs_equiv_choice=len(choice_ex) / len(utts), epochs_equiv_noul=len(noul_ex) / len(utts),
            train_sec=round(time.time() - t0, 1), log=log)
json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"), indent=1)
print("DONE", json.dumps({k: v for k, v in meta.items() if k != "log"}))
