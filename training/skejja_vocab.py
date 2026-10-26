"""Vocabulary trimming for BPE (Gemma/mmBERT-style) tokenizers + embedding remap for GLiClass uni-encoder.

Correctness argument: we keep a token set S closed under BPE ancestry (for every kept token t, both parts of
EVERY merge that produces t are kept, recursively) and keep exactly the merges (a,b) with a, b, a+b in S.
For any word whose original tokenization uses only tokens in S, every merge applied in the original greedy
derivation produces a token in S (it is an ancestor of a final token), hence is kept; removing merges that were
never chosen cannot change the greedy argmax at any step -> identical tokenization.
"""
import json, copy, collections, os

SPECIAL_KEEP = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>"]
GLICLASS_TOKENS = ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"]


def load_tok_json(path):
    with open(path) as f:
        return json.load(f)


def build_producers(tj):
    prod = collections.defaultdict(list)
    for a, b in tj["model"]["merges"]:
        prod[a + b].append((a, b))
    return prod


def closure(seed, producers, vocab):
    kept = set(t for t in seed if t in vocab)
    stack = list(kept)
    while stack:
        t = stack.pop()
        for a, b in producers.get(t, ()):
            for x in (a, b):
                if x not in kept and x in vocab:
                    kept.add(x); stack.append(x)
    return kept


def trimmed_tokenizer_json(tj, kept_tokens):
    """Return (new_tokenizer_json, old_ids list indexed by new id)."""
    vocab = tj["model"]["vocab"]
    added = {a["content"]: a for a in tj["added_tokens"]}
    byte_tokens = [t for t in vocab if t.startswith("<0x") and t.endswith(">") and len(t) == 6]
    order = []
    seen = set()
    def push(t):
        if t not in seen:
            seen.add(t); order.append(t)
    for t in SPECIAL_KEEP:
        push(t)
    for t in sorted(byte_tokens, key=lambda x: vocab[x]):
        push(t)
    for t in sorted((t for t in kept_tokens if t in vocab), key=lambda x: vocab[x]):
        push(t)
    for t in GLICLASS_TOKENS:
        push(t)
    old_id = {}
    for t in order:
        if t in vocab:
            old_id[t] = vocab[t]
        else:
            old_id[t] = added[t]["id"]
    new_vocab = {t: i for i, t in enumerate(order) if t in vocab or t in added}
    keep_set = set(new_vocab)
    new_merges = [[a, b] for a, b in tj["model"]["merges"] if a in keep_set and b in keep_set and (a + b) in keep_set]
    nt = copy.deepcopy({k: v for k, v in tj.items() if k != "model"})
    nt["model"] = {k: v for k, v in tj["model"].items() if k not in ("vocab", "merges")}
    # BPE vocab must contain the added special tokens too (they are in the original vocab for ids < 256000)
    nt["model"]["vocab"] = {t: i for t, i in new_vocab.items()}
    nt["model"]["merges"] = new_merges
    nt["added_tokens"] = []
    for a in tj["added_tokens"]:
        if a["content"] in new_vocab and (a["content"] in SPECIAL_KEEP or a["content"] in GLICLASS_TOKENS):
            b = dict(a); b["id"] = new_vocab[a["content"]]; nt["added_tokens"].append(b)
    nt["added_tokens"].sort(key=lambda a: a["id"])
    pp = nt["post_processor"]
    for k, sp in pp.get("special_tokens", {}).items():
        sp["ids"] = [new_vocab[x] for x in sp["tokens"]]
    old_ids = [old_id[t] for t in order]
    return nt, old_ids


def write_tokenizer_dir(nt, out_dir, orig_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(nt, f, ensure_ascii=False)
    with open(os.path.join(orig_dir, "tokenizer_config.json")) as f:
        cfg = json.load(f)
    ids = {a["content"]: a["id"] for a in nt["added_tokens"]}
    cfg["added_tokens_decoder"] = {str(a["id"]): {k: v for k, v in a.items() if k not in ("id",)} for a in nt["added_tokens"]}
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    with open(os.path.join(orig_dir, "special_tokens_map.json")) as f:
        stm = f.read()
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w") as f:
        f.write(stm)
    return ids
