"""Ricostruisce i checkpoint PyTorch dei modelli Skejja a 32k token (fase 1 non li aveva salvati).

1. zero-shot (`edge32k-zs`): GLiClass multilang edge originale + tokenizer ridotto del bundle; le righe della tabella
   degli embedding vengono prese dall'originale per contenuto del token (nessun addestramento).
2. fine-tuned (`edge32k-ft`): stessa architettura, pesi importati dal bundle ONNX fp32 (pesi esatti della fase 1).
   I pesi con nome anonimo (`onnx::MatMul_*`) sono le Linear trasposte: si riconoscono dal nome del nodo MatMul
   che li usa (es. `/enc/layers.0/attn/Wqkv/MatMul` -> `enc.layers.0.attn.Wqkv.weight`).
3. Verifica: logit torch contro ONNX fp32 (ft) e contro il modello originale (zs, frasi a tokenizzazione identica).

uso:
  HF_HUB_DISABLE_XET=1 python training/rebuild_checkpoints.py \
      --ft_bundle /tmp/sk/skejja-edge32k-ft_onnx-fp32 --out releases/checkpoints
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import skejja_glic as fg  # noqa: E402

ORIG_ID = "knowledgator/gliclass-multilang-edge"
ORIG_REV = "d16c08ef70547514081952104e6fc3d190d9ee39"
TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def token_table(tj):
    """content -> id, model vocab plus added tokens."""
    t = dict(tj["model"]["vocab"])
    for a in tj["added_tokens"]:
        t[a["content"]] = a["id"]
    return t


def build_zs(orig_dir, tok_dir, out_dir):
    from gliclass import GLiClassModel
    old = token_table(json.load(open(os.path.join(orig_dir, "tokenizer.json"))))
    new = token_table(json.load(open(os.path.join(tok_dir, "tokenizer.json"))))
    V = max(new.values()) + 1
    assert sorted(new.values()) == list(range(V)), "ids del tokenizer ridotto non contigui"
    inv = {i: t for t, i in new.items()}
    old_ids = torch.tensor([old[inv[i]] for i in range(V)])
    model = GLiClassModel.from_pretrained(orig_dir, dtype=torch.float32)
    cfg = model.config
    old_special = {k: getattr(cfg, k) for k in ("class_token_index", "text_token_index", "example_token_index")}
    old_inv = {i: t for t, i in old.items()}
    enc = model.model.encoder_model
    emb = enc.embeddings.tok_embeddings
    W = emb.weight.data[old_ids].clone()
    new_emb = torch.nn.Embedding(V, W.shape[1], padding_idx=emb.padding_idx)
    new_emb.weight.data.copy_(W)
    enc.embeddings.tok_embeddings = new_emb
    for k, oid in old_special.items():
        setattr(cfg, k, int(new[old_inv[oid]]))
    for c in (cfg, cfg.encoder_config, enc.config):
        c.vocab_size = V
        for k in ("pad_token_id", "bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "mask_token_id"):
            v = getattr(c, k, None)
            if v is not None:
                assert new[old_inv[v]] == v, f"{k} cambia id: aggiornare il codice"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    for f in TOK_FILES:
        shutil.copy(os.path.join(tok_dir, f), os.path.join(out_dir, f))
    return model


def import_onnx(model, onnx_path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path, load_external_data=True)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    users = {}
    for nd in m.graph.node:
        for x in nd.input:
            if x in inits:
                users.setdefault(x, []).append(nd)
    prefix = {"enc.": "model.encoder_model.", "text_projector.": "model.text_projector.",
              "classes_projector.": "model.classes_projector.", "scorer.": "model.scorer."}
    sd = model.state_dict(); new_sd = {}
    for name, arr in inits.items():
        if name.startswith("onnx::"):
            nds = users[name]
            assert len(nds) == 1 and nds[0].op_type == "MatMul", name
            path = nds[0].name.strip("/").split("/")[:-1]          # ['enc', 'layers.0', 'attn', 'Wqkv']
            # nn.Sequential ripete il nome: ['scorer', 'score_mlp', 'score_mlp.0'] -> scorer.score_mlp.0
            path = [s for i, s in enumerate(path) if not (i + 1 < len(path) and path[i + 1].startswith(s + "."))]
            pname = ".".join(path) + ".weight"; arr = arr.T
        else:
            pname = name
        tname = next((v + pname[len(k):] for k, v in prefix.items() if pname.startswith(k)), None)
        if tname is None or tname not in sd:
            continue  # costanti del grafo (es. indici, rotary) che non sono parametri
        assert tuple(sd[tname].shape) == arr.shape, (tname, sd[tname].shape, arr.shape)
        new_sd[tname] = torch.from_numpy(np.ascontiguousarray(arr)).float()
    params = {n for n, _ in model.named_parameters()}
    # logit_scale non entra nel calcolo con lo scorer cross-attn: resta quello dello zero-shot
    missing = sorted(params - set(new_sd) - {"model.logit_scale"})
    assert not missing, f"parametri non trovati nell'ONNX: {missing[:10]}"
    model.load_state_dict({**sd, **new_sd})
    return len(new_sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_bundle", required=True, help="cartella estratta di skejja-edge32k-ft_onnx-fp32")
    ap.add_argument("--orig", default="", help="cartella del modello originale (default: download HF alla revisione fissata)")
    ap.add_argument("--out", default="releases/checkpoints")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "bench", "data"))
    ap.add_argument("--n_check", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count()))
    orig = a.orig
    if not orig:
        from huggingface_hub import snapshot_download
        orig = snapshot_download(ORIG_ID, revision=ORIG_REV)
    onnx_ft = [f for f in os.listdir(a.ft_bundle) if f.endswith(".fp32.onnx")][0]
    zs_dir, ft_dir = os.path.join(a.out, "edge32k-zs"), os.path.join(a.out, "edge32k-ft")
    zs = build_zs(orig, a.ft_bundle, zs_dir)
    from gliclass import GLiClassModel
    ft = GLiClassModel.from_pretrained(zs_dir, dtype=torch.float32)
    n_imp = import_onnx(ft, os.path.join(a.ft_bundle, onnx_ft))
    ft.save_pretrained(ft_dir, safe_serialization=True)
    for f in TOK_FILES:
        shutil.copy(os.path.join(a.ft_bundle, f), os.path.join(ft_dir, f))

    # ---- verifica
    import onnxruntime as ort
    ort.disable_telemetry_events()
    import tokenizers
    from transformers import AutoTokenizer
    rows_c = [json.loads(l) for l in open(os.path.join(a.data, "test_choice.jsonl"))][:a.n_check]
    rows_n = [json.loads(l) for l in open(os.path.join(a.data, "test_noul.jsonl"))][:a.n_check]
    tok = AutoTokenizer.from_pretrained(ft_dir)
    sess = ort.InferenceSession(os.path.join(a.ft_bundle, onnx_ft), providers=["CPUExecutionProvider"])
    ortok = tokenizers.Tokenizer.from_file(os.path.join(a.ft_bundle, "tokenizer.json"))

    def ort_logits(inputs, K):
        out = []
        for s in inputs:
            ids = np.array([ortok.encode(s).ids], np.int64)
            out.append(sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids),
                                             "label_slots": np.zeros(K, np.float32)})[0][0, :K])
        return np.array(out)

    rep = {"onnx_ft": onnx_ft, "n_params_imported": n_imp}
    ft.eval(); zs.eval()
    for task, inputs, K in (("choice", fg.choice_inputs(rows_c), 8), ("noul", fg.noul_inputs(rows_n), 1)):
        lt = fg.score_torch(ft, tok, inputs, K)
        lo = ort_logits(inputs, K)
        rep[f"ft_vs_onnx_{task}_max_abs"] = float(np.abs(lt - lo).max())
        # zs: confronto con l'originale sulle frasi a tokenizzazione identica (stessi token come stringhe)
        otok = AutoTokenizer.from_pretrained(orig)
        same = [i for i, s in enumerate(inputs)
                if tok.convert_ids_to_tokens(tok(s)["input_ids"]) == otok.convert_ids_to_tokens(otok(s)["input_ids"])]
        orig_m = GLiClassModel.from_pretrained(orig, dtype=torch.float32).eval()
        sub = [inputs[i] for i in same]
        d = np.abs(fg.score_torch(zs, tok, sub, K) - fg.score_torch(orig_m, otok, sub, K))
        rep[f"zs_vs_orig_{task}_max_abs"] = float(d.max()); rep[f"zs_vs_orig_{task}_n_same_tok"] = len(same)
        del orig_m
    for k, v in rep.items():
        print(f"{k}={v}")
    json.dump(rep, open(os.path.join(a.out, "rebuild_check.json"), "w"), indent=1)
    assert rep["ft_vs_onnx_choice_max_abs"] < 1e-3 and rep["ft_vs_onnx_noul_max_abs"] < 1e-3
    assert rep["zs_vs_orig_choice_max_abs"] < 1e-3 and rep["zs_vs_orig_noul_max_abs"] < 1e-3


if __name__ == "__main__":
    main()
