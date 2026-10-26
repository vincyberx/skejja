"""Da checkpoint PyTorch a bundle ONNX come nella fase 1: fp32 (dati esterni), int8 solo pesi (MatMulNBits 8 bit,
blocco 32, simmetrico, accuracy_level 4, MatMul del rotary escluso) e int8 dinamico (QInt8 per canale).
Calibrazione (solo file calib_*): temperatura Choice su calib_choice.jsonl; Platt Noul su calib_noul.jsonl
(`platt_calib`) e, se indicato, su un secondo file (`platt_calib_ext`). `platt_a`/`platt_b` = quella scelta con
--platt (letta dal runtime). Verifica numerica contro torch su --n_check item Choice e Noul del test.

uso: python training/export_bundle.py --ckpt releases/checkpoints/run_A/best --name edge32k-ft2 \
        --out_dir /tmp/bundles --calib_ext bench/data/calib_noul_ext.jsonl --platt calib_ext
"""
import argparse, json, os, shutil, sys, tarfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import skejja_measure as fm  # noqa: E402
import skejja_glic as fg  # noqa: E402
import skejja_onnx as so  # noqa: E402


def read(p):
    return [json.loads(l) for l in open(p)]


def readable(info):
    """Nel tar ogni file è leggibile da tutti (644, cartelle 755), qualunque sia la umask di chi esporta."""
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="prefisso dei file, es. edge32k-ft2 -> edge-v32k-ft2.fp32.onnx")
    ap.add_argument("--file_stem", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "bench", "data"))
    ap.add_argument("--calib_ext", default="")
    ap.add_argument("--platt", choices=["calib", "calib_ext"], default="calib")
    ap.add_argument("--n_check", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dyn_exclude", default="mlp/Wo",
                    help="nodi lasciati in fp32 nell'int8 dinamico (sottostringhe del nome, separate da virgola; '' = nessuno). "
                         "Default mlp/Wo: con ft5 l'int8 dinamico completo perde 9 punti di Choice sulle intenzioni mai viste")
    a = ap.parse_args()
    import onnx, onnxruntime as ort
    ort.disable_telemetry_events()
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer
    torch.set_num_threads(a.threads)
    stem = a.file_stem or a.name.replace("edge32k", "edge-v32k")
    bdir = os.path.join(a.out_dir, f"skejja-{a.name}_onnx")
    os.makedirs(bdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = GLiClassModel.from_pretrained(a.ckpt, dtype=torch.float32).eval()
    scorer = fg.Scorer(model).eval()
    tmp = os.path.join(bdir, "_tmp.onnx")
    so.export_onnx(scorer, tok, tmp)
    files = {"fp32": f"{stem}.fp32.onnx", "int8": f"{stem}.int8.onnx", "int8dyn": f"{stem}.int8dyn.onnx"}
    import tempfile
    stage = tempfile.mkdtemp(prefix="_stage_", dir=bdir)

    def save_ext(model_proto, name):
        """onnx.save con dati esterni ACCODA a un .data già esistente: si scrive in una cartella nuova e poi si
        sostituiscono i file finali (os.replace), così un nuovo export non raddoppia i file."""
        onnx.save(model_proto, os.path.join(stage, name), save_as_external_data=True, all_tensors_to_one_file=True,
                  location=name + ".data", size_threshold=0)
        for f in (name, name + ".data"):
            os.replace(os.path.join(stage, f), os.path.join(bdir, f))
            # il .data nasceva con permessi 600: chi estrae il bundle con un altro utente (Docker, server) non lo
            # poteva leggere (onnxruntime: SystemError 13, 17 ottobre 2026)
            os.chmod(os.path.join(bdir, f), 0o644)
    m = onnx.load(tmp)
    save_ext(m, files["fp32"])
    rot = [n.name for n in m.graph.node if n.op_type == "MatMul" and "rotary" in n.name]
    m = onnx.load(tmp)   # copia nuova: il salvataggio con dati esterni modifica il modello in memoria
    q = MatMulNBitsQuantizer(m, bits=8, block_size=32, is_symmetric=True, accuracy_level=4, nodes_to_exclude=rot)
    q.process()
    save_ext(q.model.model, files["int8"])
    excl = tuple(x for x in a.dyn_exclude.split(",") if x)
    so.quantize_int8(tmp, os.path.join(bdir, "_tmp_dyn.onnx"), per_channel=True, exclude_patterns=excl)
    md = onnx.load(os.path.join(bdir, "_tmp_dyn.onnx"))
    save_ext(md, files["int8dyn"])
    os.rmdir(stage)
    for f in os.listdir(bdir):
        if f.startswith("_tmp"):
            os.remove(os.path.join(bdir, f))
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(os.path.join(a.ckpt, f), os.path.join(bdir, f))
    shutil.copy(os.path.join(HERE, "..", "runtime", "skejja_infer.py"), bdir)

    D = a.data
    cc, cn = read(f"{D}/calib_choice.jsonl"), read(f"{D}/calib_noul.jsonl")
    tc, tn = read(f"{D}/test_choice.jsonl")[:a.n_check], read(f"{D}/test_noul.jsonl")[:a.n_check]
    ce = read(a.calib_ext) if a.calib_ext else None
    ref_c = fg.score_torch(model, tok, fg.choice_inputs(tc), 8); ref_n = fg.score_torch(model, tok, fg.noul_inputs(tn), 1)
    cal, check = {}, {}
    for var, f in files.items():
        sess = so.ort_session(os.path.join(bdir, f), threads=a.threads)
        bs = 1 if var == "int8dyn" else 64   # int8 dinamico: scala delle attivazioni dipende dal batch
        sc = lambda rows, K, fn: so.score_ort(sess, tok, fn(rows), K, batch_size=bs)
        T = fm.fit_temperature(sc(cc, 8, fg.choice_inputs), [r["gold_idx"] for r in cc])
        pc = fm.fit_platt(sc(cn, 1, fg.noul_inputs)[:, 0], [r["label"] for r in cn])
        ent = dict(temperature=T, platt_calib=list(pc), model=a.name, variant=f"onnx-{var}",
                   fitted_on="calib_choice.jsonl / calib_noul.jsonl" + (f" / {os.path.basename(a.calib_ext)}" if ce else ""))
        if ce:
            ent["platt_calib_ext"] = list(fm.fit_platt(sc(ce, 1, fg.noul_inputs)[:, 0], [r["label"] for r in ce]))
        pa, pb = ent["platt_calib_ext" if a.platt == "calib_ext" else "platt_calib"]
        ent.update(platt_a=pa, platt_b=pb, platt_used=f"platt_{a.platt}")
        cal[f] = ent
        lc, ln = sc(tc, 8, fg.choice_inputs), sc(tn, 1, fg.noul_inputs)
        check[var] = dict(choice_max_abs=float(np.abs(lc - ref_c).max()), choice_argmax_agree=float((lc.argmax(1) == ref_c.argmax(1)).mean()),
                          noul_max_abs=float(np.abs(ln - ref_n).max()), noul_decision_agree=float(((ln[:, 0] > 0) == (ref_n[:, 0] > 0)).mean()))
        print(var, json.dumps(check[var]), flush=True)
    json.dump(cal, open(os.path.join(bdir, "calibration.json"), "w"), indent=1)
    json.dump(dict(ckpt=os.path.basename(os.path.normpath(a.ckpt)), check_vs_torch=check, n_check=a.n_check,
                   int8dyn_excluded=list(excl)),
              open(os.path.join(bdir, "export_check.json"), "w"), indent=1)
    open(os.path.join(bdir, "README.md"), "w").write(
        f"# skejja-{a.name}\n\nBundle ONNX (fase 2): `{files['fp32']}`, `{files['int8']}` (int8 solo pesi), "
        f"`{files['int8dyn']}` (int8 dinamico, consigliato su CPU ARM senza dot-product; lasciati in fp32: {', '.join(excl) or 'nessuno'}).\n"
        "Tenere ogni `.onnx` insieme al suo `.onnx.data`. Calibrazione in `calibration.json` (solo file calib_*).\n"
        "Uso: `python skejja_infer.py . \"testo\" \"Does the user want to X?\"`\n")
    for tag, keep in (("fp32", ["fp32"]), ("int8", ["int8", "int8dyn"])):
        tpath = os.path.join(a.out_dir, f"skejja-{a.name}_onnx-{tag}.tar.gz")
        with tarfile.open(tpath, "w:gz") as tf:
            for f in sorted(os.listdir(bdir)):
                if f.endswith((".onnx", ".onnx.data")) and not any(f.startswith(f"{stem}.{k}.") for k in keep):
                    continue
                tf.add(os.path.join(bdir, f), arcname=f"skejja-{a.name}_onnx-{tag}/{f}", filter=readable)
        print(tpath)


if __name__ == "__main__":
    main()
