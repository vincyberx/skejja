"""Riconfeziona un bundle .tar.gz per la pubblicazione, senza toccare i file del modello.

- permessi 644 per i file e 755 per le cartelle, proprietario 0:0 senza nomi (i bundle esportati fino al 17 ottobre
  2026 avevano i .onnx.data leggibili solo dal proprietario);
- `skejja_infer.py` sostituito con quello attuale di runtime/, README.md in inglese, LICENSE e NOTICE aggiunti;
- tutti gli altri file (.onnx, .onnx.data, tokenizer, calibration.json, export_check.json) copiati byte per byte:
  lo script lo verifica con SHA-256 dopo aver scritto l'archivio, e scrive SHA256SUMS accanto.

uso: python scripts/repack_bundle.py releases/skejja-edge32k-ft6_onnx-int8.tar.gz releases/v0.1.0 --version 0.1.0
"""
import argparse, hashlib, io, os, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPLACED = {"skejja_infer.py": os.path.join(ROOT, "runtime", "skejja_infer.py"),
            "LICENSE": os.path.join(ROOT, "LICENSE"), "NOTICE": os.path.join(ROOT, "NOTICE")}

README = """# {name}

Skejja {version}: a small local System One decision model (Choice, Noul, Score) for modest CPUs.
This archive holds the ONNX model, its tokenizer and its calibration; the code is in the `skejja` Python package.

Files: {models}. Keep every `.onnx` next to its `.onnx.data`. `calibration.json` was fitted only on the
`calib_*` files of the benchmark; `export_check.json` compares the ONNX outputs with PyTorch.
{int8_note}
Use with the package:

    pip install "skejja[server]"
    skejja serve path/to/{name}

or standalone (needs only `onnxruntime`, `tokenizers`, `numpy`):

    python skejja_infer.py . "turn on the lights" "turn on the lights|turn off the lights|play music"

License: Apache-2.0 (see LICENSE); attributions in NOTICE.
"""
INT8_NOTE = ("`*.int8.onnx` has 8-bit weights; `*.int8dyn.onnx` uses dynamic int8 and is picked automatically on ARM "
             "CPUs without the dot-product extension (for example the Raspberry Pi 3 and 4).\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def info(name, size, mtime, isdir=False):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime = (0 if isdir else size), mtime
    ti.type = tarfile.DIRTYPE if isdir else tarfile.REGTYPE
    ti.mode = 0o755 if isdir else 0o644
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dst = os.path.join(a.out_dir, os.path.basename(a.src))
    with tarfile.open(a.src, "r:gz") as tf:
        members = [m for m in tf.getmembers()]
        top = members[0].name.split("/")[0]
        if any(m.name.split("/")[0] != top or m.issym() or m.islnk() for m in members):
            raise SystemExit("archivio inatteso: una sola cartella in cima, niente link")
        kept = {}
        for m in members:
            if m.isfile():
                base = m.name.split("/", 1)[1]
                if base not in REPLACED and base != "README.md":
                    kept[base] = (tf.extractfile(m).read(), m.mtime)
    mtime = max(t for _, t in kept.values())
    files = dict((k, v[0]) for k, v in kept.items())
    for base, path in REPLACED.items():
        files[base] = open(path, "rb").read()
    models = sorted(f for f in files if f.endswith(".onnx"))
    files["README.md"] = README.format(name=top, version=a.version, models=", ".join(f"`{m}`" for m in models),
                                       int8_note=INT8_NOTE if any("int8" in m for m in models) else "").encode()
    with tarfile.open(dst, "w:gz") as out:
        out.addfile(info(top, 0, mtime, isdir=True))
        for base in sorted(files):
            out.addfile(info(f"{top}/{base}", len(files[base]), mtime), io.BytesIO(files[base]))
    # verifica: i file del modello sono identici all'originale, permessi e nomi come previsto
    with tarfile.open(dst, "r:gz") as tf:
        for m in tf.getmembers():
            assert m.mode == (0o755 if m.isdir() else 0o644), m.name
            base = m.name.split("/", 1)[1] if "/" in m.name else None
            if base in kept:
                assert sha256(tf.extractfile(m).read()) == sha256(kept[base][0]), base
    digest = sha256(open(dst, "rb").read())
    sums = os.path.join(a.out_dir, "SHA256SUMS")
    lines = [l for l in (open(sums).read().splitlines() if os.path.exists(sums) else [])
             if not l.endswith("  " + os.path.basename(dst))]
    open(sums, "w").write("\n".join(sorted(lines + [f"{digest}  {os.path.basename(dst)}"])) + "\n")
    print(f"{dst}\n  sha256 {digest}\n  identici all'originale: {', '.join(sorted(kept))}")


if __name__ == "__main__":
    main()
