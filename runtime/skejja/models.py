"""Modelli pubblicati: scaricamento al primo uso, verifica SHA-256, estrazione sicura in una cache locale.

`resolve_bundle(spec)` accetta la cartella di un bundle già estratto oppure il nome di un modello noto (predefinito
`DEFAULT_MODEL`). L'archivio si scarica una volta sola in `$SKEJJA_HOME` (altrimenti `$XDG_CACHE_HOME/skejja` o
`~/.cache/skejja`), si controlla con l'hash fissato qui sotto e si estrae in `<cache>/<sha256[:12]>/`.
`SKEJJA_MODEL_URL` sostituisce l'URL (mirror, prove locali con file://): l'hash resta quello fissato.
"""
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

# repository Hugging Face dei pesi (account personale, deciso il 26 ottobre 2026; non ancora pubblicato).
# La revisione `main` basta: l'archivio è vincolato dal suo SHA-256.
HF_REPO = "vincyber/skejja-edge32k-ft6"
HF_URL = "https://huggingface.co/{repo}/resolve/main/{file}"

MODELS = {
    "edge32k-ft6-int8": dict(file="skejja-edge32k-ft6_onnx-int8.tar.gz", size=208260729,
                             sha256="751e243f1fba39421d36d44a832fc42a34bb74c6bec68f15d956e219e1131f42"),
    "edge32k-ft6-fp32": dict(file="skejja-edge32k-ft6_onnx-fp32.tar.gz", size=212344559,
                             sha256="8551e44970a48c1b70f5c51d7f5f27e8639739e79e6c40aa02be3c697591c73c"),
}
DEFAULT_MODEL = "edge32k-ft6-int8"
DOWNLOAD_DEADLINE_S = 3600


class ModelError(Exception):
    pass


def cache_dir():
    if os.environ.get("SKEJJA_HOME"):
        return Path(os.environ["SKEJJA_HOME"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "skejja"


def model_url(name):
    return os.environ.get("SKEJJA_MODEL_URL") or HF_URL.format(repo=HF_REPO, file=MODELS[name]["file"])


def resolve_bundle(spec=None, quiet=False):
    """Cartella di un bundle pronto all'uso: `spec` è il nome di un modello noto o una cartella esistente.
    Il nome di un modello vince su una cartella con lo stesso nome nella directory corrente (per quella si scrive
    `./nome`): così il modello predefinito non si può sostituire mettendo una cartella dove si lancia il comando."""
    spec = spec or DEFAULT_MODEL
    if spec in MODELS:
        return fetch(spec, quiet=quiet)
    p = Path(spec).expanduser()
    if p.is_dir():
        return p
    raise ModelError(f"bundle non trovato: {p} (modelli noti: {', '.join(MODELS)})")


def fetch(name, quiet=False):
    m = MODELS[name]
    dest = cache_dir() / m["sha256"][:12]
    done = dest / ".complete"
    if done.exists():
        return dest / done.read_text().strip()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = model_url(name)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".scarico-") as tmp:
        archive = Path(tmp) / m["file"]
        if not quiet:
            print(f"scarico {name} ({m['size'] / 1e6:.0f} MB) da {url}", file=sys.stderr)
        h, n, deadline = hashlib.sha256(), 0, time.monotonic() + DOWNLOAD_DEADLINE_S
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(archive, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    n += len(chunk)
                    if n > m["size"]:   # mai più byte di quelli attesi: un server ostile non riempie il disco
                        raise ModelError(f"{m['file']}: più grande dei {m['size']} byte attesi, scaricamento fermato")
                    if time.monotonic() > deadline:
                        raise ModelError(f"{m['file']}: scaricamento oltre {DOWNLOAD_DEADLINE_S} s, fermato")
                    h.update(chunk)
                    f.write(chunk)
        except OSError as e:
            raise ModelError(f"scaricamento non riuscito da {url}: {e}") from e
        if n != m["size"]:
            raise ModelError(f"{m['file']}: {n} byte invece dei {m['size']} attesi: archivio scartato")
        if h.hexdigest() != m["sha256"]:
            raise ModelError(f"SHA-256 diverso da quello atteso per {m['file']}: archivio scartato")
        top = safe_extract(archive, Path(tmp) / "x")
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(Path(tmp) / "x", dest)
    done.write_text(top)
    return dest / top


def safe_extract(archive, out):
    """Estrae solo file e cartelle sotto un'unica cartella in cima, niente link né percorsi che escono."""
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        tops = {PurePosixPath(m.name).parts[0] for m in members if m.name}
        for m in members:
            parts = PurePosixPath(m.name).parts
            if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in parts or len(tops) != 1:
                raise ModelError(f"archivio non valido: {m.name}")
        out.mkdir(parents=True)
        for m in members:
            target = out / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            target.chmod(0o644)
    return tops.pop()
