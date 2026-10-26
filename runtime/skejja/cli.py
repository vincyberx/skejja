"""Riga di comando: `skejja serve [bundle]` avvia il server HTTP, `skejja mcp [bundle]` il server MCP (stdio),
`skejja run [bundle] richiesta.json` risponde a una richiesta, `skejja download [modello]` scarica un modello.
Senza bundle si usa il modello predefinito, scaricato al primo uso (skejja/models.py)."""
import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .models import DEFAULT_MODEL, MODELS, ModelError, resolve_bundle
from .systemone import DEFAULT_MAX_TOKENS, SystemOne


def bundle_path(spec):
    try:
        return resolve_bundle(spec)
    except ModelError as e:
        sys.exit(str(e))


def load_engine(a):
    from skejja_infer import Skejja

    bundle = bundle_path(a.bundle)
    model = Skejja(str(bundle), threads=a.threads, model_file=a.model_file, int8dyn=a.int8dyn)
    name = a.name or f"skejja-{bundle.name.split('_onnx')[0].removeprefix('skejja-')}:{model.model_file}"
    return SystemOne(model, name=name, max_tokens=a.max_tokens, calibrations_path=a.calibrations)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skejja", description=__doc__)
    ap.add_argument("--version", action="version", version=f"skejja {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("bundle", nargs="?", default=None,
                        help="cartella del bundle ONNX (tokenizer.json, calibration.json, *.onnx) o nome di un modello "
                             f"({', '.join(MODELS)}); predefinito {DEFAULT_MODEL}, scaricato al primo uso")
    common.add_argument("--threads", type=int, default=4)
    common.add_argument("--model-file", help="un .onnx preciso del bundle (altrimenti la scelta automatica)")
    common.add_argument("--int8dyn", choices=["auto", "yes", "no"], default="auto")
    common.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="oltre questa lunghezza l'ingresso si accorcia testa e coda")
    common.add_argument("--name", help="nome del modello nelle risposte")
    common.add_argument("--calibrations", help="file JSON dove leggere e salvare le calibrazioni per domanda")
    s = sub.add_parser("serve", parents=[common], help="server HTTP compatibile con /v1/systemone")
    s.add_argument("--host", default="127.0.0.1", help="indirizzo di ascolto (predefinito: solo questa macchina)")
    s.add_argument("--port", type=int, default=8080)
    r = sub.add_parser("run", parents=[common], help="risponde a una richiesta JSON (file o - per stdin)")
    r.add_argument("request")
    sub.add_parser("mcp", parents=[common], help="server MCP in stdio, per Claude e altri agenti")
    d = sub.add_parser("download", help="scarica e controlla un modello nella cache, poi stampa la sua cartella")
    d.add_argument("model", nargs="?", default=DEFAULT_MODEL, choices=list(MODELS))
    a = ap.parse_args(argv)

    if a.cmd == "download":
        print(bundle_path(a.model))
        return

    engine = load_engine(a)
    if a.cmd == "mcp":
        try:
            from .mcp_server import create_mcp
        except ImportError:
            sys.exit('per il server MCP serve l\'extra: pip install "skejja[mcp]"')
        create_mcp(engine).run("stdio")
        return
    if a.cmd == "run":
        raw = sys.stdin.read() if a.request == "-" else Path(a.request).read_text()
        print(json.dumps(engine.run(json.loads(raw)), ensure_ascii=False, indent=2))
        return
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        sys.exit('per il server serve l\'extra: pip install "skejja[server]"')
    api_key = os.environ.get("SKEJJA_API_KEY") or None
    if a.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print("attenzione: server raggiungibile dalla rete senza SKEJJA_API_KEY", file=sys.stderr)
    uvicorn.run(create_app(engine, api_key=api_key), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
