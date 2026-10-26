"""Test dello scaricamento dei modelli (archivi finti in file://, niente rete) e della riga di comando."""
import hashlib
import io
import tarfile

import pytest

from skejja import __version__, models
from skejja.cli import main


def make_tar(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), 0o600
            tf.addfile(ti, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    def setup(members, sha=None):
        arc = tmp_path / "bundle.tar.gz"
        digest = make_tar(arc, members)
        monkeypatch.setitem(models.MODELS, "finto", dict(file=arc.name, size=arc.stat().st_size, sha256=sha or digest))
        monkeypatch.setenv("SKEJJA_MODEL_URL", arc.as_uri())
        monkeypatch.setenv("SKEJJA_HOME", str(tmp_path / "cache"))
        return tmp_path / "cache"
    return setup


def test_fetch_extracts_verifies_and_reuses_cache(fake_model, monkeypatch):
    cache = fake_model({"skejja-finto_onnx-int8/calibration.json": b"{}", "skejja-finto_onnx-int8/m.onnx": b"x"})
    d = models.resolve_bundle("finto", quiet=True)
    assert d.name == "skejja-finto_onnx-int8" and d.parent.parent == cache
    assert (d / "m.onnx").read_bytes() == b"x"
    assert (d / "m.onnx").stat().st_mode & 0o777 == 0o644      # leggibile da tutti anche se l'archivio dice 600
    monkeypatch.setenv("SKEJJA_MODEL_URL", "file:///non/esiste.tar.gz")   # seconda volta: niente scaricamento
    assert models.resolve_bundle("finto", quiet=True) == d


def test_wrong_hash_leaves_nothing(fake_model):
    cache = fake_model({"b/x": b"x"}, sha="0" * 64)
    with pytest.raises(models.ModelError, match="SHA-256"):
        models.resolve_bundle("finto", quiet=True)
    assert not (cache / ("0" * 12)).exists()


@pytest.mark.parametrize("name", ["../fuori", "/assoluto", "a/../../fuori"])
def test_unsafe_paths_rejected(fake_model, name):
    fake_model({"b/ok": b"x", name: b"y"})
    with pytest.raises(models.ModelError, match="archivio non valido"):
        models.resolve_bundle("finto", quiet=True)


@pytest.mark.parametrize("delta", [-1, 1])
def test_size_must_match_exactly(fake_model, monkeypatch, delta):
    fake_model({"b/x": b"x"})
    m = dict(models.MODELS["finto"]); m["size"] += delta
    monkeypatch.setitem(models.MODELS, "finto", m)
    with pytest.raises(models.ModelError, match="byte"):
        models.resolve_bundle("finto", quiet=True)


def test_model_name_wins_over_folder_in_cwd(fake_model, tmp_path, monkeypatch):
    fake_model({"skejja-finto_onnx-int8/m.onnx": b"x"})
    (tmp_path / "lavoro" / "finto").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "lavoro")
    assert models.resolve_bundle("finto", quiet=True).name == "skejja-finto_onnx-int8"
    assert models.resolve_bundle("./finto", quiet=True).resolve() == (tmp_path / "lavoro" / "finto").resolve()


def test_two_top_folders_rejected(fake_model):
    fake_model({"a/x": b"x", "b/y": b"y"})
    with pytest.raises(models.ModelError):
        models.resolve_bundle("finto", quiet=True)


def test_local_dir_and_unknown_name(tmp_path):
    assert models.resolve_bundle(str(tmp_path)) == tmp_path
    with pytest.raises(models.ModelError, match="modelli noti"):
        models.resolve_bundle("non-esiste")


def test_pinned_models_are_complete():
    assert models.DEFAULT_MODEL in models.MODELS
    for m in models.MODELS.values():
        assert len(m["sha256"]) == 64 and m["size"] > 0 and m["file"].endswith(".tar.gz")


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"skejja {__version__}"


def test_cli_download_prints_folder(fake_model, capsys, monkeypatch):
    fake_model({"skejja-finto_onnx-int8/m.onnx": b"x"})
    monkeypatch.setattr(models, "DEFAULT_MODEL", "finto")
    main(["download", "finto"])
    assert capsys.readouterr().out.strip().endswith("skejja-finto_onnx-int8")


def test_pinned_hashes_match_local_release_files():
    """Se gli archivi della release ci sono (releases/ è fuori da git), gli hash fissati devono coincidere."""
    import pathlib
    sums = pathlib.Path(__file__).parent.parent / "releases" / "v0.1.0" / "SHA256SUMS"
    if not sums.exists():
        pytest.skip("archivi della release non presenti")
    local = dict(reversed(l.split("  ")) for l in sums.read_text().splitlines())
    for m in models.MODELS.values():
        assert local[m["file"]] == m["sha256"], m["file"]
        assert (sums.parent / m["file"]).stat().st_size == m["size"], m["file"]
