"""Skejja: modello decisionale System One piccolo e locale (Choice, Noul, Score) con API compatibile `/v1/systemone`.

    from skejja import Skejja, SystemOne
    engine = SystemOne(Skejja("percorso/del/bundle"), name="skejja-edge32k-ft6")
    engine.run({"state": "accendi la luce", "questions": {"azione": {"type": "choice",
                "criteria": {"luce_on": "turn on the lights", "luce_off": "turn off the lights"}}}})
"""
from skejja_infer import Skejja, fit_platt_map

from .systemone import RequestError, SystemOne

__version__ = "0.1.0"
__all__ = ["Skejja", "SystemOne", "RequestError", "fit_platt_map", "__version__"]
