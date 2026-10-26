"""Server HTTP compatibile con `POST /v1/systemone`: i client di Jev funzionano cambiando solo l'indirizzo.

Richiede l'extra `server` (`pip install "skejja[server]"`). Se `api_key` è impostata, ogni richiesta deve portare
`Authorization: Bearer <chiave>`; altrimenti il server è aperto, quindi per default ascolta solo su 127.0.0.1.
In più rispetto a Jev: `POST /v1/skejja/calibrations` (calibrazione per domanda delle noul) e
`GET /v1/skejja/calibrations`.
"""
import secrets
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from . import __version__
from .systemone import RequestError, SystemOne


def create_app(engine: SystemOne, api_key: str | None = None) -> FastAPI:
    def check_key(authorization: Annotated[str | None, Header()] = None) -> None:
        if not api_key:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="chiave API mancante o sbagliata")

    app = FastAPI(title="Skejja System One", version=__version__)

    @app.exception_handler(RequestError)
    def invalid_request(_, exc: RequestError) -> JSONResponse:
        return JSONResponse(status_code=400,
                            content={"error": {"type": "invalid_request_error", "message": exc.message,
                                               "param": exc.param}})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": engine.name}

    # `def` e non `async def`: onnxruntime blocca, quindi la chiamata gira nel threadpool di FastAPI
    @app.post("/v1/systemone", dependencies=[Depends(check_key)])
    def systemone(request: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        return engine.run(request)

    # in più rispetto a Jev: calibrazione per domanda delle noul (docs/api.md)
    @app.post("/v1/skejja/calibrations", dependencies=[Depends(check_key)])
    def calibrate(request: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        return engine.calibrate(request)

    @app.get("/v1/skejja/calibrations", dependencies=[Depends(check_key)])
    def calibrations() -> list[dict[str, Any]]:
        return engine.calibrations()

    return app
