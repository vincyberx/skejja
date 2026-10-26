# Immagine del server Skejja compatibile /v1/systemone. Il modello non è nell'immagine: si monta in /model
#   docker build -t skejja .
#   docker run --rm -p 127.0.0.1:8080:8080 -v /percorso/skejja-edge32k-ft6_onnx-int8:/model:ro \
#       -e SKEJJA_API_KEY=... -e SKEJJA_NAME=skejja-edge32k-ft6 skejja
# oppure si scarica al primo avvio in un volume (SKEJJA_BUNDLE = nome di un modello di skejja/models.py):
#   docker run --rm -p 127.0.0.1:8080:8080 -v skejja-cache:/cache -e SKEJJA_BUNDLE=edge32k-ft6-int8 skejja
# I bundle della v0.1.0 (releases/v0.1.0) sono leggibili da tutti; con quelli esportati fino al 17 ottobre 2026
# serviva --user "$(id -u):$(id -g)", perché i loro .onnx.data erano leggibili solo dal proprietario.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY runtime/ runtime/
RUN pip install --no-cache-dir ".[server]" \
    && useradd --system --no-create-home skejja \
    && mkdir /cache && chown skejja /cache

USER skejja
ENV SKEJJA_BUNDLE=/model \
    SKEJJA_HOME=/cache \
    SKEJJA_THREADS=4 \
    SKEJJA_MAX_TOKENS=512 \
    SKEJJA_NAME=skejja
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"
# dentro il container serve 0.0.0.0; l'esposizione la decide `docker run -p` (meglio 127.0.0.1:...)
CMD ["sh", "-c", "exec skejja serve \"$SKEJJA_BUNDLE\" --host 0.0.0.0 --port 8080 --threads \"$SKEJJA_THREADS\" --max-tokens \"$SKEJJA_MAX_TOKENS\" --name \"$SKEJJA_NAME\""]
