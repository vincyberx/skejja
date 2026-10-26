"""Demo di Skejja per uno Space di Hugging Face (CPU gratuita): Choice e sì/no su una frase breve.

In locale: `pip install -r space/requirements.txt && python space/app.py`; con `SKEJJA_MODEL_URL=file://...` usa un
archivio locale invece di scaricarlo.
"""
import time

import gradio as gr

from skejja import Skejja, SystemOne, __version__
from skejja.models import DEFAULT_MODEL, resolve_bundle

ENGINE = SystemOne(Skejja(str(resolve_bundle()), threads=2), name=f"skejja-{DEFAULT_MODEL}")

CHOICE_EXAMPLES = [
    ["puoi spegnere le luci della cucina?", "lights_on: turn on the lights\nlights_off: turn off the lights\nmusic: play music\nweather: ask about the weather"],
    ["¿va a llover mañana en Madrid?", "lights_on: turn on the lights\nlights_off: turn off the lights\nmusic: play music\nweather: ask about the weather"],
    ["Mach bitte etwas Jazz an", "lights_on: turn on the lights\nlights_off: turn off the lights\nmusic: play music\nweather: ask about the weather"],
    ["I was charged twice for the same order", "billing: a payment or refund problem\nshipping: a delivery problem\naccess: a login or account access problem"],
]
NOUL_EXAMPLES = [
    ["ricordami di comprare il latte domani", "Does the user want to set a reminder?"],
    ["réveille-moi à 7 heures demain", "Does the user want to set an alarm?"],
    ["what time is it in Tokyo?", "Does the user want to set an alarm?"],
    ["sudo apt install nginx", "Does this command need administrator privileges?"],
]


def parse_options(text):
    opts = {}
    for i, line in enumerate(l.strip() for l in text.splitlines()):
        if not line:
            continue
        name, sep, desc = line.partition(":")
        name, desc = (name.strip(), desc.strip()) if sep else (f"option_{i + 1}", line)
        if not name or not desc:
            raise gr.Error(f"Option line {i + 1}: write it as `name: description`.")
        opts[name] = desc
    if not 2 <= len(opts) <= 20:
        raise gr.Error("Give between 2 and 20 options, one per line.")
    return opts


def choose(text, options):
    if not text.strip():
        raise gr.Error("Write a sentence first.")
    t0 = time.perf_counter()
    out = ENGINE.run({"state": text, "questions": {"q": {"type": "choice", "criteria": parse_options(options)}}})
    ms = (time.perf_counter() - t0) * 1000
    a = out["answers"]["q"]
    return a["probabilities"], f"**{a['choice']}** · confidence {a['confidence']:.2f} · {ms:.0f} ms on this CPU"


def yes_no(text, question):
    if not text.strip() or not question.strip():
        raise gr.Error("Write a sentence and a question.")
    t0 = time.perf_counter()
    out = ENGINE.run({"state": text, "questions": {"q": {"type": "noul", "instructions": question}}})
    ms = (time.perf_counter() - t0) * 1000
    p = out["answers"]["q"]["noul"]
    return {"yes": p, "no": 1 - p}, f"P(yes) = **{p:.2f}** · {ms:.0f} ms on this CPU"


with gr.Blocks(title="Skejja") as demo:
    gr.Markdown(
        f"# Skejja {__version__}\n"
        "A 57M-parameter decision model that runs on a Raspberry Pi. It does not write text: it returns "
        "probabilities for typed questions about a short sentence, in English, Italian, Spanish, German or French. "
        "Write the options in English.\n\n"
        "[Code](https://github.com/vincyberx/skejja) · [Model card](https://huggingface.co/vincyber/skejja-edge32k-ft6)"
        " · Built for short commands and intents: it is weak at reasoning, long texts and many options."
    )
    with gr.Tab("Choice"):
        with gr.Row():
            with gr.Column():
                c_text = gr.Textbox(label="Sentence", lines=2)
                c_opts = gr.Textbox(label="Options, one per line: name: description", lines=5)
                c_btn = gr.Button("Decide", variant="primary")
            with gr.Column():
                c_label = gr.Label(label="Probabilities", num_top_classes=8)
                c_info = gr.Markdown()
        gr.Examples(CHOICE_EXAMPLES, [c_text, c_opts])
        c_btn.click(choose, [c_text, c_opts], [c_label, c_info])
    with gr.Tab("Yes / no"):
        with gr.Row():
            with gr.Column():
                n_text = gr.Textbox(label="Sentence", lines=2)
                n_q = gr.Textbox(label="Question (works best as: Does the user want to ...?)")
                n_btn = gr.Button("Decide", variant="primary")
            with gr.Column():
                n_label = gr.Label(label="Answer")
                n_info = gr.Markdown()
        gr.Examples(NOUL_EXAMPLES, [n_text, n_q])
        n_btn.click(yes_no, [n_text, n_q], [n_label, n_info])
    gr.Markdown(
        "Yes/no probabilities on questions the model has never seen are not calibrated. "
        "Nothing you type is stored by this demo."
    )

if __name__ == "__main__":
    demo.launch()
