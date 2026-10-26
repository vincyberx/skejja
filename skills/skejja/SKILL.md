---
name: skejja
description: Fast, local, private decisions on short texts with the Skejja MCP server (tools choose, yes_no, systemone, calibrate_question). Use it to route a user message to one of a few concrete actions, to check yes/no conditions about what the user wants, or to screen agent actions, in English, Italian, Spanish, German or French, when calling a large model would be slower, costlier or would send the text off the machine. Do not use it for reasoning, long documents, math, facts it would have to know, or choices among many options.
---

# Skejja: small local decisions

Skejja is a 57M-parameter decision model. It never writes text: it returns probabilities, in tens of
milliseconds on a laptop CPU and in a few hundred milliseconds on a Raspberry Pi. Everything runs locally.

## Setup

```sh
pip install "skejja[mcp]"
claude mcp add skejja -- skejja mcp        # downloads the default model on first use (about 208 MB)
```

`skejja mcp <folder>` uses a model archive that was extracted by hand instead.

## When to use it, and when not

Good fit:
- routing a short message (up to a few sentences) to one of 2–8 concrete actions;
- yes/no checks such as "Does the user want to delete files?" before acting;
- the same few questions asked many times, where per-question calibration pays off.

Poor fit, use your own judgement instead:
- reasoning, comparing two documents, entailment, arithmetic, world knowledge;
- inputs longer than about 200 tokens (it was trained on short inputs; longer ones are cut);
- more than about 10 options, or a catch-all option such as "anything else".

## How to ask

- **Options:** short English descriptions of concrete actions, e.g.
  `{"lights_on": "turn on the lights", "lights_off": "turn off the lights", "music": "play music"}`.
  The text itself can be in any of the five languages.
- **Yes/no:** phrase questions like the training data: `"Does the user want to <concrete action>?"`.
  Other phrasings can return values near 0.5 that carry no information.
- **No catch-all options.** Instead of an "other" option, ask one yes/no per action.

## Tools

| Tool | Use |
|---|---|
| `choose(text, options, instructions?)` | one Choice; returns `choice`, `probabilities`, `confidence` |
| `yes_no(text, question)` | probability of yes; `calibration` says `question` or `global` |
| `systemone(state, questions)` | several typed questions on the same state, `/v1/systemone` format (`choice`, `noul`, `score`) |
| `calibrate_question(question, examples)` | fit a yes/no question to 8–64 labelled examples `[{"state": ..., "answer": true}]` |

## Reading the answers

- Treat a low `confidence` on a Choice as "unsure" (pick the threshold on your own examples; 0.6 is a starting
  point, not a measured value): ask the user or fall back to your own reasoning.
- A `yes_no` with `calibration: "global"` on a question Skejja has never seen is **not calibrated**: use it to
  rank, not as a true probability, until you calibrate the question with examples in their real proportions
  (32–64 examples bring the calibration error down to about 0.04 on unseen intents).
- `truncated: true` means the text was cut: the answer may miss information.
- Never act irreversibly on a Skejja answer alone. It is a fast first filter, not a final judge.
