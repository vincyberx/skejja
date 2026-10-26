---
title: Skejja
emoji: ⚡
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
license: apache-2.0
short_description: A 57M decision model running in your browser, 5 languages
models:
  - vincyber/skejja-edge32k-ft6
---

Demo of [Skejja](https://github.com/vincyberx/skejja) that runs entirely in the browser with onnxruntime-web:
Choice and yes/no decisions on a short sentence, nothing sent to a server. The model files in `model/` are the dynamic
int8 variant of [vincyber/skejja-edge32k-ft6](https://huggingface.co/vincyber/skejja-edge32k-ft6) (Apache-2.0,
attributions in that repository's NOTICE).
