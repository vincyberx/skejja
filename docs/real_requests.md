# Skejja on real requests to a personal voice assistant (October 2026)

Benchmarks like MASSIVE are close to what Skejja was trained on. To see how it behaves on real traffic, edge32k-ft6
(ONNX fp32) was run on requests that one person actually made to their own voice assistant, in Italian. **The
sentences are private and are not published**, so this test cannot be repeated by others; only aggregate metrics are
in the repository. Labels were written before running the model. Generic results:
`results/fase3/real_requests_*_ft6.csv` (built by `bench/real_requests_summary.py`).

## 1. Routing short requests to five destinations

236 requests from the assistant's history (18 to 220 characters) were labelled with five destinations: mail, calendar
and online services (34), web search (11), software development (17), images and video (3), and the assistant itself
for quick questions, commands and chat (102); 69 were not requests and were excluded. 43 labels are ambiguous and are
reported separately. File: `real_requests_routing_ft6.csv`.

| Choice among 5 destinations | accuracy | mean recall | ECE |
|---|---|---|---|
| English descriptions, all 167 requests | 0.491 | 0.508 | 0.120 |
| English descriptions, 124 unambiguous | 0.589 | 0.698 | 0.061 |
| Italian descriptions, 124 unambiguous | 0.460 | 0.527 | 0.083 |

On the unambiguous requests with English descriptions, recall is 0.78 for mail and online services, 0.86 for web
search, 0.57 for the assistant itself and 0.29 for software development. Two causes explain most errors: "the
assistant itself" is a catch-all option ("a question, a command, or chat"), and software-development requests are
recognised by the user's own jargon (project and session names), which the model does not know.

| Yes/no in English (167 requests) | AUROC | correct yes | wrong yes |
|---|---|---|---|
| does the user want to check mail, calendar or an online service? | 0.865 | 13 / 34 | 0 |
| does the user want to search the web? | 0.862 | 11 / 11 | 49 |
| does the user want to create an image, a chart or a video? | 0.866 | 2 / 3 | 9 |
| is it about a software project or a development session? | 0.617 | 2 / 17 | 31 |

The yes/no questions rank the requests well (AUROC about 0.86) but with the global calibration the 0.5 threshold is
off: too cautious on mail, too eager on web search. Images and video have only 3 positives.

## 2. Per-question calibration on those yes/no questions

For each yes/no question and k = 8, 16, 24 examples drawn from the same requests: per-question Platt calibration
(the API's own function), measured on the remaining requests against the global calibration, 50 draws per k.

- **Half yes, half no** (design fixed beforehand, `real_requests_calibration_balanced_ft6.csv`): it does not help.
  Real requests have few yes answers (20% for mail, 10% for software development), and a calibration fitted on 50/50
  examples says yes too often.
- **Random examples in their real proportions** (added after the first run,
  `real_requests_calibration_natural_ft6.csv`):

| 24 examples, global → per question | accuracy | Brier |
|---|---|---|
| mail and online services | 0.874 → 0.878 | 0.103 → 0.104 |
| web search | 0.706 → 0.814 | 0.214 → 0.111 |
| software development | 0.729 → 0.813 | 0.183 → 0.141 |

Calibration improves the probabilities (the Brier score for web search halves) but cannot fix a weak ranking:
software development (AUROC 0.617) gains only by learning to say no. **Give calibration examples in their real
proportions**, not balanced.

## 3. A long-context decision: should the assistant hand the turn to a larger model?

The assistant has a fast voice front end that decides, from its rules, a list of categories and the last turns of the
conversation, whether to pass a request to a larger model. On the same 235 cases used to choose that component,
with references from an LLM judge, Skejja reached AUROC 0.793 with Italian categories, 0.634 with English categories
and 0.596 with a single English yes/no (`real_requests_handoff_ft6.csv`). Every input exceeded 512 tokens and had to be
shortened; with 1,024 tokens it did worse (AUROC 0.697, `real_requests_handoff_ft6_1024.csv`), because the model was
trained on inputs up to 192 tokens. **Skejja is not suitable for this decision**; it is a nuanced one even for large
models.

## What follows

- **Where it works:** concrete, well-described requests in short sentences (mail, web search).
- **Where it does not:** nuanced decisions over long contexts, catch-all options, categories recognisable only from
  the user's jargon.
- **How to use it:** concrete English options, one yes/no per action instead of a generic "other", and per-question
  calibration with a few dozen real examples, in their real proportions, for the questions that matter.
