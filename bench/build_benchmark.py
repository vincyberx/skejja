#!/usr/bin/env python3
"""Rebuild the Skejja phase-1 benchmark from MASSIVE (CC BY 4.0, via mteb/amazon_massive_intent).

Outputs (in --out, default bench/data):
  test_choice.jsonl / test_noul.jsonl     400 utterance ids x 5 languages (MASSIVE test split)
  calib_choice.jsonl / calib_noul.jsonl   300 ids x 5 languages (MASSIVE validation split) - calibration ONLY
  train_utterances.jsonl                  all MASSIVE train utterances for the 5 languages
  intent_descriptions.json                59 intents -> English option text ('general_quirky' excluded)

Choice: 8 options = gold + up to 3 same-scenario distractors + random others (same options in all 5 languages).
Noul:   'Does the user want to <description>?' ~50% positives; negatives are same-scenario with p=0.7.
Deterministic: seeds 1/2 (id sampling) and 11/12 (item construction).

    HF_HUB_DISABLE_XET=1 python build_benchmark.py --download   # fetch MASSIVE then build
"""
import argparse, collections, gzip, json, os
import numpy as np
import pandas as pd

LANGS = ["en", "it", "es", "de", "fr"]
DESC = {
    'alarm_query': 'check existing alarms', 'alarm_remove': 'delete an alarm', 'alarm_set': 'set an alarm',
    'audio_volume_down': 'turn the volume down', 'audio_volume_mute': 'mute the audio',
    'audio_volume_other': 'change other audio settings', 'audio_volume_up': 'turn the volume up',
    'calendar_query': 'check the calendar', 'calendar_remove': 'delete a calendar event',
    'calendar_set': 'add a calendar event or reminder',
    'cooking_query': 'ask a cooking question', 'cooking_recipe': 'get a recipe',
    'datetime_convert': 'convert time zones or dates', 'datetime_query': 'ask the current date or time',
    'email_addcontact': 'add an email contact', 'email_query': 'check emails', 'email_querycontact': 'look up a contact',
    'email_sendemail': 'send an email',
    'general_greet': 'greet the assistant', 'general_joke': 'hear a joke',
    'iot_cleaning': 'start the vacuum cleaner', 'iot_coffee': 'make coffee', 'iot_hue_lightchange': 'change the light color',
    'iot_hue_lightdim': 'dim the lights', 'iot_hue_lightoff': 'turn off the lights', 'iot_hue_lighton': 'turn on the lights',
    'iot_hue_lightup': 'make the lights brighter', 'iot_wemo_off': 'turn off a smart plug', 'iot_wemo_on': 'turn on a smart plug',
    'lists_createoradd': 'create a list or add an item to a list', 'lists_query': 'read a list',
    'lists_remove': 'remove a list or a list item',
    'music_dislikeness': 'say they dislike a song', 'music_likeness': 'say they like a song',
    'music_query': 'ask what music is playing', 'music_settings': 'change music playback settings',
    'news_query': 'get the news', 'play_audiobook': 'play an audiobook', 'play_game': 'play a game', 'play_music': 'play music',
    'play_podcasts': 'play a podcast', 'play_radio': 'play the radio',
    'qa_currency': 'ask about exchange rates', 'qa_definition': 'ask for the definition of a word',
    'qa_factoid': 'ask a general knowledge question', 'qa_maths': 'solve a math problem', 'qa_stock': 'ask about stock prices',
    'recommendation_events': 'get event recommendations', 'recommendation_locations': 'get place recommendations',
    'recommendation_movies': 'get movie recommendations',
    'social_post': 'post on social media', 'social_query': 'check social media updates',
    'takeaway_order': 'order food delivery', 'takeaway_query': 'ask about a food delivery order',
    'transport_query': 'ask about public transport', 'transport_taxi': 'book a taxi',
    'transport_ticket': 'book a train or plane ticket', 'transport_traffic': 'check the traffic',
    'weather_query': 'ask about the weather'}


def download(raw):
    from huggingface_hub import hf_hub_download
    for split in ["train", "validation", "test"]:
        for l in LANGS:
            hf_hub_download("mteb/amazon_massive_intent", f"{split}/{l}.json.gz", repo_type="dataset", local_dir=raw)


def load(raw):
    dfs = []
    for split in ["train", "validation", "test"]:
        for l in LANGS:
            d = pd.DataFrame([json.loads(x) for x in gzip.open(f"{raw}/{split}/{l}.json.gz", "rt")])
            d["split"] = split; dfs.append(d)
    M = pd.concat(dfs, ignore_index=True)
    return M[M.label_text != "general_quirky"].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/massive")
    ap.add_argument("--out", default="bench/data")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download:
        download(a.raw)
    M = load(a.raw)
    assert set(M.label_text.unique()) == set(DESC)
    INTENTS = sorted(DESC); scen = lambda i: i.split("_")[0]
    BY = collections.defaultdict(list)
    for i in INTENTS:
        BY[scen(i)].append(i)

    def sample_ids(split, n, seed):
        d = M[(M.split == split) & (M.lang == "en")]
        rng = np.random.default_rng(seed); per = max(1, n // len(INTENTS)); ids = []
        for _, g in d.groupby("label_text"):
            ids += list(rng.choice(g.id.values, size=min(per, len(g)), replace=False))
        rest = d[~d.id.isin(ids)]
        ids += list(rng.choice(rest.id.values, size=n - len(ids), replace=False))
        return sorted(ids, key=int)

    def build(split, ids, seed, K=8):
        rng = np.random.default_rng(seed); ch, no = [], []
        d = M[M.split == split].set_index(["id", "lang"])
        for uid in ids:
            gold = d.loc[(uid, "en"), "label_text"]
            same = [i for i in BY[scen(gold)] if i != gold]; other = [i for i in INTENTS if scen(i) != scen(gold)]
            hard = list(rng.choice(same, size=min(3, len(same)), replace=False)) if same else []
            opts = [gold] + hard + list(rng.choice(other, size=K - 1 - len(hard), replace=False)); rng.shuffle(opts)
            opts = [str(o) for o in opts]
            pos = rng.random() < 0.5
            neg = rng.choice(same) if (same and rng.random() < 0.7) else rng.choice(other)
            cand = str(gold if pos else neg)
            for l in LANGS:
                t = d.loc[(uid, l), "text"]
                ch.append(dict(id=uid, lang=l, text=t, options=opts, option_desc=[DESC[o] for o in opts], gold=gold,
                               gold_idx=int(opts.index(gold))))
                no.append(dict(id=uid, lang=l, text=t, intent=cand, question=f"Does the user want to {DESC[cand]}?",
                               label=int(pos)))
        return pd.DataFrame(ch), pd.DataFrame(no)

    os.makedirs(a.out, exist_ok=True)
    for name, (split, ids, seed) in {"test": ("test", sample_ids("test", 400, 1), 11),
                                     "calib": ("validation", sample_ids("validation", 300, 2), 12)}.items():
        c, n = build(split, ids, seed)
        c.to_json(f"{a.out}/{name}_choice.jsonl", orient="records", lines=True, force_ascii=False)
        n.to_json(f"{a.out}/{name}_noul.jsonl", orient="records", lines=True, force_ascii=False)
        print(name, len(c), len(n))
    tr = M[M.split == "train"][["id", "lang", "text", "label_text"]].rename(columns={"label_text": "intent"})
    tr.to_json(f"{a.out}/train_utterances.jsonl", orient="records", lines=True, force_ascii=False)
    json.dump(DESC, open(f"{a.out}/intent_descriptions.json", "w"), indent=1)
    print("train", len(tr))


if __name__ == "__main__":
    main()
