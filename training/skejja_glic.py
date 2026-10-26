"""GLiClass uni-encoder helpers for Skejja: input formatting identical to gliclass UniEncoder pipeline
(prompt_first=True: '<<LABEL>>opt1<<LABEL>>opt2...<<SEP>>text'), batched scoring, ONNX-exportable wrapper."""
import re
import numpy as np
import torch
import torch.nn as nn

Q_RE = re.compile(r"^Does the user want to (.*)\?$")


def noul_label(question):
    """Noul question 'Does the user want to X?' -> label text 'X' (fed as the single GLiClass label)."""
    m = Q_RE.match(question.strip())
    return m.group(1) if m else question


def fmt(text, labels):
    return "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>" + text


def choice_inputs(rows):
    return [fmt(r["text"], r["option_desc"]) for r in rows]


def noul_inputs(rows):
    return [fmt(r["text"], [noul_label(r["question"])]) for r in rows]


class Scorer(nn.Module):
    """Clean re-implementation of GLiClassUniEncoder inference (class_token_pooling='average',
    embed_class_token=True, extract_text_features=True, pooling 'pass', cross-attn scorer, normalize_features=False).
    K (number of labels) is taken from the SHAPE of `label_slots` so one ONNX graph serves Choice (K=8) and Noul (K=1)."""

    def __init__(self, glimodel):
        super().__init__()
        m = glimodel.model
        self.enc = m.encoder_model
        self.text_projector = m.text_projector
        self.classes_projector = m.classes_projector
        self.scorer = m.scorer
        self.cls_id = int(glimodel.config.class_token_index)
        self.sep_id = int(glimodel.config.text_token_index)
        assert getattr(glimodel.config, "class_token_pooling", "first") == "average"
        assert glimodel.config.embed_class_token and glimodel.config.extract_text_features
        assert not glimodel.config.normalize_features and glimodel.config.pooling_strategy == "pass"

    def forward(self, input_ids, attention_mask, label_slots):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B,S,D)
        B, S, D = h.shape
        K = label_slots.shape[0]
        is_cls = (input_ids == self.cls_id)
        is_sep = (input_ids == self.sep_id)
        class_cum = is_cls.long().cumsum(-1)                    # (B,S)
        text_cum = is_sep.long().cumsum(-1)
        k_range = torch.arange(K, device=h.device).view(1, -1, 1) + 1
        span = (class_cum.unsqueeze(1) == k_range) & (text_cum.unsqueeze(1) == 0) & attention_mask.unsqueeze(1).bool()
        span_f = span.to(h.dtype)
        cnt = span_f.sum(-1, keepdim=True).clamp(min=1.0)
        cls_emb = torch.einsum("bks,bsd->bkd", span_f, h) / cnt    # (B,K,D)
        # text tokens: from <<SEP>> (inclusive) to end, shifted to start at 0 (same as gliclass)
        sep_idx = is_sep.int().argmax(-1)                           # (B,)
        ar = torch.arange(S, device=h.device).unsqueeze(0).expand(B, -1)
        valid = ar < (S - sep_idx).unsqueeze(1)
        src = (sep_idx.unsqueeze(1) + ar).clamp(max=S - 1)
        bidx = torch.arange(B, device=h.device).unsqueeze(1).expand(-1, S)
        txt = h[bidx, src] * valid.unsqueeze(-1).to(h.dtype)
        tmask = attention_mask[bidx, src] * valid.to(attention_mask.dtype)
        txt = self.text_projector(txt)
        cls_emb = self.classes_projector(cls_emb)
        return self.scorer(txt, cls_emb, text_mask=tmask)          # (B,K)


@torch.no_grad()
def score_torch(model, tok, inputs, K, batch_size=64, max_length=512, use_scorer=None):
    """Batched logits (N,K). model: GLiClassModel (uses gliclass forward) or Scorer (use_scorer=True)."""
    out = []
    order = np.argsort([len(s) for s in inputs])  # length bucketing for speed; restored below
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    sorted_inputs = [inputs[i] for i in order]
    for i in range(0, len(sorted_inputs), batch_size):
        enc = tok(sorted_inputs[i:i + batch_size], truncation=True, max_length=max_length, padding="longest",
                  return_tensors="pt")
        dev = next(model.parameters()).device
        enc = {k_: v_.to(dev) for k_, v_ in enc.items()}
        if use_scorer:
            lg = model(enc["input_ids"], enc["attention_mask"], torch.zeros(K))
        else:
            lg = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_num_classes=K).logits
        out.append(lg[:, :K].float().cpu().numpy())
    return np.concatenate(out)[inv]
