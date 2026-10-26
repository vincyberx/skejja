// Skejja nel browser: lo stesso calcolo di runtime/skejja_infer.py (Choice e sì/no), con onnxruntime-web (WASM) e
// @huggingface/tokenizers. Tokenizer verificato identico a quello Python su 6.003 frasi del benchmark.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";

const Q_RE = /^Does the user want to (.*)\?$/;

export class Skejja {
  static async load(base, modelFile, { threads = 1, onProgress } = {}) {
    ort.env.wasm.numThreads = threads;
    const get = async (f) => {
      const r = await fetch(`${base}/${f}`);
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r;
    };
    const [tj, tc, cal] = await Promise.all(
      ["tokenizer.json", "tokenizer_config.json", "calibration.json"].map(async (f) => (await get(f)).json()));
    const model = new Uint8Array(await (await get(modelFile)).arrayBuffer());
    const data = await readWithProgress(await get(`${modelFile}.data`), onProgress);
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
      externalData: [{ path: `${modelFile}.data`, data }],
    });
    return new Skejja(new Tokenizer(tj, tc), session, cal[modelFile], modelFile);
  }

  constructor(tokenizer, session, cal, modelFile) {
    Object.assign(this, { tokenizer, session, cal, modelFile });
  }

  async logits(text, labels) {
    const s = labels.map((l) => `<<LABEL>>${l}`).join("") + "<<SEP>>" + text;
    const ids = this.tokenizer.encode(s).ids;
    const n = ids.length;
    const out = await this.session.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]),
      label_slots: new ort.Tensor("float32", new Float32Array(labels.length), [labels.length]),
    });
    return { z: Array.from(out.logits.data).slice(0, labels.length), tokens: n };
  }

  async choice(text, options) {
    const { z, tokens } = await this.logits(text, options);
    const t = this.cal.temperature;
    const m = Math.max(...z.map((v) => v / t));
    const e = z.map((v) => Math.exp(v / t - m));
    const sum = e.reduce((a, b) => a + b, 0);
    const p = e.map((v) => v / sum);
    const n = p.length;
    return { probabilities: p, confidence: (Math.max(...p) - 1 / n) / (1 - 1 / n), tokens };
  }

  async yesNo(text, question) {
    const m = question.trim().match(Q_RE);
    const { z, tokens } = await this.logits(text, [m ? m[1] : question]);
    const s = this.cal.platt_a * z[0] + this.cal.platt_b;
    return { yes: 1 / (1 + Math.exp(-s)), tokens };
  }
}

async function readWithProgress(response, onProgress) {
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !onProgress) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
