function linear(x, W, b, outDim, inDim) {
  const y = new Float32Array(outDim);
  for (let i = 0; i < outDim; i++) {
    let s = b[i];
    const rowOff = i * inDim;
    for (let j = 0; j < inDim; j++) s += W[rowOff + j] * x[j];
    y[i] = s;
  }
  return y;
}

function layerNorm(x, gamma, beta) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; variance += d * d; }
  variance /= n;
  const invStd = 1 / Math.sqrt(variance + 1e-5);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * invStd * gamma[i] + beta[i];
  return y;
}

function batchNormEval(x, gamma, beta, runningMean, runningVar) {
  const n = x.length;
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const invStd = 1 / Math.sqrt(runningVar[i] + 1e-5);
    y[i] = gamma[i] * (x[i] - runningMean[i]) * invStd + beta[i];
  }
  return y;
}

function gelu(x) {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    y[i] = 0.5 * x[i] * (1 + erf(x[i] / Math.SQRT2));
  }
  return y;
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function softmax(x) {
  let max = x[0];
  for (let i = 1; i < x.length; i++) if (x[i] > max) max = x[i];
  const y = new Float32Array(x.length);
  let sum = 0;
  for (let i = 0; i < x.length; i++) { y[i] = Math.exp(x[i] - max); sum += y[i]; }
  for (let i = 0; i < x.length; i++) y[i] /= sum;
  return y;
}

function normalizeFeatures(raw, breakpoints) {
  const result = new Float32Array(raw.length);
  for (let j = 0; j < raw.length; j++) {
    const bx = breakpoints[j].x, by = breakpoints[j].y;
    let x = raw[j];
    if (x <= bx[0]) { result[j] = by[0]; continue; }
    if (x >= bx[bx.length - 1]) { result[j] = by[by.length - 1]; continue; }
    let lo = 0, hi = bx.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (bx[mid] <= x) lo = mid; else hi = mid;
    }
    const dx = bx[hi] - bx[lo];
    result[j] = dx === 0 ? by[lo] : by[lo] + (by[hi] - by[lo]) * (x - bx[lo]) / dx;
  }
  return result;
}

function buildTensorAccessor(weights, metaTensors, perModelNumel, modelIdx) {
  const base = modelIdx * perModelNumel;
  const map = {};
  let offset = base;
  for (const t of metaTensors) {
    const view = new Float32Array(weights.buffer, weights.byteOffset + offset * 4, t.numel);
    map[t.name] = view;
    offset += t.numel;
  }
  return map;
}

function subsystemEncoder(x, W, subIdx) {
  const d = subIdx.length;
  const inp = new Float32Array(d);
  for (let i = 0; i < d; i++) inp[i] = x[subIdx[i]];

  let h = linear(inp, W[`gating.enc.${subIdx.encIdx}.net.0.weight`],
                 W[`gating.enc.${subIdx.encIdx}.net.0.bias`], 128, d);
  h = layerNorm(h, W[`gating.enc.${subIdx.encIdx}.net.1.weight`],
                W[`gating.enc.${subIdx.encIdx}.net.1.bias`]);
  h = gelu(h);
  h = linear(h, W[`gating.enc.${subIdx.encIdx}.net.4.weight`],
             W[`gating.enc.${subIdx.encIdx}.net.4.bias`], 64, 128);
  h = layerNorm(h, W[`gating.enc.${subIdx.encIdx}.net.5.weight`],
                W[`gating.enc.${subIdx.encIdx}.net.5.bias`]);
  h = gelu(h);
  return h;
}

function multiheadAttention(stack, W) {
  const seqLen = 5, dim = 64;
  const qkvW = W['gating.att.in_proj_weight'];
  const qkvB = W['gating.att.in_proj_bias'];

  const Q = new Float32Array(seqLen * dim);
  const K = new Float32Array(seqLen * dim);
  const V = new Float32Array(seqLen * dim);

  for (let s = 0; s < seqLen; s++) {
    const tok = stack.subarray(s * dim, (s + 1) * dim);
    for (let i = 0; i < dim; i++) {
      let sum = qkvB[i];
      const rowOff = i * dim;
      for (let j = 0; j < dim; j++) sum += qkvW[rowOff + j] * tok[j];
      Q[s * dim + i] = sum;
    }
    for (let i = 0; i < dim; i++) {
      let sum = qkvB[dim + i];
      const rowOff = (dim + i) * dim;
      for (let j = 0; j < dim; j++) sum += qkvW[rowOff + j] * tok[j];
      K[s * dim + i] = sum;
    }
    for (let i = 0; i < dim; i++) {
      let sum = qkvB[2 * dim + i];
      const rowOff = (2 * dim + i) * dim;
      for (let j = 0; j < dim; j++) sum += qkvW[rowOff + j] * tok[j];
      V[s * dim + i] = sum;
    }
  }

  const scale = 1 / Math.sqrt(dim);
  const scores = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      let s = 0;
      for (let k = 0; k < dim; k++) s += Q[i * dim + k] * K[j * dim + k];
      scores[i * seqLen + j] = s * scale;
    }
  }

  for (let i = 0; i < seqLen; i++) {
    const row = scores.subarray(i * seqLen, (i + 1) * seqLen);
    const sm = softmax(row);
    for (let j = 0; j < seqLen; j++) scores[i * seqLen + j] = sm[j];
  }

  const attnOut = new Float32Array(seqLen * dim);
  for (let i = 0; i < seqLen; i++) {
    for (let d = 0; d < dim; d++) {
      let s = 0;
      for (let j = 0; j < seqLen; j++) s += scores[i * seqLen + j] * V[j * dim + d];
      attnOut[i * dim + d] = s;
    }
  }

  const outW = W['gating.att.out_proj.weight'];
  const outB = W['gating.att.out_proj.bias'];
  const result = new Float32Array(seqLen * dim);
  for (let s = 0; s < seqLen; s++) {
    for (let i = 0; i < dim; i++) {
      let sum = outB[i];
      const rowOff = i * dim;
      for (let j = 0; j < dim; j++) sum += outW[rowOff + j] * attnOut[s * dim + j];
      result[s * dim + i] = sum;
    }
  }
  return result;
}

function cagGating(xn, W, subIndices) {
  const dim = 64, K = 5, seqLen = 5;
  const embs = [];
  for (let i = 0; i < subIndices.length; i++) {
    const subIdx = Object.assign([], subIndices[i]);
    subIdx.encIdx = i;
    embs.push(subsystemEncoder(xn, W, subIdx));
  }
  const stack = new Float32Array(seqLen * dim);
  for (let i = 0; i < seqLen; i++) {
    for (let d = 0; d < dim; d++) stack[i * dim + d] = embs[i][d];
  }

  const attnOut = multiheadAttention(stack, W);

  const post = new Float32Array(seqLen * dim);
  for (let i = 0; i < seqLen; i++) {
    const normIn = new Float32Array(dim);
    for (let d = 0; d < dim; d++) normIn[d] = attnOut[i * dim + d] + stack[i * dim + d];
    const normOut = layerNorm(normIn, W['gating.norm.weight'], W['gating.norm.bias']);
    for (let d = 0; d < dim; d++) post[i * dim + d] = normOut[d];
  }

  let flat = gelu(linear(post, W['gating.head.0.weight'], W['gating.head.0.bias'], 64, 320));
  let logits = linear(flat, W['gating.head.3.weight'], W['gating.head.3.bias'], K, 64);

  let t = Math.abs(W['gating.temp'][0]);
  if (t < 0.5) t = 0.5;
  if (t > 5.0) t = 5.0;
  const scaled = new Float32Array(K);
  for (let i = 0; i < K; i++) scaled[i] = logits[i] / t;
  return softmax(scaled);
}

function linearExpert(xn, W) {
  return linear(xn, W['experts.0.net.weight'], W['experts.0.net.bias'], 1, 31)[0];
}

function shallowExpert(xn, W, idx) {
  const p = `experts.${idx}.net`;
  let h = linear(xn, W[`${p}.0.weight`], W[`${p}.0.bias`], 512, 31);
  h = batchNormEval(h, W[`${p}.1.weight`], W[`${p}.1.bias`],
                    W[`${p}.1.running_mean`], W[`${p}.1.running_var`]);
  h = gelu(h);
  const out = linear(h, W[`${p}.4.weight`], W[`${p}.4.bias`], 1, 512);
  return out[0];
}

function residualExpert(xn, W, idx) {
  const p = `experts.${idx}`;
  let x1 = linear(xn, W[`${p}.input_proj.weight`], W[`${p}.input_proj.bias`], 256, 31);
  x1 = batchNormEval(x1, W[`${p}.bn0.weight`], W[`${p}.bn0.bias`],
                     W[`${p}.bn0.running_mean`], W[`${p}.bn0.running_var`]);
  x1 = gelu(x1);

  let main = linear(x1, W[`${p}.block.0.weight`], W[`${p}.block.0.bias`], 256, 256);
  main = batchNormEval(main, W[`${p}.block.1.weight`], W[`${p}.block.1.bias`],
                       W[`${p}.block.1.running_mean`], W[`${p}.block.1.running_var`]);
  main = gelu(main);
  main = linear(main, W[`${p}.block.4.weight`], W[`${p}.block.4.bias`], 64, 256);
  main = batchNormEval(main, W[`${p}.block.5.weight`], W[`${p}.block.5.bias`],
                       W[`${p}.block.5.running_mean`], W[`${p}.block.5.running_var`]);

  const skip = linear(x1, W[`${p}.proj.weight`], W[`${p}.proj.bias`], 64, 256);

  const combined = new Float32Array(64);
  for (let i = 0; i < 64; i++) combined[i] = main[i] + skip[i];
  const postBlock = gelu(combined);

  const postHeadGelu = gelu(postBlock);
  const out = linear(postHeadGelu, W[`${p}.head.2.weight`], W[`${p}.head.2.bias`], 1, 64);
  return out[0];
}

function forwardOneModel(xRaw, W, breakpoints, subIndices) {
  const xN = normalizeFeatures(xRaw, breakpoints);
  const xn = layerNorm(xN, W['input_norm.weight'], W['input_norm.bias']);
  const gate = cagGating(xN, W, subIndices);

  const preds = [
    linearExpert(xn, W),
    shallowExpert(xn, W, 1),
    shallowExpert(xn, W, 2),
    residualExpert(xn, W, 3),
    residualExpert(xn, W, 4),
  ];

  let moe = 0;
  for (let i = 0; i < 5; i++) moe += gate[i] * preds[i];

  const alpha = sigmoid(W['pirp.alpha'][0]);
  const yLin = linear(xn, W['pirp.linear.weight'], W['pirp.linear.bias'], 1, 31)[0];
  return alpha * yLin + (1 - alpha) * moe;
}

function buildFeatureVector(v) {
  const cement = v.cement, sf = v.sf, fa = v.fa, ggbfs = v.ggbfs;
  const qp = v.qp || 0, filler = 0, sand = v.sand;
  const fiber = v.fiber, fl = v.fl, fd = v.fd;
  const water = v.water, sp = v.sp, curing = v.curing;
  const tb = cement + sf + fa + ggbfs;
  const tb_safe = Math.max(tb, 1e-6);
  const wb = water / tb_safe;
  const far = fl / Math.max(fd, 1e-6);
  const fri = fiber * far / 100.0;
  const sfc = sf / Math.max(cement, 1e-6);
  const scm = (tb - cement) / tb_safe;
  const sbr = sand / tb_safe;
  const spb = sp / tb_safe;
  const hc = curing > 50 ? 1.0 : 0.0;
  const pv = (cement/3150 + sf/2200 + water/1000 + sp/1100) * 1000;
  const wcr = water / Math.max(cement, 1e-6);
  const fvf = fiber / 7850;
  const bi = tb / Math.max(tb + sand + water, 1e-6);
  return new Float32Array([
    cement, sf, fa, qp, ggbfs, filler, sand,
    fiber, fl, fd, water, sp, curing,
    tb, wb, far, fri, sfc, scm, sbr, spb, hc,
    0, 0, pv, wcr, fvf, bi,
    50.0, 52.5, 0.6
  ]);
}

function predictEnsemble(userInputs, meta, weightsBuffer) {
  const raw = buildFeatureVector(userInputs);
  const predictions = [];
  for (let i = 0; i < meta.num_models; i++) {
    const W = buildTensorAccessor(weightsBuffer, meta.tensors, meta.per_model_numel, i);
    const yN = forwardOneModel(raw, W, meta.breakpoints, meta.sub_indices);
    predictions.push(yN);
  }
  const avgN = predictions.reduce((a, b) => a + b, 0) / predictions.length;
  const strength = avgN * meta.y_scale + meta.y_mean;
  return { strength, normalized: avgN, perModel: predictions };
}

async function loadModel(metaUrl, weightsUrl) {
  const [metaResp, weightsResp] = await Promise.all([
    fetch(metaUrl).then(r => r.json()),
    fetch(weightsUrl).then(r => r.arrayBuffer()),
  ]);
  const weights = new Float32Array(weightsResp);
  return { meta: metaResp, weights };
}

window.CompMoE = { loadModel, predictEnsemble, buildFeatureVector, normalizeFeatures };
