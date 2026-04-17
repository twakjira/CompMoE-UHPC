function topsis(solutions, weights) {
  var n = solutions.length, m = 3;
  var raw = new Array(n);
  for (var i = 0; i < n; i++) {
    raw[i] = [solutions[i].strength, solutions[i].co2, solutions[i].cost];
  }
  var norm = new Array(n);
  var colSum = [0, 0, 0];
  for (var i = 0; i < n; i++)
    for (var j = 0; j < m; j++) colSum[j] += raw[i][j] * raw[i][j];
  for (var j = 0; j < m; j++) colSum[j] = Math.sqrt(colSum[j]);
  for (var i = 0; i < n; i++) {
    norm[i] = new Array(m);
    for (var j = 0; j < m; j++) norm[i][j] = raw[i][j] / colSum[j];
  }
  var wNorm = new Array(n);
  for (var i = 0; i < n; i++) {
    wNorm[i] = new Array(m);
    for (var j = 0; j < m; j++) wNorm[i][j] = norm[i][j] * weights[j];
  }
  var aPlus = [wNorm[0][0], wNorm[0][1], wNorm[0][2]];
  var aMinus = [wNorm[0][0], wNorm[0][1], wNorm[0][2]];
  for (var i = 1; i < n; i++) {
    if (wNorm[i][0] > aPlus[0]) aPlus[0] = wNorm[i][0];
    if (wNorm[i][0] < aMinus[0]) aMinus[0] = wNorm[i][0];
    if (wNorm[i][1] < aPlus[1]) aPlus[1] = wNorm[i][1];
    if (wNorm[i][1] > aMinus[1]) aMinus[1] = wNorm[i][1];
    if (wNorm[i][2] < aPlus[2]) aPlus[2] = wNorm[i][2];
    if (wNorm[i][2] > aMinus[2]) aMinus[2] = wNorm[i][2];
  }
  var scores = new Array(n);
  for (var i = 0; i < n; i++) {
    var dPlus = 0, dMinus = 0;
    for (var j = 0; j < m; j++) {
      dPlus += (wNorm[i][j] - aPlus[j]) * (wNorm[i][j] - aPlus[j]);
      dMinus += (wNorm[i][j] - aMinus[j]) * (wNorm[i][j] - aMinus[j]);
    }
    dPlus = Math.sqrt(dPlus);
    dMinus = Math.sqrt(dMinus);
    scores[i] = { idx: i, score: dMinus / (dPlus + dMinus + 1e-12) };
  }
  scores.sort(function(a, b) { return b.score - a.score; });
  return scores;
}

function findInverse(solutions, targetStrength) {
  var candidates = [];
  for (var i = 0; i < solutions.length; i++) {
    if (solutions[i].strength >= targetStrength) {
      candidates.push(solutions[i]);
    }
  }
  if (candidates.length === 0) return null;
  candidates.sort(function(a, b) {
    if (a.co2 !== b.co2) return a.co2 - b.co2;
    return a.cost - b.cost;
  });
  return candidates[0];
}

window.MixOptimizer = { topsis: topsis, findInverse: findInverse };
