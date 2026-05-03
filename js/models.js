const TRAINING_DATA = [
  // --- Safe (0) ---
  [60,110,8,30,170,0,0],[65,115,7,35,180,0,0],[70,120,6,28,175,0,0],
  [55,118,9,25,160,0,0],[72,122,7,32,185,0,0],[58,112,8,29,172,0,0],
  [68,119,6,40,190,0,0],[75,125,5,38,195,0,0],[62,116,7,45,200,0,0],
  [80,130,4,50,195,0,0],[76,126,5,42,188,0,0],[78,128,5,36,182,0,0],
  [84,134,4,44,210,0,0],[82,132,4,41,205,0,0],[73,123,6,33,186,0,0],
  [69,121,7,31,183,0,0],[77,127,5,39,193,0,0],[66,117,8,27,174,0,0],
  [64,114,8,26,171,0,0],[61,113,9,24,168,0,0],[57,111,9,24,165,0,0],
  [59,113,8,26,169,0,0],[74,124,6,34,187,0,0],[71,122,7,30,184,0,0],
  [67,118,8,28,173,0,0],[63,115,8,27,171,0,0],
  // --- At Risk (1) ---
  [110,160,2,55,250,1,1],[120,170,1,60,260,1,1],[130,180,1,65,270,1,1],
  [105,155,2,58,245,0,1],[125,175,1,62,265,1,1],[115,165,2,57,255,1,1],
  [118,168,1,63,258,1,1],[108,158,2,56,248,0,1],[112,162,1,59,252,1,1],
  [122,172,1,64,262,1,1],[90,140,3,50,220,1,1],[95,145,3,52,225,1,1],
  [88,138,3,48,218,0,1],[92,142,3,51,222,1,1],[85,135,4,46,215,1,1],
  [100,150,2,55,230,1,1],[87,137,3,47,216,1,1],[93,143,3,53,226,1,1],
  [97,147,2,54,228,1,1],[103,152,2,56,232,1,1],[107,157,2,58,242,1,1],
  [83,133,4,43,208,1,1],[86,136,3,45,213,1,1],[89,139,3,49,219,1,1],
  [94,144,2,52,223,0,1],[96,146,2,53,227,1,1],[99,149,2,54,229,1,1],
  [101,151,2,55,231,1,1],[104,153,2,57,233,1,1],[106,156,2,58,240,1,1],
];

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function normHR(v)   { return (v - 40)  / 140; }
function normBP(v)   { return (v - 80)  / 120; }
function normAct(v)  { return v          / 10; }
function normAge(v)  { return (v - 20)  / 70; }
function normChol(v) { return (v - 100) / 200; }

function logisticRegPredict(hr, bp, act, age, chol, smoking) {
  const z = -5.0
    + 4.2  * normBP(bp)
    + 2.5  * normHR(hr)
    - 2.1  * normAct(act)
    + 1.8  * normAge(age)
    + 2.0  * normChol(chol)
    + (smoking ? 1.1 : 0)
    + 1.4  * normBP(bp) * normHR(hr)
    + 0.8  * normAge(age) * normChol(chol);
  return Math.min(0.99, Math.max(0.01, sigmoid(z)));
}

function knnPredict(hr, bp, act, age, chol, smoking, k = 7) {
  const nHR = normHR(hr), nBP = normBP(bp), nAct = normAct(act);
  const nAge = normAge(age), nChol = normChol(chol), nSmoke = smoking ? 1 : 0;

  const dists = TRAINING_DATA.map(([thr, tbp, tact, tage, tchol, tsmoke, label]) => ({
    d: Math.sqrt(
      1.5 * (nBP  - normBP(tbp))   ** 2 +
      1.0 * (nHR  - normHR(thr))   ** 2 +
      0.8 * (nAct - normAct(tact)) ** 2 +
      1.0 * (nAge - normAge(tage)) ** 2 +
      1.2 * (nChol- normChol(tchol))** 2 +
      0.6 * (nSmoke - tsmoke)      ** 2
    ),
    label,
  }))
  .sort((a, b) => a.d - b.d)
  .slice(0, k);

  let wRisk = 0, wTotal = 0;
  dists.forEach(({ d, label }) => {
    const w = 1 / (d + 0.001);
    wRisk  += w * label;
    wTotal += w;
  });
  return Math.min(0.99, Math.max(0.01, wRisk / wTotal));
}

function decisionTreePredict(hr, bp, act, age, chol, smoking) {
  if (bp > 145) return 0.94;
  if (bp > 130) {
    if (hr > 100)               return 0.90;
    if (smoking && age > 50)    return 0.84;
    if (chol > 240)             return 0.80;
    if (act < 3)                return 0.76;
    return 0.56;
  }
  if (chol > 240) {
    if (age > 55)    return 0.75;
    if (smoking)     return 0.70;
    return 0.52;
  }
  if (hr > 105) {
    if (act < 4)   return 0.82;
    if (smoking)   return 0.65;
    return 0.46;
  }
  if (smoking && age > 50 && act < 4) return 0.72;
  if (act < 3)                        return 0.66;
  if (bp > 120 && hr > 90)            return 0.40;
  if (age > 60 && chol > 220)         return 0.48;
  return 0.08;
}

// Random Forest: ensemble of 5 trees with bootstrapped splits
const RF_CONFIGS = [
  { bpT: 140, hrT: 100, cholT: 235, ageT: 50 },
  { bpT: 142, hrT: 105, cholT: 240, ageT: 52 },
  { bpT: 138, hrT: 102, cholT: 245, ageT: 48 },
  { bpT: 145, hrT:  98, cholT: 230, ageT: 55 },
  { bpT: 136, hrT: 108, cholT: 250, ageT: 45 },
];

function rfSingleTree(hr, bp, act, age, chol, smoking, cfg) {
  const { bpT, hrT, cholT, ageT } = cfg;
  if (bp > bpT + 10)   return 0.93;
  if (chol > cholT + 20) return 0.86;
  if (bp > bpT) {
    if (hr > hrT)                  return 0.87;
    if (smoking && age > ageT)     return 0.79;
    return 0.55;
  }
  if (hr > hrT) {
    if (act < 4)   return 0.80;
    if (smoking)   return 0.62;
    return 0.42;
  }
  if (smoking && age > ageT && act < 4)  return 0.70;
  if (chol > cholT) return age > ageT ? 0.67 : 0.48;
  if (act < 3)    return 0.63;
  return 0.08;
}

function randomForestPredict(hr, bp, act, age, chol, smoking) {
  const avg = RF_CONFIGS.reduce((s, cfg) => s + rfSingleTree(hr, bp, act, age, chol, smoking, cfg), 0) / RF_CONFIGS.length;
  return Math.min(0.99, Math.max(0.01, avg));
}

function getRiskLevel(score) {
  if (score < 0.3) return 'safe';
  if (score < 0.7) return 'moderate';
  return 'danger';
}

function getColor(level) {
  return level === 'safe' ? 'var(--safe)' : level === 'moderate' ? 'var(--moderate)' : 'var(--danger)';
}

function getLabelText(level) {
  return level === 'safe' ? 'Safe' : level === 'moderate' ? 'Moderate' : 'At Risk';
}
