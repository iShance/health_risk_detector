// App Logic — Health Risk Detector
// Depends on: models.js, visualization.js

let smokingOn = false;
let predictionHistory = [];
let liveTimer = null;

const SLIDER_CFG = {
  hr:   { min: 40,  max: 180, safeMin: 60,  safeMax: 100, type: 'range' },
  bp:   { min: 80,  max: 200, safeMin: 90,  safeMax: 130, type: 'range' },
  act:  { min: 0,   max: 10,  safeMin: 4,   safeMax: 10,  type: 'inverted' },
  age:  { min: 20,  max: 90,  safeMin: 20,  safeMax: 55,  type: 'age' },
  chol: { min: 100, max: 320, safeMin: 100, safeMax: 200, type: 'range' },
};

function updateSlider(which) {
  const cfg = SLIDER_CFG[which];
  const v   = parseFloat(document.getElementById(`${which}-slider`).value);
  document.getElementById(`${which}-val`).textContent = v;

  const pct = (v - cfg.min) / (cfg.max - cfg.min) * 100;
  let color;
  if      (cfg.type === 'inverted') color = v < 4 ? 'var(--danger)' : v < 7 ? 'var(--moderate)' : 'var(--safe)';
  else if (cfg.type === 'age')      color = v > 65 ? 'var(--danger)' : v > 50 ? 'var(--moderate)' : 'var(--safe)';
  else {
    if      (v > cfg.safeMax) color = 'var(--danger)';
    else if (v < cfg.safeMin) color = 'var(--moderate)';
    else                      color = 'var(--safe)';
  }
  document.getElementById(`${which}-bar`).style.background =
    `linear-gradient(90deg, ${color} ${pct}%, rgba(255,255,255,0.05) ${pct}%)`;
}

function getInputs() {
  return {
    hr:      parseFloat(document.getElementById('hr-slider').value),
    bp:      parseFloat(document.getElementById('bp-slider').value),
    act:     parseFloat(document.getElementById('act-slider').value),
    age:     parseFloat(document.getElementById('age-slider').value),
    chol:    parseFloat(document.getElementById('chol-slider').value),
    smoking: smokingOn,
  };
}

function onSliderChange(which) {
  updateSlider(which);
  clearTimeout(liveTimer);
  liveTimer = setTimeout(updateLiveChips, 280);
}

function updateLiveChips() {
  const { hr, bp, act, age, chol, smoking } = getInputs();
  const scores = {
    lr:  logisticRegPredict(hr, bp, act, age, chol, smoking),
    knn: knnPredict(hr, bp, act, age, chol, smoking),
    dt:  decisionTreePredict(hr, bp, act, age, chol, smoking),
    rf:  randomForestPredict(hr, bp, act, age, chol, smoking),
  };
  Object.entries(scores).forEach(([id, score]) => {
    const level = getRiskLevel(score);
    const pred  = document.getElementById(`${id}-pred`);
    const conf  = document.getElementById(`${id}-conf`);
    pred.textContent = getLabelText(level);
    pred.style.color = getColor(level);
    conf.textContent = Math.round(score * 100) + '%';
  });
}

function toggleSmoking() {
  smokingOn = !smokingOn;
  const btn = document.getElementById('smoking-btn');
  const lbl = document.getElementById('smoking-label');
  const ico = document.getElementById('smoking-icon');
  if (smokingOn) {
    btn.classList.add('active');
    lbl.textContent = 'Smoker';
    ico.textContent = '🚬';
  } else {
    btn.classList.remove('active');
    lbl.textContent = 'Non-Smoker';
    ico.textContent = '🚭';
  }
  clearTimeout(liveTimer);
  liveTimer = setTimeout(updateLiveChips, 100);
}

function runPrediction() {
  const { hr, bp, act, age, chol, smoking } = getInputs();
  const scores = {
    lr:  logisticRegPredict(hr, bp, act, age, chol, smoking),
    knn: knnPredict(hr, bp, act, age, chol, smoking),
    dt:  decisionTreePredict(hr, bp, act, age, chol, smoking),
    rf:  randomForestPredict(hr, bp, act, age, chol, smoking),
  };
  const avgScore = (scores.lr + scores.knn + scores.dt + scores.rf) / 4;
  const level    = getRiskLevel(avgScore);

  // Update chips
  Object.entries(scores).forEach(([id, score]) => {
    const lvl  = getRiskLevel(score);
    document.getElementById(`${id}-pred`).textContent = getLabelText(lvl);
    document.getElementById(`${id}-pred`).style.color  = getColor(lvl);
    document.getElementById(`${id}-conf`).textContent  = Math.round(score * 100) + '%';
  });

  // Show result card
  const card = document.getElementById('result-card');
  card.className    = `result-card ${level}`;
  card.style.display = 'block';

  const icons  = { safe: '✓', moderate: '△', danger: '⚠' };
  const titles = {
    safe:     'LOW RISK — Indicators look healthy',
    moderate: 'MODERATE RISK — Monitor closely',
    danger:   'HIGH RISK — Seek medical attention',
  };
  document.getElementById('result-icon').textContent       = icons[level];
  document.getElementById('result-title').textContent      = titles[level];
  document.getElementById('result-score-line').textContent = `Ensemble score: ${(avgScore * 100).toFixed(1)}%`;

  // Animate bars
  setTimeout(() => {
    ['lr','knn','dt','rf'].forEach(id => {
      const pct = Math.round(scores[id] * 100);
      document.getElementById(`${id}-bar`).style.width    = pct + '%';
      document.getElementById(`${id}-pct`).textContent    = pct + '%';
    });
    document.getElementById('gauge-needle').style.left = (avgScore * 100).toFixed(1) + '%';
  }, 60);

  // Warnings
  const warnings = buildWarnings(hr, bp, act, age, chol, smoking);
  document.getElementById('warnings-list').innerHTML = warnings.map(w =>
    `<div class="warning-item"><div class="warning-dot ${w.cls}"></div><span>${w.text}</span></div>`
  ).join('');

  // Recommendations
  const recs    = buildRecommendations(hr, bp, act, age, chol, smoking, level);
  const recsEl  = document.getElementById('recommendations');
  recsEl.innerHTML = recs.length
    ? `<div class="section-title" style="margin-top:1.2rem">Recommendations</div>
       ${recs.map(r => `<div class="rec-item"><span class="rec-icon">${r.icon}</span><span>${r.text}</span></div>`).join('')}`
    : '';

  // History
  predictionHistory.unshift({ hr, bp, act, age, chol, smoking, scores, avgScore, level, ts: new Date() });
  if (predictionHistory.length > 10) predictionHistory.pop();
}

function buildWarnings(hr, bp, act, age, chol, smoking) {
  const w = [];
  if      (bp   > 140) w.push({ text: 'High blood pressure detected (BP > 140 mmHg)',           cls: 'dot-danger'   });
  else if (bp   > 130) w.push({ text: 'Elevated blood pressure — borderline high',               cls: 'dot-moderate' });
  if      (hr   > 100) w.push({ text: 'Elevated heart rate — tachycardia risk',                  cls: 'dot-danger'   });
  else if (hr   >  90) w.push({ text: 'Heart rate slightly above optimal range',                 cls: 'dot-moderate' });
  if      (act  <   3) w.push({ text: 'Very low activity increases cardiovascular risk',         cls: 'dot-danger'   });
  else if (act  <   5) w.push({ text: 'Moderate activity — more exercise may reduce risk',       cls: 'dot-moderate' });
  if      (chol > 240) w.push({ text: 'High cholesterol (> 240 mg/dL) — significant risk factor', cls: 'dot-danger' });
  else if (chol > 200) w.push({ text: 'Borderline high cholesterol (200–240 mg/dL)',             cls: 'dot-moderate' });
  if (smoking)         w.push({ text: 'Smoking is a major cardiovascular risk factor',           cls: 'dot-danger'   });
  if (age       >  60) w.push({ text: 'Age over 60 raises baseline cardiovascular risk',         cls: 'dot-moderate' });
  if (w.length  ===  0) w.push({ text: 'All indicators within healthy range — great work!',      cls: 'dot-safe'     });
  return w;
}

function buildRecommendations(hr, bp, act, age, chol, smoking, level) {
  if (level === 'safe') return [];
  const r = [];
  if (bp > 130)              r.push({ icon: '💊', text: 'Discuss blood pressure management with your doctor. Lifestyle changes or medication may help.' });
  if (act < 5)               r.push({ icon: '🏃', text: 'Aim for 150+ min/week of moderate aerobic exercise — it directly lowers BP and resting HR.' });
  if (chol > 200)            r.push({ icon: '🥗', text: 'Reduce saturated fats and increase fiber. Ask your doctor for a full lipid panel.' });
  if (smoking)               r.push({ icon: '🚭', text: 'Quitting smoking can cut cardiovascular risk by up to 50% within one year.' });
  if (hr > 90)               r.push({ icon: '🧘', text: 'Practice stress reduction techniques — chronic stress elevates resting heart rate.' });
  if (age > 50 && level !== 'safe') r.push({ icon: '🩺', text: 'Schedule a cardiovascular screening with your physician — age is a key risk factor.' });
  return r;
}

function renderHistory() {
  const container = document.getElementById('history-list');
  if (predictionHistory.length === 0) {
    container.innerHTML = '<p class="no-history">No predictions yet. Run an analysis first.</p>';
    return;
  }
  const icons  = { safe: '✓', moderate: '△', danger: '⚠' };
  const labels = { safe: 'Low Risk', moderate: 'Moderate', danger: 'High Risk' };
  container.innerHTML = predictionHistory.map(entry => `
    <div class="history-item history-${entry.level}">
      <div class="history-icon history-icon-${entry.level}">${icons[entry.level]}</div>
      <div class="history-body">
        <div class="history-title">
          ${labels[entry.level]}
          <span class="history-score">${(entry.avgScore * 100).toFixed(0)}%</span>
        </div>
        <div class="history-meta">
          HR ${entry.hr} bpm · BP ${entry.bp} mmHg · Act ${entry.act}/10 · Age ${entry.age} · Chol ${entry.chol}${entry.smoking ? ' · Smoker' : ''}
        </div>
        <div class="history-time">${entry.ts.toLocaleDateString()} at ${entry.ts.toLocaleTimeString()}</div>
      </div>
    </div>
  `).join('');
}

function clearHistory() {
  predictionHistory = [];
  renderHistory();
}

function switchTab(tab, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  ['predict', 'viz', 'history', 'edu'].forEach(t =>
    document.getElementById(`tab-${t}`).classList.toggle('hidden', t !== tab)
  );
  if (tab === 'viz')     setTimeout(drawBoundary, 50);
  if (tab === 'history') renderHistory();
}

function exportResult() {
  const { hr, bp, act, age, chol, smoking } = getInputs();
  const lrS  = logisticRegPredict(hr, bp, act, age, chol, smoking);
  const knnS = knnPredict(hr, bp, act, age, chol, smoking);
  const dtS  = decisionTreePredict(hr, bp, act, age, chol, smoking);
  const rfS  = randomForestPredict(hr, bp, act, age, chol, smoking);
  const avg  = (lrS + knnS + dtS + rfS) / 4;
  const level = getRiskLevel(avg);

  const text = [
    '=== HEALTH RISK ASSESSMENT REPORT ===',
    `Generated: ${new Date().toLocaleString()}`,
    '',
    '-- PATIENT INPUTS --',
    `Heart Rate:      ${hr} bpm`,
    `Blood Pressure:  ${bp} mmHg (systolic)`,
    `Activity Level:  ${act} / 10`,
    `Age:             ${age} years`,
    `Cholesterol:     ${chol} mg/dL`,
    `Smoking:         ${smoking ? 'Yes' : 'No'}`,
    '',
    '-- MODEL PREDICTIONS --',
    `Logistic Regression: ${Math.round(lrS  * 100)}%`,
    `KNN (k=7, weighted): ${Math.round(knnS * 100)}%`,
    `Decision Tree:       ${Math.round(dtS  * 100)}%`,
    `Random Forest (x5):  ${Math.round(rfS  * 100)}%`,
    `Ensemble Score:      ${(avg * 100).toFixed(1)}%`,
    `Risk Level:          ${level.toUpperCase()}`,
    '',
    '-- DISCLAIMER --',
    'Simulated educational tool. Not a substitute for professional medical advice.',
    'Always consult a qualified healthcare provider.',
  ].join('\n');

  const blob = new Blob([text], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `health_risk_${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

// Initialise sliders on load
Object.keys(SLIDER_CFG).forEach(updateSlider);
