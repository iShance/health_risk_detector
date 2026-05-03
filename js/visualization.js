// Visualization — Health Risk Detector
// Depends on: models.js (TRAINING_DATA, all predict functions)

function drawBoundary() {
  const canvas = document.getElementById('boundaryCanvas');
  if (!canvas) return;

  const modelSel   = document.getElementById('viz-model');
  const xSel       = document.getElementById('viz-xfeature');
  const model      = modelSel ? modelSel.value : 'lr';
  const xFeature   = xSel    ? xSel.value     : 'bp';

  const W = canvas.clientWidth || 640;
  const H = 310;
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  const PAD = { top: 12, right: 24, bottom: 48, left: 46 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  const xRanges = {
    bp:   { min: 80,  max: 200, label: 'Blood Pressure (mmHg)' },
    chol: { min: 100, max: 320, label: 'Cholesterol (mg/dL)' },
    age:  { min: 20,  max: 90,  label: 'Age (years)' },
  };
  const yRange = { min: 40, max: 180, label: 'Heart Rate (bpm)' };
  const xR = xRanges[xFeature];

  // Background
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#1a2235';
  ctx.fillRect(PAD.left, PAD.top, plotW, plotH);

  // Decision region (step-sampled for performance)
  const step = Math.max(2, Math.floor(plotW / 180));
  for (let px = 0; px < plotW; px += step) {
    for (let py = 0; py < plotH; py += step) {
      const xVal  = xR.min    + (px / plotW) * (xR.max    - xR.min);
      const hrVal = yRange.max - (py / plotH) * (yRange.max - yRange.min);

      const args = { hr: hrVal, bp: 120, act: 5, age: 40, chol: 180, smoking: false };
      if (xFeature === 'bp')   args.bp   = xVal;
      if (xFeature === 'chol') args.chol = xVal;
      if (xFeature === 'age')  args.age  = xVal;

      let score;
      if      (model === 'lr')  score = logisticRegPredict(args.hr, args.bp, args.act, args.age, args.chol, args.smoking);
      else if (model === 'knn') score = knnPredict(args.hr, args.bp, args.act, args.age, args.chol, args.smoking);
      else if (model === 'dt')  score = decisionTreePredict(args.hr, args.bp, args.act, args.age, args.chol, args.smoking);
      else                      score = randomForestPredict(args.hr, args.bp, args.act, args.age, args.chol, args.smoking);

      if      (score > 0.7) ctx.fillStyle = 'rgba(248,113,113,0.22)';
      else if (score > 0.3) ctx.fillStyle = 'rgba(251,191,36,0.18)';
      else                  ctx.fillStyle = 'rgba(52,211,153,0.18)';
      ctx.fillRect(PAD.left + px, PAD.top + py, step, step);
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const x = PAD.left + i / 5 * plotW;
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + plotH); ctx.stroke();
  }
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + i / 4 * plotH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + plotW, y); ctx.stroke();
  }

  // Training data points
  TRAINING_DATA.forEach(([thr, tbp, tact, tage, tchol, , label]) => {
    const xVal = xFeature === 'bp' ? tbp : xFeature === 'chol' ? tchol : tage;
    const px = PAD.left + (xVal - xR.min)    / (xR.max    - xR.min)    * plotW;
    const py = PAD.top  + (1 - (thr - yRange.min) / (yRange.max - yRange.min)) * plotH;
    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
    ctx.fillStyle   = label === 1 ? 'rgba(248,113,113,0.92)' : 'rgba(52,211,153,0.92)';
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth   = 1.5;
    ctx.fill();
    ctx.stroke();
  });

  // Axis labels
  ctx.fillStyle  = 'rgba(148,163,184,0.75)';
  ctx.font       = '11px JetBrains Mono, monospace';
  ctx.textAlign  = 'center';
  ctx.fillText(xR.label + ' →', PAD.left + plotW / 2, H - 6);
  ctx.save();
  ctx.translate(11, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yRange.label + ' →', 0, 0);
  ctx.restore();

  // X tick labels
  ctx.fillStyle = 'rgba(100,116,139,0.85)';
  ctx.font      = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const val = Math.round(xR.min + i / 5 * (xR.max - xR.min));
    ctx.fillText(val, PAD.left + i / 5 * plotW, H - PAD.bottom + 14);
  }

  // Y tick labels
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = Math.round(yRange.min + i / 4 * (yRange.max - yRange.min));
    const y   = PAD.top + (1 - i / 4) * plotH;
    ctx.fillText(val, PAD.left - 5, y + 4);
  }

  // Legend
  const legend = [
    { color: 'rgba(52,211,153,0.85)',  label: '● Safe' },
    { color: 'rgba(251,191,36,0.85)',  label: '● Moderate' },
    { color: 'rgba(248,113,113,0.85)', label: '● At Risk' },
  ];
  ctx.font      = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  legend.forEach(({ color, label }, i) => {
    ctx.fillStyle = color;
    ctx.fillText(label, PAD.left + 4 + i * 82, PAD.top + 14);
  });
}
