const logEl = document.querySelector("#log");
const statusPill = document.querySelector("#status-pill");
const buttons = [...document.querySelectorAll("button[data-command]")];

const formatNumber = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "No run";
  return Number(value).toFixed(4);
};

const setBusy = (busy, label = "Ready") => {
  statusPill.textContent = label;
  buttons.forEach((button) => {
    button.disabled = busy;
  });
};

const writeLog = (payload) => {
  logEl.textContent = JSON.stringify(payload, null, 2);
};

const fetchStatus = async () => {
  const response = await fetch("/api/status");
  const status = await response.json();
  renderStatus(status);
  return status;
};

const runCommand = async (command) => {
  setBusy(true, "Running");
  writeLog({ command, status: "running" });
  try {
    const response = await fetch(`/api/run/${command}`, { method: "POST" });
    const payload = await response.json();
    writeLog(payload);
    await fetchStatus();
    setBusy(false, payload.ok ? "Ready" : "Failed");
  } catch (error) {
    writeLog({ command, ok: false, error: String(error) });
    setBusy(false, "Failed");
  }
};

buttons.forEach((button) => {
  button.addEventListener("click", () => runCommand(button.dataset.command));
});

const renderStatus = (status) => {
  const artifacts = status.artifacts || {};
  const metrics = status.metrics || {};
  document.querySelector("#lstm-state").textContent = artifacts["lstm_world_model.pt"]
    ? "Available"
    : "Missing";

  const returns = [
    metrics["reinforce_metrics.json"]?.evaluation?.average_return,
    metrics["a2c_metrics.json"]?.evaluation?.average_return,
    metrics["random_policy_metrics.json"]?.average_return,
  ].filter((value) => value !== undefined);
  document.querySelector("#best-return").textContent = returns.length
    ? formatNumber(Math.max(...returns))
    : "No run";

  const safety = [
    metrics["reinforce_metrics.json"]?.evaluation?.safety_violations,
    metrics["a2c_metrics.json"]?.evaluation?.safety_violations,
    metrics["random_policy_metrics.json"]?.safety_violations,
  ].filter((value) => value !== undefined);
  document.querySelector("#safety-count").textContent = safety.length
    ? String(safety.reduce((sum, value) => sum + Number(value), 0))
    : "No run";

  drawReturnChart(metrics);
  drawLossChart(metrics);
};

const drawReturnChart = (metrics) => {
  const values = [
    ["REINFORCE", metrics["reinforce_metrics.json"]?.evaluation?.average_return, "#b64b37"],
    ["A2C", metrics["a2c_metrics.json"]?.evaluation?.average_return, "#126f64"],
    ["Random", metrics["random_policy_metrics.json"]?.average_return, "#315f9f"],
  ].filter((item) => item[1] !== undefined);
  drawBars(document.querySelector("#return-chart"), values);
};

const drawLossChart = (metrics) => {
  const losses = metrics["world_model_metrics.json"]?.train_losses || [];
  drawLine(document.querySelector("#loss-chart"), losses, "#126f64", "Run LSTM training to populate loss");
};

const drawBars = (canvas, values) => {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  drawEmptyAxes(ctx, width, height);
  if (!values.length) {
    drawCenteredText(ctx, width, height, "Run policy commands to populate this chart");
    return;
  }
  const numbers = values.map((item) => Number(item[1]));
  const min = Math.min(0, ...numbers);
  const max = Math.max(0, ...numbers);
  const range = max - min || 1;
  const zeroY = height - 42 - ((0 - min) / range) * (height - 82);
  const barWidth = Math.min(110, (width - 120) / values.length - 20);
  values.forEach(([label, value, color], index) => {
    const x = 70 + index * ((width - 120) / values.length) + 16;
    const y = height - 42 - ((Number(value) - min) / range) * (height - 82);
    const top = Math.min(y, zeroY);
    const barHeight = Math.abs(zeroY - y);
    ctx.fillStyle = color;
    ctx.fillRect(x, top, barWidth, Math.max(2, barHeight));
    ctx.fillStyle = "#17211b";
    ctx.font = "18px system-ui";
    ctx.fillText(formatNumber(value), x, top - 8);
    ctx.font = "16px system-ui";
    ctx.fillText(label, x, height - 15);
  });
};

const drawLine = (canvas, values, color, emptyText) => {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  drawEmptyAxes(ctx, width, height);
  if (!values.length) {
    drawCenteredText(ctx, width, height, emptyText);
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = 44 + (index / Math.max(values.length - 1, 1)) * (width - 80);
    const y = height - 42 - ((value - min) / range) * (height - 82);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#607066";
  ctx.font = "16px system-ui";
  ctx.fillText(`min ${formatNumber(min)}   max ${formatNumber(max)}`, 48, 28);
};

const drawEmptyAxes = (ctx, width, height) => {
  ctx.strokeStyle = "#d9dfd8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(42, 18);
  ctx.lineTo(42, height - 42);
  ctx.lineTo(width - 24, height - 42);
  ctx.stroke();
};

const drawCenteredText = (ctx, width, height, text) => {
  ctx.fillStyle = "#607066";
  ctx.font = "18px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
  ctx.textAlign = "start";
};

fetchStatus().then(() => setBusy(false, "Ready"));
