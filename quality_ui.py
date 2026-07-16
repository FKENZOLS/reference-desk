"""Self-contained reference-quality dashboard."""

from __future__ import annotations

import json
from typing import Any, Sequence


def _json_for_html(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def quality_dashboard_html(
    summary: dict[str, Any],
    feedback: Sequence[dict[str, Any]],
) -> str:
    """Render feedback, benchmark, and calibration state without a framework."""

    data = _json_for_html({"summary": summary, "feedback": list(feedback)})
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reference quality</title>
  <style>
    :root { color-scheme:dark; --bg:#0f1117; --panel:#171a22; --soft:#20242e; --line:#343a48; --ink:#f3f5f8; --muted:#aab2c0; --blue:#3b82f6; --green:#8de4b7; --amber:#f4c35b; --danger:#ff9d9d; }
    * { box-sizing:border-box; }
    [hidden] { display:none !important; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    button,a { font:inherit; }
    a { color:#9cc2ff; }
    button,.button { display:inline-flex; align-items:center; justify-content:center; min-height:40px; padding:8px 13px; border:1px solid var(--line); border-radius:9px; background:#252a35; color:var(--ink); text-decoration:none; cursor:pointer; }
    button:hover,.button:hover { background:#333a48; }
    .primary { background:#2563eb; border-color:#60a5fa; color:white; }
    .topbar { position:sticky; top:0; z-index:20; display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 24px; background:rgba(15,17,23,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }
    .nav,.actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .shell { width:min(1450px,calc(100% - 34px)); margin:0 auto; padding:28px 0 56px; }
    .hero { display:flex; justify-content:space-between; align-items:flex-start; gap:24px; margin-bottom:22px; }
    h1 { margin:0; font-size:clamp(1.65rem,2.5vw,2.2rem); }
    h2 { margin:0 0 13px; font-size:1.04rem; }
    p { color:var(--muted); line-height:1.5; }
    .stats { display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:10px; margin-bottom:16px; }
    .stat,.panel { border:1px solid var(--line); border-radius:13px; background:var(--panel); }
    .stat { padding:14px 16px; }
    .stat span { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
    .stat strong { display:block; margin-top:4px; font-size:1.45rem; }
    .panel { margin-bottom:16px; padding:18px; }
    .calibration { display:grid; grid-template-columns:minmax(260px,1fr) minmax(260px,1fr); gap:18px; }
    .meter { height:8px; overflow:hidden; border-radius:999px; background:#292f3b; }
    .meter span { display:block; height:100%; border-radius:inherit; background:#5ea0ff; }
    .progress-row { display:grid; grid-template-columns:90px 1fr auto; gap:10px; align-items:center; margin:10px 0; color:var(--muted); font-size:.82rem; }
    .status-pill { display:inline-flex; padding:5px 9px; border-radius:999px; font-size:.75rem; font-weight:700; }
    .status-pill.collecting { background:#45391f; color:var(--amber); }
    .status-pill.active { background:#203b31; color:var(--green); }
    .status-pill.paused { background:#3b2c30; color:#ffc0c0; }
    .metrics { display:grid; grid-template-columns:repeat(2,minmax(120px,1fr)); gap:9px; }
    .metric { padding:11px; border-radius:9px; background:var(--soft); }
    .metric span { display:block; color:var(--muted); font-size:.72rem; }
    .metric strong { display:block; margin-top:4px; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:11px; }
    table { width:100%; min-width:1000px; border-collapse:collapse; }
    th,td { padding:11px 12px; border-bottom:1px solid #2d3340; text-align:left; vertical-align:top; }
    th { background:#20242d; color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; }
    tbody tr:last-child td { border-bottom:0; }
    .query { min-width:220px; font-weight:650; line-height:1.4; }
    .source { max-width:260px; overflow-wrap:anywhere; color:var(--muted); font-size:.78rem; }
    .judgment { white-space:nowrap; }
    .judgment.relevant { color:var(--green); }
    .judgment.wrong_passage,.judgment.wrong_document,.judgment.no_relevant_result { color:var(--amber); }
    .empty { padding:34px; color:var(--muted); text-align:center; }
    .danger { color:var(--danger); }
    #pageStatus { min-height:1.2em; color:var(--green); }
    @media (max-width:850px) { .topbar,.hero{align-items:stretch;flex-direction:column}.stats{grid-template-columns:repeat(2,1fr)}.calibration{grid-template-columns:1fr}.nav,.actions{width:100%}.button,.actions button{flex:1} }
  </style>
</head>
<body>
  <header class="topbar">
    <strong>Reference quality</strong>
    <nav class="nav"><a class="button" href="/documents">Manage documents</a><a class="button" href="/workspace">Research workspace</a><a class="button primary" href="/">Search documents</a></nav>
  </header>
  <main class="shell">
    <section class="hero">
      <div><h1>Retrieval quality</h1><p>Human judgments become reproducible benchmark cases and, after enough labels, a safe evidence threshold.</p></div>
      <div class="actions"><a class="button" href="/quality/export/benchmark">Export benchmark JSONL</a><button id="recalibrate" type="button">Recalculate threshold</button></div>
    </section>
    <section id="stats" class="stats"></section>
    <section class="panel">
      <div class="calibration">
        <div>
          <h2>Evidence calibration <span id="calibrationState" class="status-pill collecting">Collecting</span></h2>
          <p id="calibrationMessage"></p>
          <div id="labelProgress"></div>
          <div class="actions"><button id="toggleGate" type="button" hidden></button><span id="pageStatus" aria-live="polite"></span></div>
        </div>
        <div id="calibrationMetrics" class="metrics"></div>
      </div>
    </section>
    <section class="panel">
      <h2>Recent judgments</h2>
      <div id="feedbackTable"></div>
    </section>
  </main>
  <script id="qualityData" type="application/json">@@DATA@@</script>
  <script>
    const state = JSON.parse(document.getElementById('qualityData').textContent);
    const byId = id => document.getElementById(id);
    const labels = {relevant:'Relevant',wrong_passage:'Wrong passage',wrong_document:'Wrong document',no_relevant_result:'No relevant result'};
    const percent = value => value == null ? 'Not available' : `${(Number(value) * 100).toFixed(1)}%`;
    const metric = (label,value) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;

    function render() {
      const summary = state.summary;
      const counts = summary.counts;
      const cards = [
        ['Judgments',summary.total],['Relevant',counts.relevant],['Incorrect',counts.wrong_passage + counts.wrong_document],['No-result labels',counts.no_relevant_result],['Benchmark cases',summary.benchmark_cases]
      ];
      byId('stats').innerHTML = cards.map(([label,value]) => `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`).join('');
      const calibration = summary.calibration;
      const pill = byId('calibrationState');
      pill.className = `status-pill ${calibration.active ? 'active' : calibration.ready ? 'paused' : 'collecting'}`;
      pill.textContent = calibration.active ? 'Active' : calibration.ready ? 'Paused' : 'Collecting';
      byId('calibrationMessage').textContent = calibration.ready
        ? (calibration.active ? 'Queries with no passage above the learned cutoff now return “No strong evidence found.”' : 'The learned cutoff is ready but paused; search continues to show the closest passages.')
        : 'Live rejection remains off until both label minimums are met. Benchmark export is available immediately.';
      const progress = [
        ['Relevant',calibration.positive_count,calibration.minimum_positive],
        ['Incorrect',calibration.negative_count,calibration.minimum_negative]
      ];
      byId('labelProgress').innerHTML = progress.map(([label,value,target]) => `<div class="progress-row"><span>${label}</span><div class="meter"><span style="width:${Math.min(100,value / target * 100)}%"></span></div><strong>${value}/${target}</strong></div>`).join('');
      byId('calibrationMetrics').innerHTML = [
        metric('Threshold',calibration.threshold == null ? 'Not learned' : Number(calibration.threshold).toFixed(4)),
        metric('Relevant recall',percent(calibration.positive_recall)),
        metric('Incorrect rejection',percent(calibration.specificity)),
        metric('Balanced accuracy',percent(calibration.balanced_accuracy)),
        metric('Answerable cases',summary.answerable_cases),
        metric('Unanswerable cases',summary.unanswerable_cases)
      ].join('');
      const toggle = byId('toggleGate');
      toggle.hidden = !calibration.ready;
      toggle.textContent = calibration.enabled ? 'Pause evidence gate' : 'Use calibrated gate';
      renderFeedback();
    }

    function renderFeedback() {
      const host = byId('feedbackTable');
      if (!state.feedback.length) { host.innerHTML = '<div class="empty">No judgments yet. Search for a passage and use the compact quality controls under a result.</div>'; return; }
      const table = document.createElement('div'); table.className = 'table-wrap';
      table.innerHTML = '<table><thead><tr><th>Query</th><th>Judgment</th><th>Source</th><th>Rank / logit</th><th>Updated</th><th></th></tr></thead><tbody></tbody></table>';
      const body = table.querySelector('tbody');
      state.feedback.forEach(item => {
        const row = document.createElement('tr');
        const query = row.appendChild(document.createElement('td')); query.className = 'query'; query.textContent = item.query;
        const judgment = row.appendChild(document.createElement('td')); judgment.className = `judgment ${item.judgment}`; judgment.textContent = labels[item.judgment] || item.judgment;
        const source = row.appendChild(document.createElement('td')); source.className = 'source'; source.textContent = [item.source_id,item.page_start ? `page ${item.page_start}` : '',item.section].filter(Boolean).join(' · ') || 'Query-level judgment';
        const score = row.appendChild(document.createElement('td')); score.textContent = [item.result_rank ? `#${item.result_rank}` : '',item.rerank_logit == null ? '' : Number(item.rerank_logit).toFixed(4)].filter(Boolean).join(' / ') || '—';
        const updated = row.appendChild(document.createElement('td')); updated.textContent = new Date(item.updated_at).toLocaleString();
        const action = row.appendChild(document.createElement('td')); const remove = action.appendChild(document.createElement('button')); remove.type = 'button'; remove.className = 'danger'; remove.textContent = 'Delete';
        remove.addEventListener('click',async () => {
          if (!confirm('Delete this quality judgment?')) return;
          const response = await fetch(`/quality/api/feedback/${item.id}`,{method:'DELETE'});
          if (!response.ok) return;
          location.reload();
        });
        body.appendChild(row);
      });
      host.replaceChildren(table);
    }

    byId('recalibrate').addEventListener('click',async () => {
      const response = await fetch('/quality/api/calibrate',{method:'POST'});
      if (!response.ok) { byId('pageStatus').textContent = 'Could not recalculate'; return; }
      state.summary.calibration = await response.json(); byId('pageStatus').textContent = 'Recalculated'; render();
    });
    byId('toggleGate').addEventListener('click',async () => {
      const enabled = !state.summary.calibration.enabled;
      const response = await fetch('/quality/api/calibration',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})});
      if (!response.ok) { byId('pageStatus').textContent = 'Could not update gate'; return; }
      state.summary.calibration = await response.json(); render();
    });
    render();
  </script>
</body>
</html>""".replace("@@DATA@@", data)
