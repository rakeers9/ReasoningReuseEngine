// ARC experiments UI: run panel, experiments list, experiment detail, compare.
// Plain JS, no build step. Talks to web/server.py.
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const app = $('#app');

const S = {
  meta: null,
  experiments: [],
  current: null,          // run id shown in the run panel
  out: { id: null, offset: 0, text: '' },
  detail: null,           // experiment detail payload
  detailReports: null,
  judgements: [],         // every judge decision with its evidence (server groups committees)
  reviews: {},            // your own verdicts, keyed like "call:57"
  callFilter: { stage: 'all', role: 'all', q: '', hits: null },
  openCall: null,
  timers: [],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function esc(x) {
  return String(x ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function money(x) {
  if (x == null || isNaN(x)) return '—';
  if (x === 0) return '$0';
  if (Math.abs(x) < 0.01) return '$' + x.toFixed(5);
  return '$' + x.toFixed(4);
}
function num(x) { return x == null ? '—' : Math.round(x).toLocaleString(); }
function tok(t, key) { return t.calls_with_tokens ? num(t[key]) : '—'; }
function secs(s) {
  if (s == null) return '—';
  if (s < 60) return s.toFixed(1) + ' s';
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m < 60 ? `${m}m ${r}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}
function when(t) {
  if (!t) return '—';
  const d = new Date(t * 1000);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' +
    d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
function status(s) { return `<span class="status ${esc(s)}">${esc(s)}</span>`; }
function yn(v) { return v == null ? '—' : v ? '<span class="pass">yes</span>' : '<span class="fail">no</span>'; }
function frac(m) { return m && m.total ? `${m.passed}/${m.total}` : '—'; }
function mult(x) { return x == null ? '—' : x.toFixed(1) + '×'; }
function cmdLine(e) { return 'python -m arc ' + (e.argv || []).map(a => /^[\w./:=-]+$/.test(a) ? a : JSON.stringify(a)).join(' '); }
function stageName(s) { return (S.meta && S.meta.stage_names[s]) || s; }

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({ error: 'bad response' }));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });

function clearTimers() { S.timers.forEach(clearInterval); S.timers = []; }
function every(ms, fn) { S.timers.push(setInterval(fn, ms)); }

function store(key, val) { try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; } }

function confirmDialog(title, bodyHtml, yes = 'Run') {
  const dlg = $('#confirm-dlg');
  $('#confirm-title').textContent = title;
  $('#confirm-body').innerHTML = bodyHtml;
  $('#confirm-yes').textContent = yes;
  return new Promise(resolve => {
    const done = v => { dlg.close(); $('#confirm-yes').onclick = $('#confirm-no').onclick = null; resolve(v); };
    $('#confirm-yes').onclick = () => done(true);
    $('#confirm-no').onclick = () => done(false);
    dlg.oncancel = () => done(false);
    dlg.showModal();
  });
}

// --- Verdict pills and your own review ------------------------------------

function pill(ok, yes = 'PASS', no = 'FAIL') {
  if (ok == null) return '<span class="pill meh">—</span>';
  return `<span class="pill ${ok ? 'ok' : 'bad'}">${ok ? yes : no}</span>`;
}
function ncPill(nc) {
  if (!nc || !nc.status) return '';
  const cls = { match: 'ok', mismatch: 'bad', unclear: 'meh' }[nc.status] || 'meh';
  const label = { match: 'numbers match', mismatch: 'numbers differ', unclear: 'no numbers to compare' }[nc.status];
  return `<span class="pill ${cls}" title="No-LLM number check against the ${esc(nc.against || 'reference')}: ${esc(nc.detail || '')}">${label}</span>`;
}
function ncLine(nc) {
  if (!nc || !nc.status) return '';
  return `<p class="small muted"><b>Number check (no LLM):</b> ${esc(nc.detail || nc.status)}${nc.against ? ` · against the ${esc(nc.against)}` : ''}. A simple text check, useful as a second opinion only.</p>`;
}
function judgePills(verdicts) {
  return (verdicts || []).map((v, k) => `<span class="pill ${v.passed ? 'ok' : v.passed === false ? 'bad' : 'meh'}" title="${esc(v.critique || '')}">Judge ${k + 1}: ${v.passed ? 'pass' : v.passed === false ? 'fail' : '?'}</span>`).join('');
}
function judgeNotes(verdicts) {
  return (verdicts || []).map((v, k) => `<div class="verdict ${v.passed ? 'ok' : 'bad'}"><b>Judge ${k + 1} · ${v.passed ? 'pass' : 'fail'}</b>${v.model ? ` <span class="muted small mono">${esc(v.model.replace('claude-', ''))}</span>` : ''}<p>${esc(v.critique || '(no critique)')}</p></div>`).join('');
}
function yourPill(key) {
  const r = S.reviews[key];
  if (!r || !r.verdict) return '<span class="pill meh">not reviewed</span>';
  return `<span class="pill ${{ correct: 'ok', wrong: 'bad', unsure: 'meh' }[r.verdict]}">you: ${esc(r.verdict)}</span>`;
}
function disagrees(key, judgePassed) {
  const r = S.reviews[key];
  return !!(r && (r.verdict === 'correct' || r.verdict === 'wrong') && judgePassed != null && (r.verdict === 'correct') !== !!judgePassed);
}
function reviewBox(key, judgePassed, question = 'Was the judges’ decision right?') {
  const r = S.reviews[key] || {};
  const b = (v, label) => `<button type="button" class="btn rv${r.verdict === v ? ' on ' + v : ''}" data-rv="${v}" aria-pressed="${r.verdict === v}">${label}</button>`;
  return `<div class="review" data-key="${esc(key)}" data-judge="${judgePassed == null ? '' : judgePassed ? '1' : '0'}">
    <div class="row"><span class="lbl">Your review</span><span class="muted small">${esc(question)}</span><span class="spacer"></span>
      ${disagrees(key, judgePassed) ? '<span class="pill bad">you disagree with the judges</span>' : ''}
      <span class="rvset">${b('correct', '✓ Correct')}${b('wrong', '✕ Wrong')}${b('unsure', '? Unsure')}</span></div>
    <textarea class="in note-in" rows="1" placeholder="Note (optional): why?">${esc(r.note || '')}</textarea></div>`;
}
async function saveReview(box, verdict) {
  const key = box.dataset.key, jp = box.dataset.judge === '' ? null : box.dataset.judge === '1';
  const note = $('.note-in', box).value;
  try {
    S.reviews = await post(`/api/experiments/${encodeURIComponent(S.detail.id)}/reviews`, { key, verdict, note, judge_passed: jp });
  } catch (e) { alert(e.message); return; }
  $$(`.review[data-key="${CSS.escape(key)}"]`).forEach(el => { el.outerHTML = reviewBox(key, jp, $('.muted', el)?.textContent); });
  $$(`[data-your="${CSS.escape(key)}"]`).forEach(el => { el.innerHTML = yourPill(key); });
}
document.addEventListener('click', ev => {
  const btn = ev.target.closest('.review [data-rv]');
  if (!btn) return;
  const box = btn.closest('.review'), cur = (S.reviews[box.dataset.key] || {}).verdict;
  saveReview(box, cur === btn.dataset.rv ? null : btn.dataset.rv);
});
document.addEventListener('change', ev => {
  const t = ev.target.closest('.review .note-in');
  if (t) { const box = t.closest('.review'); saveReview(box, (S.reviews[box.dataset.key] || {}).verdict || null); }
});

// One stat tile. `tip` is a plain-English hover explanation of the number.
function stat(label, value, sub = '', tip = '', cls = '') {
  return `<div class="stat ${cls}"${tip ? ` title="${esc(tip)}"` : ''}><span>${tip ? `<span data-tip>${esc(label)}</span>` : esc(label)}</span><b>${value}</b>${sub ? `<span>${sub}</span>` : ''}</div>`;
}
// Short plain-English intro at the top of a view or tab.
function explain(title, ...paras) {
  return `<div class="card explain"><h3>${esc(title)}</h3>${paras.map((p, k) => `<p${k ? ' class="small"' : ''}>${p}</p>`).join('')}</div>`;
}
function seg(attr, items, label) {
  return `<span class="fgroup">${label ? `<span class="lbl">${esc(label)}</span>` : ''}<span class="seg">${items.map(([v, l]) => `<button class="chip" type="button" data-${attr}="${esc(v)}">${esc(l)}</button>`).join('')}</span></span>`;
}
function head(title, lede, right = '') {
  return `<div class="pagehead"><div class="row"><h1>${esc(title)}</h1><span class="spacer"></span>${right}</div>${lede ? `<p class="lede">${lede}</p>` : ''}</div>`;
}

function block(summary, body, open = false) {
  return `<details class="blk"${open ? ' open' : ''}><summary>${summary}</summary><div class="body">${body}</div></details>`;
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  $('#theme-btn').textContent = { light: '☀︎ Light', dark: '☾ Dark' }[t] || '◐ Auto';
}
$('#theme-btn').addEventListener('click', () => {
  const cur = store('arc-theme') || 'auto';
  const next = { auto: 'light', light: 'dark', dark: 'auto' }[cur] || 'auto';
  store('arc-theme', next);
  applyTheme(next === 'auto' ? null : next);
});
applyTheme(store('arc-theme'));

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

function route() {
  clearTimers();
  closeDrawer();
  const h = location.hash.replace(/^#/, '') || 'run';
  const [view, a, b] = h.split('/');
  $$('nav.tabs a').forEach(x => x.setAttribute('aria-current', x.dataset.nav === (view === 'exp' ? 'experiments' : view) ? 'page' : 'false'));
  if (view === 'experiments') return viewExperiments();
  if (view === 'exp' && a) return viewDetail(decodeURIComponent(a), b || 'overview');
  if (view === 'compare') return viewCompare(a ? decodeURIComponent(a).split(',').filter(Boolean) : []);
  return viewRun();
}
window.addEventListener('hashchange', route);

async function refreshMeta() {
  S.meta = await api('/api/meta');
  const pill = $('#running-pill');
  if (S.meta.running.length) {
    pill.classList.add('on');
    $('#running-text').textContent = 'running ' + S.meta.running[0];
    pill.href = '#exp/' + encodeURIComponent(S.meta.running[0]) + '/output';
    if (!S.current) S.current = S.meta.running[0];
  } else pill.classList.remove('on');
}
setInterval(() => refreshMeta().catch(() => {}), 4000);

// ---------------------------------------------------------------------------
// Run view
// ---------------------------------------------------------------------------

const CMD_INFO = {
  paper: 'Step 1: run the paper\'s published functions on our runtime. Only the small model is called (~$0.00003).',
  ask: 'Step 2: serve one question through a cache. A hit costs ~$0.0001; a miss falls back to the source model.',
  solve: 'Step 3: the source model writes a worked solution. Overwrites this profile\'s transcript for the problem.',
  build: 'Step 4: the builder turns the transcript into a cache entry (builder + judges). Overwrites this profile\'s cache for the problem and merges into its master.',
  eval: 'Step 5: correctness, test variants and cost multipliers (generator + judges).',
  compile: 'Step 6: static compilation to plain Python, then checks it matches the dynamic cache. Free unless --extract.',
  all: 'Steps 3–6 in order: solve, build, eval, compile.',
};
const SETTING_USE = {
  PROFILE: 'haiku or paper (Opus)', SOURCE_MODEL: 'solve, ask fallback', BUILDER_MODEL: 'build', JUDGE_MODEL: 'build, eval', GENERATOR_MODEL: 'eval',
  TARGET_MODEL: 'all cache use', SOURCE_EFFORT: 'solve', BUILDER_EFFORT: 'build', JUDGE_EFFORT: 'build, eval',
  GENERATOR_EFFORT: 'eval', EMBED_MODEL: 'lookups', LOAD_THRESHOLD: 'load()', DO_THRESHOLD: 'do()',
  ENTRY_THRESHOLD: 'entry match', MAX_FUNCTION_ATTEMPTS: 'build', MAX_ROOT_REPAIRS: 'build', EXTRACTION_ROUNDS: 'build',
  GRANULARITY: 'build', FUNCTION_TIMEOUT_SECONDS: 'cache code', VARIANTS_PER_MODE: 'eval', JUDGE_COMMITTEE_SIZE: 'build, eval',
};

function formState() {
  const f = $('#run-form');
  const overrides = {};
  $$('.ov-grid input', f).forEach(i => { if (i.value.trim() !== '') overrides[i.name] = i.value.trim(); });
  return {
    command: f.command.value, problem: f.problem.value, cache: f.cache.value, question: f.question.value,
    no_golden: f.no_golden.checked, extract: f.extract.checked, fresh_eval: f.fresh_eval.checked,
    label: f.label.value, overrides,
  };
}

function saveForm() { try { store('arc-form', JSON.stringify(formState())); } catch (e) {} }

async function viewRun() {
  await refreshMeta();
  const m = S.meta;
  let saved = {};
  try { saved = JSON.parse(store('arc-form') || '{}') || {}; } catch (e) {}
  const opt = (v, cur) => `<option value="${esc(v)}"${v === cur ? ' selected' : ''}>${esc(v)}</option>`;
  const ovRows = m.setting_names.map(n => `
      <span class="mono" title="ARC_${esc(n)}">${esc(n)}<br><span class="muted small">${esc(SETTING_USE[n] || '')}</span></span>
      <span class="mono muted">${esc(m.settings[n])}</span>
      <input name="${esc(n)}" placeholder="default" value="${esc((saved.overrides || {})[n] || '')}" aria-label="Override ${esc(n)}">`).join('');
  app.innerHTML = `
  <div class="view">
    ${head('Run an experiment', 'Pick one step of the pipeline (or all of them) and watch it run. Viewing results is free; <b>solve, build, eval and all</b> make paid model calls, so you are asked to confirm first.')}
    <div class="run-grid">
      <form class="card form" id="run-form" autocomplete="off">
        <h2>New experiment</h2>
        <div class="form-row">
          <label class="field">Command
            <select name="command">${m.commands.map(c => opt(c, saved.command || 'paper')).join('')}</select></label>
          <label class="field" data-show="problem">Problem
            <select name="problem">${m.problems.map(p => opt(p.id, saved.problem || 'gas_law')).join('')}</select></label>
        </div>
        <p class="muted small" id="cmd-info"></p>
        <label class="field" data-show="cache">Cache
          <select name="cache">${m.caches.map(c => opt(c, saved.cache)).join('')}</select></label>
        <label class="field" data-show="question">Question
          <textarea name="question" rows="4" placeholder="A question to serve through the cache">${esc(saved.question || '')}</textarea></label>
        <div class="stack">
          <label class="check" data-show="no_golden"><input type="checkbox" name="no_golden"${saved.no_golden ? ' checked' : ''}> <span><span class="mono">--no-golden</span> build without the answer key</span></label>
          <label class="check" data-show="extract"><input type="checkbox" name="extract"${saved.extract ? ' checked' : ''}> <span><span class="mono">--extract</span> extract parameters with the small model</span></label>
          <label class="check" data-show="fresh_eval"><input type="checkbox" name="fresh_eval"${saved.fresh_eval === false ? '' : ' checked'}> <span>Fresh eval: move <span class="mono">eval_progress.json</span> into this experiment first. Without this, eval resumes and skips work already done.</span></label>
        </div>
        <label class="field">Label (optional)
          <input name="label" placeholder="what you are testing" value="${esc(saved.label || '')}"></label>
        <details class="blk" id="ov-box">
          <summary><b>Setting overrides</b> <span class="muted small" id="ov-count"></span></summary>
          <div class="body">
            <p class="muted small">Blank uses the value in <span class="mono">arc/config.py</span>. Each is passed as an <span class="mono">ARC_&lt;NAME&gt;</span> env var to this run only.</p>
            <div class="ov-grid"><span class="h">Setting</span><span class="h">Default</span><span class="h">This run</span>${ovRows}</div>
            <div><button class="btn sm" type="button" id="ov-clear">Clear overrides</button></div>
          </div>
        </details>
        <div class="estimate" id="estimate">Estimating…</div>
        <div class="row"><button class="btn primary" type="submit" id="run-btn">Run</button>
          <span class="muted small" id="run-msg"></span></div>
      </form>
      <section class="card stack" id="live">
        ${liveShell()}
      </section>
    </div>
  </div>`;
  const f = $('#run-form');
  const sync = () => {
    const st = formState();
    const show = {
      problem: st.command !== 'ask', cache: ['ask', 'eval', 'compile', 'all'].includes(st.command),
      question: st.command === 'ask', no_golden: ['build', 'all'].includes(st.command),
      extract: ['compile', 'all'].includes(st.command), fresh_eval: ['eval', 'all'].includes(st.command),
    };
    $$('[data-show]', f).forEach(x => { x.style.display = show[x.dataset.show] ? '' : 'none'; });
    $('#cmd-info').textContent = CMD_INFO[st.command] || '';
    const n = Object.keys(st.overrides).length;
    $('#ov-count').textContent = n ? `${n} set` : 'none';
    $$('.ov-grid input', f).forEach(i => i.classList.toggle('set', i.value.trim() !== ''));
    saveForm();
    updateEstimate(st);
  };
  f.command.addEventListener('change', () => {
    const c = f.command.value;
    if (c === 'ask' && !f.cache.value.startsWith('paper_') && m.caches.includes('paper_gas_law')) f.cache.value = 'paper_gas_law';
    if (['eval', 'compile', 'all'].includes(c) && m.caches.includes('master')) f.cache.value = 'master';
    sync();
  });
  f.addEventListener('input', sync);
  $('#ov-clear').addEventListener('click', () => { $$('.ov-grid input', f).forEach(i => { i.value = ''; }); sync(); });
  f.addEventListener('submit', ev => { ev.preventDefault(); startRun(); });
  if (!f.cache.value && m.caches.length) f.cache.value = m.caches.includes('paper_gas_law') ? 'paper_gas_law' : m.caches[0];
  sync();
  if (S.current) attachLive(S.current);
}

let estTimer = null;
function updateEstimate(st) {
  clearTimeout(estTimer);
  estTimer = setTimeout(async () => {
    try {
      const q = new URLSearchParams({ command: st.command, problem: st.command === 'ask' ? '' : st.problem, extract: st.extract ? '1' : '0', profile: st.overrides.PROFILE || '' });
      const e = await api('/api/estimate?' + q);
      S.lastEstimate = e;
      const range = e.min != null && e.max !== e.min ? ` (range ${money(e.min)}–${money(e.max)})` : '';
      $('#estimate').innerHTML = `Estimated cost <b>${money(e.cost)}</b>${range}<br><span class="muted small">${esc(e.basis)}</span>`;
    } catch (err) { $('#estimate').textContent = 'Estimate unavailable: ' + err.message; }
  }, 150);
}

async function startRun() {
  const st = formState();
  const m = S.meta;
  const desc = st.command === 'ask' ? `ask ${st.cache} "${st.question}"` : `${st.command} ${st.problem}`;
  if (m.confirm.includes(st.command)) {
    const e = S.lastEstimate || { cost: 0, basis: '' };
    const ov = Object.entries(st.overrides).map(([k, v]) => `${k}=${v}`).join(', ');
    const ok = await confirmDialog(`Run ${st.command}?`,
      `<span class="mono">${esc(desc)}</span> makes paid model calls (profile <b>${esc(st.overrides.PROFILE || m.profile)}</b>).<br><br>Estimated cost: <b>${money(e.cost)}</b> <span class="muted">(${esc(e.basis)})</span>` +
      (ov ? `<br>Overrides: <span class="mono">${esc(ov)}</span>` : '') +
      (['eval', 'all'].includes(st.command) && !st.fresh_eval ? '<br><br><span class="warn">Fresh eval is off: eval will resume from the saved progress.</span>' : ''),
      `Run (${money(e.cost)})`);
    if (!ok) return;
  }
  $('#run-btn').disabled = true;
  $('#run-msg').textContent = 'starting…';
  try {
    const r = await post('/api/run', st);
    S.current = r.run_id;
    S.out = { id: null, offset: 0, text: '' };
    $('#run-msg').textContent = '';
    await refreshMeta();
    attachLive(r.run_id);
  } catch (err) {
    $('#run-msg').innerHTML = `<span class="fail">${esc(err.message)}</span>`;
  } finally { $('#run-btn').disabled = false; }
}

function liveShell() {
  return `<div class="row"><h2>Live output</h2><span class="spacer"></span><span id="live-status"></span></div>
    <p class="muted" id="live-empty">Nothing running. Start an experiment and its console output, cost and call count appear here, live.</p>
    <div id="live-body" style="display:none" class="stack">
      <div class="row"><a id="live-link" class="mono" href="#"></a><span class="spacer"></span>
        <button class="btn sm danger" type="button" id="stop-btn">Stop</button></div>
      <div class="stats" id="live-stats"></div>
      <pre class="console" id="console"></pre>
    </div>`;
}

function attachLive(id) {
  $('#live-empty').style.display = 'none';
  $('#live-body').style.display = '';
  $('#live-link').textContent = id;
  $('#live-link').href = '#exp/' + encodeURIComponent(id);
  $('#stop-btn').onclick = async () => {
    const ok = await confirmDialog('Stop this run?', 'Sends an interrupt to the process. Calls already made are kept and still billed.', 'Stop');
    if (ok) post(`/api/experiments/${encodeURIComponent(id)}/stop`).catch(e => alert(e.message));
  };
  if (S.out.id !== id) S.out = { id, offset: 0, text: '' };
  $('#console').textContent = S.out.text;
  let finished = false;
  const tick = async () => {
    if (finished || S.current !== id || !$('#console')) return;
    try {
      const o = await api(`/api/experiments/${encodeURIComponent(id)}/output?offset=${S.out.offset}`);
      if (o.text) {
        const con = $('#console');
        const atBottom = con.scrollHeight - con.scrollTop - con.clientHeight < 40;
        S.out.text += o.text; S.out.offset = o.offset;
        con.textContent = S.out.text;
        if (atBottom) con.scrollTop = con.scrollHeight;
      }
      const d = await api(`/api/experiments/${encodeURIComponent(id)}`);
      renderLiveStats(d);
      if (d.status !== 'running') { finished = true; $('#stop-btn').style.display = 'none'; refreshMeta(); }
      else $('#stop-btn').style.display = '';
    } catch (e) { /* server restarting; try again */ }
  };
  tick();
  every(1000, tick);
}

function renderLiveStats(d) {
  const t = d.totals;
  $('#live-status').innerHTML = status(d.status);
  const est = d.estimate ? `<span>est. ${money(d.estimate.cost)}</span>` : '';
  let extra = '';
  if (d.ask_path) extra = `<div class="stat"><span>Path</span><b class="small">${esc(d.ask_path)}</b></div>`;
  if (d.eval) extra += `<div class="stat"><span>Variants passed</span><b>${['phrasal', 'parametric', 'combined'].map(k => frac(d.eval.modes[k])).join(' · ')}</b></div>`;
  $('#live-stats').innerHTML = `
    <div class="stat"><span>Cost so far</span><b>${money(t.cost)}</b>${est}</div>
    <div class="stat"><span>Calls</span><b>${t.calls}</b><span>${t.errors ? t.errors + ' errors' : ''}${t.refusals ? ' ' + t.refusals + ' refusals' : ''}</span></div>
    <div class="stat"><span>Tokens in / out</span><b>${tok(t, 'input')} / ${tok(t, 'output')}</b></div>
    <div class="stat"><span>Elapsed</span><b>${secs(d.seconds)}</b></div>${extra}`;
}

// ---------------------------------------------------------------------------
// Experiments list
// ---------------------------------------------------------------------------

function resultCell(e) {
  if (e.eval) {
    const m = e.eval.modes;
    return `${frac(m.phrasal)} · ${frac(m.parametric)} · ${frac(m.combined)}` + (e.eval.costs.savings_multiplier ? ` <span class="muted">· ${mult(e.eval.costs.savings_multiplier)} savings</span>` : '');
  }
  if (e.build) return e.build.success ? `<span class="pass">built</span> ${e.build.functions} fns` : '<span class="fail">build failed</span>';
  if (e.ask_path) return esc(e.ask_path);
  if (e.error) return `<span class="fail">${esc(e.error.slice(0, 80))}</span>`;
  return '';
}

async function viewExperiments() {
  const selected = new Set(JSON.parse(store('arc-selected') || '[]'));
  app.innerHTML = `<div class="view">
    ${head('Experiments', 'Every run so far, newest first. Click a row to open its story: what was solved, built and tested, what it cost, and whether the cache held up. Tick two or more to compare them side by side.',
      '<span class="muted small" id="exp-count"></span><button class="btn primary" type="button" id="cmp-btn">Compare selected</button>')}
    <div class="wide card"><table id="exp-table"><thead><tr>
      <th></th><th>Experiment</th><th>Command</th><th>Status</th><th>Started</th><th class="n">Duration</th>
      <th class="n">Calls</th><th class="n">Tokens in</th><th class="n">Tokens out</th><th class="n">Cost</th><th>Result</th>
    </tr></thead><tbody></tbody></table></div>
    <p class="muted small">In the Result column, pass counts read <b>phrasal · parametric · combined</b> (reworded questions · new numbers · both). “Savings” is how many times cheaper a cached answer is than a fresh solve. The legacy run's per-call tokens were not logged; its costs are exact.</p>
  </div>`;
  const draw = () => {
    const rows = S.experiments.map(e => `
      <tr class="click${selected.has(e.id) ? ' sel' : ''}" data-id="${esc(e.id)}">
        <td><input type="checkbox" aria-label="Select ${esc(e.id)}" data-sel="${esc(e.id)}"${selected.has(e.id) ? ' checked' : ''}></td>
        <td class="id"><span class="mono">${esc(e.id)}</span>${e.label ? `<br><span class="muted small">${esc(e.label)}</span>` : ''}</td>
        <td class="trunc mono small" title="${esc(cmdLine(e))}">${esc((e.argv || []).join(' '))}${Object.keys(e.overrides || {}).length ? ` <span class="tag">${Object.keys(e.overrides).length} overrides</span>` : ''}</td>
        <td>${status(e.status)}</td><td class="small">${when(e.started)}</td><td class="n">${secs(e.seconds)}</td>
        <td class="n">${e.totals.calls}</td><td class="n">${tok(e.totals, 'input')}</td><td class="n">${tok(e.totals, 'output')}</td>
        <td class="n"><b>${money(e.totals.cost)}</b></td><td class="small res">${resultCell(e)}</td></tr>`).join('');
    $('#exp-table tbody').innerHTML = rows || '<tr><td colspan="11" class="muted">No experiments yet.</td></tr>';
    const total = S.experiments.reduce((a, e) => a + e.totals.cost, 0);
    $('#exp-count').textContent = `${S.experiments.length} experiments · ${money(total)} total`;
    $('#cmp-btn').textContent = `Compare selected (${selected.size})`;
    $('#cmp-btn').disabled = selected.size < 2;
  };
  $('#exp-table').addEventListener('click', ev => {
    const cb = ev.target.closest('[data-sel]');
    if (cb) {
      cb.checked ? selected.add(cb.dataset.sel) : selected.delete(cb.dataset.sel);
      store('arc-selected', JSON.stringify([...selected]));
      draw();
      return;
    }
    const tr = ev.target.closest('tr[data-id]');
    if (tr) location.hash = 'exp/' + encodeURIComponent(tr.dataset.id);
  });
  $('#cmp-btn').addEventListener('click', () => {
    const ids = S.experiments.map(e => e.id).filter(id => selected.has(id));
    location.hash = 'compare/' + ids.map(encodeURIComponent).join(',');
  });
  const load = async () => { S.experiments = await api('/api/experiments'); draw(); };
  await load();
  every(3000, () => { if (S.experiments.some(e => e.status === 'running') || (S.meta && S.meta.running.length)) load(); });
}

// ---------------------------------------------------------------------------
// Experiment detail
// ---------------------------------------------------------------------------

async function viewDetail(id, tab) {
  const fresh = !S.detail || S.detail.id !== id;
  if (fresh) {
    app.innerHTML = '<p class="muted">Loading…</p>';
    S.callFilter = { stage: 'all', role: 'all', q: '', hits: null };
    S.detailReports = null;
  }
  try {
    if (!S.meta) await refreshMeta();
    S.detail = await api('/api/experiments/' + encodeURIComponent(id));
    if (!S.detailReports || fresh) S.detailReports = await api(`/api/experiments/${encodeURIComponent(id)}/reports`);
    if (fresh || S.detail.status === 'running') {
      [S.judgements, S.reviews] = await Promise.all([
        api(`/api/experiments/${encodeURIComponent(id)}/judgements`),
        api(`/api/experiments/${encodeURIComponent(id)}/reviews`)]);
    }
  } catch (e) { app.innerHTML = `<p class="note err">${esc(e.message)}</p>`; return; }
  drawDetail(tab);
  if (S.detail.status === 'running') {
    every(2000, async () => {
      if (!S.detail || S.detail.id !== id) return;
      const d = await api('/api/experiments/' + encodeURIComponent(id)).catch(() => null);
      if (!d) return;
      const was = S.detail.calls.length;
      S.detail = d;
      if (d.status !== 'running') S.detailReports = await api(`/api/experiments/${encodeURIComponent(id)}/reports`).catch(() => S.detailReports);
      drawHeader();
      if (tab === 'calls' && d.calls.length !== was) drawCalls();
      if (tab === 'totals' || tab === 'overview') drawTab(tab);
      if (d.status !== 'running') {
        clearTimers();
        S.judgements = await api(`/api/experiments/${encodeURIComponent(id)}/judgements`).catch(() => S.judgements);
        drawDetail(tab);
      }
    });
  }
}

// Tabs follow the pipeline story first, then the technical views.
function detailTabs() {
  const d = S.detail, r = S.detailReports || {};
  const story = [['overview', 'Overview']];
  if (r.transcript) story.push(['source', 'Source solve', null, 3]);
  if (r.build) story.push(['build', 'Build', r.build.attempts.length + ' attempts', 4]);
  if (r.eval || r.eval_progress) story.push(['eval', 'Eval', r.eval ? r.eval.variants.length + ' variants' : 'partial', 5]);
  if (S.judgements.length) story.push(['review', 'Review verdicts', S.judgements.length, null, true]);
  if (r.compiled) story.push(['compiled', 'Compiled', null, 6]);
  const tech = [['calls', 'Calls', d.calls.length], ['totals', 'Totals'], ['output', 'Console'], ['settings', 'Settings']];
  return { story, tech };
}

function drawDetail(tab) {
  const d = S.detail;
  const { story, tech } = detailTabs();
  if (![...story, ...tech].some(t => t[0] === tab)) tab = 'overview';
  const link = ([k, label, cnt, stepn, hl]) =>
    `<a href="#exp/${encodeURIComponent(d.id)}/${k}" class="${hl ? 'hl' : ''}" aria-current="${k === tab ? 'page' : 'false'}">${stepn ? `<span class="stepn">${stepn}</span>` : ''}${esc(label)}${cnt != null ? `<span class="cnt">${esc(cnt)}</span>` : ''}</a>`;
  app.innerHTML = `<div class="view">
    <div id="d-head"></div>
    <nav class="tabbar" aria-label="Experiment sections">
      <div class="tabgroup">${story.map(link).join('')}</div>
      <span class="tabsep">Technical</span>
      <div class="tabgroup tech">${tech.map(link).join('')}</div>
    </nav>
    <div id="d-tab"></div></div>`;
  S.tab = tab;
  drawHeader();
  drawTab(tab);
}

function drawHeader() {
  const d = S.detail, t = d.totals;
  const el = $('#d-head');
  if (!el) return;
  const tokNote = t.calls && t.calls_with_tokens < t.calls ? `${t.calls_with_tokens}/${t.calls} calls have token counts` : '';
  el.innerHTML = `
    <div class="dhead">
      <a href="#experiments" class="crumb">← All experiments</a>
      <div class="row"><h1 class="mono">${esc(d.id)}</h1>${status(d.status)}
        ${d.label ? `<span class="muted">${esc(d.label)}</span>` : ''}<span class="spacer"></span>
        ${d.status === 'running' ? `<button class="btn sm danger" type="button" id="d-stop">Stop</button>` : ''}
        <a class="btn sm" href="#compare/${encodeURIComponent(d.id)}">Compare with…</a></div>
      ${modelsLine(d)}
      <p class="cmdline">${esc(d.legacy ? 'legacy: steps 1–6 run by hand on gas_law, before experiment IDs existed' : cmdLine(d))}
        ${Object.entries(d.overrides || {}).map(([k, v]) => ` <span class="tag">${esc(k)}=${esc(v)}</span>`).join('')}</p>
      ${d.error ? `<p class="note err mono small">${esc(d.error)}</p>` : ''}
    </div>
    <div class="stats compact" style="margin-top:14px">
      ${stat('Total cost', money(t.cost), d.estimate ? `estimated ${money(d.estimate.cost)}` : '', 'Money spent on every model call in this experiment.')}
      ${stat('Calls', t.calls, `${t.errors ? t.errors + ' errors ' : ''}${t.refusals ? t.refusals + ' refusals' : ''}`, 'How many times a model was called.')}
      ${stat('Tokens in / out', `${tok(t, 'input')} / ${tok(t, 'output')}`, tokNote, 'Tokens are the units models read (in) and write (out). Output includes hidden thinking.')}
      ${stat('Cache read / write', `${tok(t, 'cache_read')} / ${tok(t, 'cache_write')}`, '', 'Prompt-caching tokens: text the API stored and re-read at a discount. Not the same as the reasoning cache.')}
      ${stat('Duration', secs(d.seconds), when(d.started), 'Wall-clock time from start to finish.')}
    </div>`;
  const stop = $('#d-stop');
  if (stop) stop.onclick = async () => {
    if (await confirmDialog('Stop this run?', 'Calls already made are kept and still billed.', 'Stop'))
      post(`/api/experiments/${encodeURIComponent(d.id)}/stop`).catch(e => alert(e.message));
  };
}

function modelsLine(d) {
  const m = d.models || {};
  if (!Object.keys(m).length) return '';
  const short = x => esc((x || '—').replace('claude-', ''));
  return `<p class="models"><span class="pill ${d.profile === 'paper' ? 'meh' : 'ok'}" title="Which set of models this run used">${esc(d.profile || '?')} profile</span>
    <span>source <b>${short(m.SOURCE_MODEL)}</b></span><span>builder <b>${short(m.BUILDER_MODEL)}</b></span><span>judges <b>${short(m.JUDGE_MODEL)}</b></span>
    <span>generator <b>${short(m.GENERATOR_MODEL)}</b></span><span>small model <b>${short(m.TARGET_MODEL)}</b></span></p>`;
}

function drawTab(tab) {
  const fn = { overview: drawOverview, review: drawReview, calls: drawCalls, totals: drawTotals, source: drawSource, build: drawBuild, eval: drawEval,
    compiled: drawCompiled, output: drawOutput, settings: drawSettings }[tab] || drawOverview;
  fn();
}

// --- Overview ----------------------------------------------------------------

function reviewTally() {
  const t = { total: S.judgements.length, reviewed: 0, correct: 0, wrong: 0, unsure: 0, disagree: 0, look: 0 };
  S.judgements.forEach(g => {
    const v = (S.reviews[g.key] || {}).verdict;
    if (v) { t.reviewed++; t[v]++; }
    if (disagrees(g.key, g.passed)) t.disagree++;
    if (needsLook(g)) t.look++;
  });
  return t;
}

function drawOverview() {
  const d = S.detail, r = S.detailReports || {}, ev = d.eval, b = d.build, c = ev && ev.costs;
  const by = d.by_stage || {};
  const base = `#exp/${encodeURIComponent(d.id)}`;
  const cmd = d.legacy ? 'legacy' : d.command;
  const intro = {
    legacy: 'An early hand-run of the whole pipeline on one problem: the paper’s functions, serving, then solve → build → eval → compile.',
    all: 'The full pipeline on one problem: solve it once, build the cache, test it, and compile it to plain Python.',
    solve: 'Step 3 only: a large model solves the problem once and writes out its working.',
    build: 'Step 4 only: turn a saved worked solution into a cache of reusable Python functions.',
    eval: 'Step 5 only: test an existing cache on reworded and new-number questions, graded by LLM judges.',
    compile: 'Step 6 only: compile an existing cache to plain Python and check it still gives the same answers.',
    paper: 'Step 1: run the paper’s hand-written functions on our runtime. Only the small model is called.',
    ask: 'Step 2: serve one question through a cache. A cache hit is cheap; a miss falls back to the large source model.',
  }[cmd] || 'One run of the pipeline.';

  // Headline numbers: only the ones this experiment actually has.
  const passAll = ev ? Object.values(ev.modes).reduce((a, m) => [a[0] + m.passed, a[1] + m.total], [0, 0]) : null;
  const head = [];
  if (passAll) head.push(stat('Test questions passed', `${passAll[0]}/${passAll[1]}`,
    `${['phrasal', 'parametric', 'combined'].map(k => frac(ev.modes[k])).join(' · ')}`,
    'Of the new test questions the cache had never seen, how many it answered correctly (per the judges). Split: phrasal · parametric · combined.', 'big'));
  if (c) head.push(
    stat('Build ×', mult(c.build_multiplier), 'build cost ÷ one normal solve', 'Build × = what it cost to build the cache ÷ what one normal solve by the source model costs. Lower is better.', 'big'),
    stat('Savings ×', mult(c.savings_multiplier), 'a cached answer vs a normal solve', 'Savings × = cost of one normal solve ÷ cost of one answer from the cache. Higher is better.', 'big'),
    stat('Break-even', c.break_even_invocations != null ? c.break_even_invocations + ' uses' : '—', 'uses until the cache pays for itself', 'After this many questions answered from the cache, the build cost has been earned back.', 'big'));
  if (d.ask_path) head.push(stat('Serving path', esc(d.ask_path), 'how the question was answered', 'Whether the cache answered (hit) or the source model had to (miss).', 'big'));

  // Review hero
  const rt = reviewTally();
  const pct = n => rt.total ? (n / rt.total * 100).toFixed(1) : 0;
  const hero = rt.total ? `
    <div class="card hero">
      <div>
        <div class="lbl" style="color:var(--accent)">Your check</div>
        <h2>You have reviewed ${rt.reviewed} of ${rt.total} judge decisions</h2>
        <p>LLM judges decided every pass/fail in this run. Don’t just trust them: open each decision, see the evidence they saw, and mark it Correct, Wrong or Unsure.
          ${rt.disagree ? `<b class="fail">You disagree with ${rt.disagree}.</b> ` : ''}${rt.look ? `${rt.look} ${rt.look === 1 ? 'decision needs' : 'decisions need'} a look (a fail, split judges, or numbers that differ).` : ''}</p>
        <div class="progress" role="img" aria-label="${rt.reviewed} of ${rt.total} reviewed"><i class="p-ok" style="width:${pct(rt.correct)}%"></i><i class="p-bad" style="width:${pct(rt.wrong)}%"></i><i class="p-un" style="width:${pct(rt.unsure)}%"></i></div>
        <p class="small" style="margin-top:6px">${rt.correct} correct · ${rt.wrong} wrong · ${rt.unsure} unsure · ${rt.total - rt.reviewed} to go</p>
      </div>
      <div class="stack"><a class="btn primary" href="${base}/review">${rt.reviewed ? 'Continue reviewing' : 'Start reviewing'} →</a></div>
    </div>` : '';

  // Pipeline steps
  const yes = v => v == null ? '—' : v ? '<span class="pass">yes</span>' : '<span class="fail">no</span>';
  const num2 = (label, val, sub = '') => `<div><span>${label}</span><b>${val}</b>${sub ? `<small>${sub}</small>` : ''}</div>`;
  const cost = k => by[k] ? money(by[k].cost) : '—';
  const steps = [];
  const add = (n, stage, title, text, present, nums, tab, linkText) => steps.push({ n, title, text, present, nums, tab, linkText, stage });
  add(1, 'step1', 'Paper functions', 'Runs the paper’s hand-typed functions to check our runtime matches theirs.', !!by.step1,
    [num2('Cost', cost('step1')), num2('Calls', by.step1 ? by.step1.calls : '—')], 'calls', 'See the calls');
  add(2, 'step2', 'Serving', 'Answers a question through a cache, falling back to the big model on a miss.', !!by.step2,
    [num2('Cost', cost('step2')), num2('Calls', by.step2 ? by.step2.calls : '—')], 'calls', 'See the calls');
  add(3, 'source', 'Source solve', 'A big model solves the problem once and writes out its working. This is all the builder learns from.', !!(r.transcript || by.source),
    [num2('Cost', cost('source')), num2('Correct vs answer key', yes(ev && ev.source_correct))], 'source', 'Read the solution');
  add(4, 'build', 'Build', 'The builder turns that working into small reusable Python functions: the cache.', !!(b || by.build),
    [num2('Result', b ? (b.success ? '<span class="pass">success</span>' : '<span class="fail">failed</span>') : '…'), num2('Cost', cost('build')),
     num2('Functions', b ? b.functions : '—'), num2('Attempts', b ? b.attempts : '—', b ? `${b.failed_attempts} failed` : '')], 'build', 'See the build');
  add(5, 'eval', 'Evaluation', 'Tests the cache on reworded and new-number questions it has never seen. LLM judges grade the answers.', !!(ev || r.eval_progress || by.eval),
    [num2('Passed', passAll ? `${passAll[0]}/${passAll[1]}` : '…'), num2('Cost', cost('eval')),
     num2('Cache correct', yes(ev && ev.cache_correct)), num2('Matches source', yes(ev && ev.fidelity))], 'eval', 'See the results');
  add(6, 'compile', 'Compile', 'Turns the cache into plain Python that runs with no model at all, then checks it gives the same answers.', !!(r.compiled || by.compile),
    [num2('Cost', cost('compile')), num2('Python lines', r.compiled ? r.compiled.split('\n').length : '—')], 'compiled', 'View the code');
  const pipelineRun = ['all', 'legacy'].includes(cmd);
  const shown = steps.filter(s => s.present || (pipelineRun && s.n >= 3));
  const stepHtml = shown.map(s => s.present ? `
    <div class="card step">
      <div class="step-head"><span class="step-n">${s.n}</span><h3>${esc(s.title)}</h3></div>
      <p class="step-body">${s.text}</p>
      <div class="step-nums">${s.nums.join('')}</div>
      <a class="go" href="${base}/${s.tab}">${esc(s.linkText)} →</a>
    </div>` : `
    <div class="card step off">
      <div class="step-head"><span class="step-n">${s.n}</span><h3>${esc(s.title)}</h3></div>
      <p class="step-body">Not run in this experiment.</p>
    </div>`).join('');

  // Where the money went
  const stages = Object.entries(by).filter(([, t]) => t.cost > 0);
  const totalCost = stages.reduce((a, [, t]) => a + t.cost, 0) || 1;
  const money2 = stages.length ? `
    <div class="card stack">
      <div class="row"><h3>Where the money went</h3><span class="spacer"></span><a class="small" href="${base}/totals">Full breakdown →</a></div>
      <div class="stackbar" role="img" aria-label="Cost by stage">${stages.map(([k, t], i) => `<i class="c${i % 6}" style="width:${(t.cost / totalCost * 100).toFixed(2)}%" title="${esc(stageName(k))}: ${money(t.cost)}"></i>`).join('')}</div>
      <div class="legend">${stages.map(([k, t], i) => `<span class="c${i % 6}">${esc(stageName(k))} <b>${money(t.cost)}</b> (${(t.cost / totalCost * 100).toFixed(0)}%)</span>`).join('')}</div>
    </div>` : '';

  $('#d-tab').innerHTML = `<div class="stack" style="gap:20px">
    ${explain('What this experiment did', esc(intro),
      'The pipeline: a large model solves a problem <b>once</b> (3) → a builder turns that solution into a reusable <b>cache</b> of Python functions (4) → the cache is <b>tested</b> on questions it has never seen (5) → it is <b>compiled</b> to plain Python (6). Each card below summarises one step; click through for the detail.')}
    ${head.length ? `<div class="stats">${head.join('')}</div>` : ''}
    ${hero}
    ${stepHtml ? `<div class="sec-title"><h3>The pipeline, step by step</h3></div><div class="steps">${stepHtml}</div>` : ''}
    ${money2}
  </div>`;
}

// --- Calls ------------------------------------------------------------------

function filteredCalls() {
  const f = S.callFilter;
  return S.detail.calls.filter(c => (f.stage === 'all' || c.stage === f.stage) && (f.role === 'all' || c.role === f.role) &&
    (!f.hits || f.hits.has(c.i)));
}

function drawCalls() {
  const d = S.detail, f = S.callFilter;
  const host = $('#d-tab');
  if (!host) return;
  const stages = [...new Set(d.calls.map(c => c.stage))];
  const roles = [...new Set(d.calls.map(c => c.role))];
  if (!$('#call-filters')) {
    host.innerHTML = `<div class="stack">
      ${explain('Every model call', 'One row per call to a model, in the order they happened. The bar shows when each call ran; the thin bar on the right shows its share of the cost. Click a row to read the exact prompt and response.',
        'Roles: <b>source</b> solves, <b>builder</b> writes the cache, <b>judges</b> grade, <b>generator</b> writes test questions, <b>target</b> is the small model that uses the cache.')}
      <div class="filters" id="call-filters">
        ${seg('stage', [['all', 'All'], ...stages.map(s => [s, stageName(s)])], 'Stage')}
        <span class="lbl">Role</span>
        <select class="in" id="role-sel"><option value="all">all roles</option>${roles.map(r => `<option>${esc(r)}</option>`).join('')}</select>
        <input class="in" id="call-q" type="search" placeholder="Search prompts, responses, tags" style="flex:1 1 220px" value="${esc(f.q)}">
        <span class="muted small" id="call-count"></span>
      </div>
      <div class="calls-table"><table><thead><tr>
        <th class="n">#</th><th class="n">At</th><th>Timeline</th><th>Stage</th><th>Role</th><th>Step / tag</th><th>Model</th>
        <th class="n">In</th><th class="n">Out</th><th class="n">Cache rd</th><th class="n">Cache wr</th><th class="n">Seconds</th><th class="n">Cost</th><th style="min-width:70px"></th>
      </tr></thead><tbody id="calls-body"></tbody></table></div>
      <p class="muted small">In the call panel: ← → step through calls, Esc closes.</p>
    </div>`;
    host.querySelector('#call-filters').addEventListener('click', ev => {
      const b = ev.target.closest('[data-stage]');
      if (b) { f.stage = b.dataset.stage; drawCalls(); }
    });
    $('#role-sel').value = f.role;
    $('#role-sel').addEventListener('change', ev => { f.role = ev.target.value; drawCalls(); });
    let qTimer;
    $('#call-q').addEventListener('input', ev => {
      clearTimeout(qTimer);
      qTimer = setTimeout(async () => {
        f.q = ev.target.value.trim();
        f.hits = f.q ? new Set(await api(`/api/experiments/${encodeURIComponent(d.id)}/search?q=${encodeURIComponent(f.q)}`)) : null;
        drawCalls();
      }, 250);
    });
    $('#calls-body').addEventListener('click', ev => {
      const tr = ev.target.closest('tr[data-i]');
      if (tr) openCall(+tr.dataset.i);
    });
  }
  $$('#call-filters [data-stage]').forEach(b => b.setAttribute('aria-pressed', b.dataset.stage === f.stage ? 'true' : 'false'));
  const rows = filteredCalls();
  const span = Math.max(1, ...d.calls.map(c => (c.offset || 0) + (c.seconds || 0)));
  const maxCost = Math.max(1e-9, ...d.calls.map(c => c.cost || 0));
  const pid = d.problem;
  const shortTag = c => {
    const t = c.tag || '';
    return pid && t.startsWith(pid + ':') ? t.slice(pid.length + 1) : (t || '—');
  };
  const val = v => v == null ? '<span class="muted">—</span>' : num(v);
  $('#calls-body').innerHTML = rows.map(c => `
    <tr class="click${c.error || c.refused ? ' bad' : ''}${S.openCall === c.i ? ' sel' : ''}" data-i="${c.i}">
      <td class="n muted">${c.i + 1}</td><td class="n muted">+${(c.offset || 0).toFixed(0)}s</td>
      <td><div class="tl"><i style="left:${((c.offset || 0) / span * 100).toFixed(2)}%;width:${((c.seconds || 0) / span * 100).toFixed(2)}%"></i></div></td>
      <td class="small">${esc(stageName(c.stage))}</td><td><span class="tag">${esc(c.role)}</span></td>
      <td class="mono small">${esc(shortTag(c))}${c.error ? ` <span class="fail">${esc(c.error)}</span>` : ''}${c.refused ? ` <span class="fail">refused: ${esc(c.refusal)}</span>` : ''}</td>
      <td class="small muted">${esc((c.model || '').replace('claude-', ''))}</td>
      <td class="n">${val(c.input_tokens)}</td><td class="n">${val(c.output_tokens)}</td>
      <td class="n">${val(c.cache_read_tokens)}</td><td class="n">${val(c.cache_write_tokens)}</td>
      <td class="n">${c.seconds ?? '—'}</td><td class="n">${money(c.cost)}</td>
      <td><div class="bar" style="width:${Math.max(2, (c.cost || 0) / maxCost * 100).toFixed(1)}%"></div></td></tr>`).join('')
    || '<tr><td colspan="14" class="muted">No calls match.</td></tr>';
  const cost = rows.reduce((a, c) => a + (c.cost || 0), 0);
  $('#call-count').textContent = `${rows.length} of ${d.calls.length} calls · ${money(cost)}`;
}

async function openCall(i) {
  const d = S.detail;
  S.openCall = i;
  $$('#calls-body tr').forEach(tr => tr.classList.toggle('sel', +tr.dataset.i === i));
  let c;
  try { c = await api(`/api/experiments/${encodeURIComponent(d.id)}/calls/${i}`); } catch (e) { alert(e.message); return; }
  closeDrawer(true);
  const scrim = document.createElement('div');
  scrim.className = 'scrim'; scrim.id = 'scrim';
  const dr = document.createElement('aside');
  dr.className = 'drawer'; dr.id = 'drawer'; dr.setAttribute('aria-label', 'Call detail');
  const ex = c.extra || {};
  dr.innerHTML = `
    <div class="drawer-head">
      <b class="mono">#${c.i + 1}</b><span class="tag">${esc(c.role)}</span><span class="mono small">${esc(c.tag || stageName(c.stage))}</span>
      <span class="spacer"></span>
      <button class="btn sm" type="button" data-nav="-1" title="Previous (←)">← Prev</button>
      <button class="btn sm" type="button" data-nav="1" title="Next (→)">Next →</button>
      <button class="btn sm" type="button" id="drawer-close" title="Close (Esc)">Close</button>
    </div>
    <div class="drawer-body">
      ${c.error ? `<p class="note err">${esc(c.error)}</p>` : ''}
      ${c.refused ? `<p class="note err">Refused (${esc(c.refusal)}): ${esc(c.refusal_explanation || '')}</p>` : ''}
      <dl class="kv">
        <dt>Stage</dt><dd>${esc(stageName(c.stage))} · step <span class="mono">${esc(c.step)}</span></dd>
        <dt>Model</dt><dd class="mono">${esc(c.model)}${ex.effort ? ` · effort ${esc(ex.effort)}` : ''}${ex.max_tokens ? ` · max_tokens ${esc(ex.max_tokens)}` : ''}${ex.stop_reason ? ` · stop ${esc(ex.stop_reason)}` : ''}</dd>
        <dt>Tokens</dt><dd>${c.input_tokens == null ? '<span class="muted">not logged for this call</span>' :
          `${num(c.input_tokens)} in · ${num(c.output_tokens)} out · ${num(c.cache_read_tokens)} cache read · ${num(c.cache_write_tokens)} cache write`}
          ${ex.thinking_chars != null ? `<br><span class="muted small">${num(ex.visible_text_chars)} visible chars; output tokens include hidden thinking</span>` : ''}</dd>
        <dt>Cost</dt><dd>${money(c.cost)} · ${c.seconds ?? '—'} s · started ${when(c.time)}</dd>
      </dl>
      ${c.system ? block(`<span class="lbl">System prompt</span><span class="muted small">${num(c.system.length)} chars</span>`, `<pre>${esc(c.system)}</pre>`, c.system.length < 1500) : '<p class="muted small">No system prompt.</p>'}
      ${block(`<span class="lbl">Prompt</span><span class="muted small">${num(c.prompt.length)} chars</span>`, `<pre>${esc(c.prompt)}</pre>`, true)}
      ${block(`<span class="lbl">Response</span><span class="muted small">${num((c.response || '').length)} chars</span>`, `<pre>${esc(c.response || '(none)')}</pre>`, true)}
    </div>`;
  document.body.append(scrim, dr);
  scrim.addEventListener('click', () => closeDrawer());
  $('#drawer-close').addEventListener('click', () => closeDrawer());
  $$('[data-nav]', dr).forEach(b => b.addEventListener('click', () => stepCall(+b.dataset.nav)));
  $('#drawer-close').focus();
}

function stepCall(dir) {
  const rows = filteredCalls();
  const k = rows.findIndex(c => c.i === S.openCall);
  const next = rows[k + dir];
  if (next) openCall(next.i);
}

function closeDrawer(keepSel) {
  $('#drawer')?.remove();
  $('#scrim')?.remove();
  if (!keepSel) { S.openCall = null; $$('#calls-body tr.sel').forEach(tr => tr.classList.remove('sel')); }
}

document.addEventListener('keydown', ev => {
  if (!$('#drawer')) return;
  if (ev.key === 'Escape') closeDrawer();
  else if (ev.key === 'ArrowRight' || ev.key === 'j') stepCall(1);
  else if (ev.key === 'ArrowLeft' || ev.key === 'k') stepCall(-1);
});

// --- Totals -----------------------------------------------------------------

function totalsTable(title, groups, nameFn = x => x) {
  const entries = Object.entries(groups);
  const all = entries.reduce((a, [, t]) => a + t.cost, 0) || 1;
  const max = Math.max(1e-9, ...entries.map(([, t]) => t.cost));
  return `<div class="card stack"><h3>${esc(title)}</h3><div class="wide"><table><thead><tr><th>${esc(title.split(' ').pop())}</th>
    <th class="n">Calls</th><th class="n">Input</th><th class="n">Output</th><th class="n">Cache read</th><th class="n">Cache write</th>
    <th class="n">Cost</th><th class="n">Share</th><th style="min-width:100px"></th></tr></thead><tbody>
    ${entries.map(([k, t]) => `<tr><td>${esc(nameFn(k))}</td><td class="n">${t.calls}</td><td class="n">${tok(t, 'input')}</td>
      <td class="n">${tok(t, 'output')}</td><td class="n">${tok(t, 'cache_read')}</td><td class="n">${tok(t, 'cache_write')}</td>
      <td class="n">${money(t.cost)}</td><td class="n">${(t.cost / all * 100).toFixed(1)}%</td>
      <td><div class="bar" style="width:${Math.max(1, t.cost / max * 100).toFixed(1)}%"></div></td></tr>`).join('')}
    </tbody></table></div></div>`;
}

function drawTotals() {
  const d = S.detail;
  const byRoleBuild = d.build && d.build.cost_by_role;
  let legacyNote = '';
  if (d.totals.calls_with_tokens < d.totals.calls && byRoleBuild) {
    const rows = Object.entries(byRoleBuild).map(([r, x]) => `<tr><td>${esc(r)}</td><td class="n">${x.calls}</td><td class="n">${num(x.input)}</td>
      <td class="n">${num(x.output)}</td><td class="n">${num(x.cache_read)}</td><td class="n">${num(x.cache_write)}</td><td class="n">${money(x.cost)}</td></tr>`).join('');
    legacyNote = `<div class="card stack"><h3>Build tokens by role (from build_report.json)</h3>
      <p class="muted small">Per-call tokens were not logged for some calls in this run; the builder's own ledger has exact per-role token totals for the build.</p>
      <div class="wide"><table><thead><tr><th>Role</th><th class="n">Calls</th><th class="n">Input</th><th class="n">Output</th><th class="n">Cache read</th>
      <th class="n">Cache write</th><th class="n">Cost</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }
  $('#d-tab').innerHTML = `<div class="stack">
    ${explain('Where the cost and tokens went', 'The same calls, grouped three ways: by pipeline <b>stage</b> (which step spent it), by <b>role</b> (which model job), and by <b>step</b> (the individual task). Share is the percent of the total cost.')}
    ${totalsTable('Totals by stage', d.by_stage, stageName)}
    ${totalsTable('Totals by role', d.by_role)}
    ${totalsTable('Totals by step', d.by_step)}
    ${legacyNote}</div>`;
}

// --- Reports ----------------------------------------------------------------

// Long evidence (transcripts, code) starts collapsed; short evidence is shown open.
const LONG_SECTIONS = /^(transcript|source transcript|code|worked solution of the original|narration|transcript answer)$/i;

function evidence(sections) {
  const shown = (sections || []).filter(x => x.title || x.text);
  const short = shown.filter(x => !LONG_SECTIONS.test(x.title) && x.text.length < 900);
  const long = shown.filter(x => !short.includes(x));
  return `<div class="evidence">${short.map(x => `<div class="ev"><div class="lbl">${esc(x.title || 'Instruction')}</div><pre>${esc(x.text)}</pre></div>`).join('')}</div>` +
    long.map(x => block(`<span class="lbl">${esc(x.title || 'Instruction')}</span><span class="muted small">${num(x.text.length)} chars</span>`, `<pre>${esc(x.text)}</pre>`)).join('');
}

function needsLook(g) {
  const nc = g.number_check;
  return !g.passed || g.split || (nc && nc.status === 'mismatch') || disagrees(g.key, g.passed);
}

function judgementCard(g, k) {
  const tagRest = (g.tag || '').split(':').slice(1).join(':') || g.tag || '';
  return block(
    `<span class="muted mono">${k + 1}</span><span class="tag">${esc(g.kind)}</span><span class="mono small">${esc(tagRest)}</span>
     ${judgePills(g.verdicts)}${g.split ? '<span class="pill bad">judges split</span>' : ''}${ncPill(g.number_check)}<span data-your="${esc(g.key)}">${yourPill(g.key)}</span>`,
    `${evidence(g.sections)}
     <div class="lbl">What the judges said</div>${judgeNotes(g.verdicts)}
     ${ncLine(g.number_check)}
     ${reviewBox(g.key, g.passed)}
     <p class="small"><a href="#" data-open-call="${g.calls[0]}">Open the raw call →</a></p>`,
    needsLook(g));
}

const KIND_HELP = {
  'Build: function check': 'While building, a judge checks each new function: did it compute what the transcript computes for this step?',
  'Build: end-to-end': 'After all functions exist, the whole program runs on the original question and a judge checks the final answer.',
  'Build: extraction round': 'The small model pulls the values out of the raw question, the program runs, and a judge checks the answer. Repeated N times.',
  'Correctness': 'Is the source answer right (vs the answer key)? Is the cache answer right? Do they agree?',
  'Variant validation': 'Before testing, judges accept or reject each generated test question (is it a fair variant?).',
  'Grading': 'Each test question runs through the cache; judges decide if the cache answer matches the reference.',
};

function drawReview() {
  const all = S.judgements;
  const kinds = [...new Set(all.map(g => g.kind))];
  let kind = 'all', only = 'all';
  const host = $('#d-tab');
  const counts = {
    failed: all.filter(g => !g.passed).length, split: all.filter(g => g.split).length,
    mismatch: all.filter(g => g.number_check && g.number_check.status === 'mismatch').length,
  };
  host.innerHTML = `<div class="stack">
    ${explain('Check the judges yourself',
      'LLM judges made every pass/fail decision in this run. Open a decision to see the exact evidence the judge saw, what it said, and a <b>number check</b> that uses no LLM. Then mark it <b>Correct</b>, <b>Wrong</b> or <b>Unsure</b>: you are judging the <i>judges’ decision</i>, not redoing the problem.',
      'Cards open on their own when they need a look: a fail, judges that disagree, numbers that differ, or your own disagreement. Your reviews save instantly with this experiment.')}
    <div class="stats compact">
      ${stat('Judge decisions', all.length, `${all.reduce((a, g) => a + g.verdicts.length, 0)} judge calls`, 'Each decision is made by a small committee of LLM judges.')}
      ${stat('Judged as fail', counts.failed, '', 'Decisions where the judges said the answer or function was wrong.')}
      ${stat('Judges split', counts.split, '', 'Decisions where the judges did not agree with each other.')}
      ${stat('Number check differs', counts.mismatch, 'worth a look', 'A simple no-LLM check compares the numbers in the answer with the reference. A mismatch is a hint, not proof.')}
      ${stat('Reviewed by you', '<span id="rv-count"></span>', '', 'How many decisions you have marked Correct, Wrong or Unsure.')}
    </div>
    <div class="filters">
      ${seg('kind', ['all', ...kinds].map(x => [x, x === 'all' ? 'All kinds' : x]), 'Kind')}
      ${seg('only', [['all', 'Everything'], ['look', 'Needs a look'], ['todo', 'Not reviewed'], ['disagree', 'You disagree']], 'Show')}
    </div>
    <p class="muted small" id="kind-help"></p>
    <div class="stack" id="jlist"></div></div>`;
  const draw = () => {
    $$('[data-kind]', host).forEach(b => b.setAttribute('aria-pressed', b.dataset.kind === kind ? 'true' : 'false'));
    $$('[data-only]', host).forEach(b => b.setAttribute('aria-pressed', b.dataset.only === only ? 'true' : 'false'));
    $('#kind-help').textContent = KIND_HELP[kind] || 'Pick a kind to see what that check is for.';
    $('#rv-count').textContent = `${all.filter(g => (S.reviews[g.key] || {}).verdict).length}/${all.length}`;
    const rows = all.map((g, k) => ({ g, k })).filter(({ g }) => (kind === 'all' || g.kind === kind) &&
      (only === 'all' || (only === 'look' && needsLook(g)) || (only === 'todo' && !(S.reviews[g.key] || {}).verdict) ||
       (only === 'disagree' && disagrees(g.key, g.passed))));
    $('#jlist').innerHTML = rows.map(({ g, k }) => judgementCard(g, k)).join('') || '<p class="muted">Nothing matches.</p>';
  };
  host.addEventListener('click', ev => {
    const a = ev.target.closest('[data-kind]'), b = ev.target.closest('[data-only]'), c = ev.target.closest('[data-open-call]');
    if (a) { kind = a.dataset.kind; draw(); }
    if (b) { only = b.dataset.only; draw(); }
    if (c) { ev.preventDefault(); openCall(+c.dataset.openCall); }
  });
  draw();
}

// Find the judge decision that graded a given question (works for old runs too).
function gradingGroup(question, stepPrefix) {
  return S.judgements.find(g => g.role === 'test_judge' && (g.step || '').startsWith(stepPrefix) &&
    (g.sections.find(x => x.title === 'Question') || {}).text === question.trim());
}

function drawSource() {
  const t = S.detailReports.transcript;
  const golden = S.detail.golden;
  const ev = S.detailReports.eval || {};
  const sc = (ev.correctness_checks || {}).source_correct;
  const g = S.judgements.find(x => x.step === 's_acc');
  $('#d-tab').innerHTML = `<div class="stack">
    ${explain('Step 3 · The source solve', 'A large model solves the problem <b>once</b> and writes out its working. That written solution (the <i>transcript</i>) is the only thing the builder learns from, so if it is wrong, the cache will be wrong too.')}
    <dl class="kv card"><dt>Problem</dt><dd class="mono">${esc(t.problem_id)}</dd><dt>Model</dt><dd class="mono">${esc(t.model)}</dd>
      <dt>Cost</dt><dd>${money(t.cost)}</dd></dl>
    <div class="evidence">
      <div class="ev"><div class="lbl">Question</div><pre>${esc(t.question)}</pre></div>
      ${golden ? `<div class="ev"><div class="lbl">Answer key (golden label, from the dataset)</div><pre>${esc(golden)}</pre></div>` : ''}
    </div>
    ${sc ? `<div class="card stack"><div class="row"><b>Was the source right?</b>${judgePills(sc.verdicts)}${ncPill({ ...sc.number_check, against: 'golden label' })}</div>${judgeNotes(sc.verdicts)}${ncLine({ ...sc.number_check, against: 'golden label' })}</div>` : ''}
    ${g ? reviewBox(g.key, g.passed, 'Is the source answer correct per the answer key?') : ''}
    ${block('<b>Visible worked solution</b><span class="muted small">what the builder reads</span>', `<pre>${esc(t.solution)}</pre>`, true)}
    ${block('<b>Thinking summary</b><span class="muted small">for humans only, not given to the builder</span>', `<pre>${esc(t.thinking_summary || '(none returned)')}</pre>`)}
  </div>`;
}

function kvPre(obj) {
  if (!obj || !Object.keys(obj).length) return '<p class="muted small">(none)</p>';
  return `<table class="kvt">${Object.entries(obj).map(([k, v]) => `<tr><th>${esc(k)}</th><td class="mono">${esc(v)}</td></tr>`).join('')}</table>`;
}

function drawBuild() {
  const b = S.detailReports.build;
  const m = S.detail.build || {};
  const pid = S.detail.problem;
  const byFn = {};
  b.attempts.forEach(a => (byFn[a.function_id] = byFn[a.function_id] || []).push(a));
  // Pair each attempt that reached the judge with that judge decision, in order.
  const jByFn = {};
  S.judgements.filter(g => g.role === 'build_judge').forEach(g => {
    const fid = (g.tag || '').split(':').slice(2).join(':');
    (jByFn[fid] = jByFn[fid] || []).push(g);
  });
  const reachedJudge = a => a.failed_at ? a.failed_at === 'judge' || a.passed : a.passed || (a.feedback || '').startsWith('The judge rejected');
  Object.entries(byFn).forEach(([fid, list]) => { const js = (jByFn[fid] || []).slice(); list.forEach(a => { if (reachedJudge(a)) a._g = js.shift(); }); });
  const roleRows = Object.entries(b.cost_by_role || {}).map(([r, x]) => `<tr><td>${esc(r)}</td><td class="n">${x.calls}</td>
    <td class="n">${num(x.input)}</td><td class="n">${num(x.output)}</td><td class="n">${num(x.cache_read)}</td><td class="n">${num(x.cache_write)}</td>
    <td class="n">${money(x.cost)}</td></tr>`).join('');
  const failedAt = a => a.passed ? '' : ({ static: 'failed the code check', run: 'crashed when run', judge: 'rejected by the judge' }[a.failed_at] || 'failed');
  const attemptBody = a => `
    ${a.description ? `<p><b>What it should do:</b> ${esc(a.description)}</p>` : ''}
    ${a.test_inputs || a.output ? `<div class="evidence">
      <div class="ev"><div class="lbl">Test inputs</div>${kvPre(a.test_inputs)}</div>
      <div class="ev"><div class="lbl">Expected (builder's reading of the transcript)</div><pre>${esc(a.expected_result || '—')}</pre>${a.transcript_fragment ? `<div class="lbl">Transcript step</div><pre>${esc(a.transcript_fragment)}</pre>` : ''}</div>
      <div class="ev"><div class="lbl">Actual output</div>${a.output ? kvPre(a.output) : '<p class="muted small">(did not produce output)</p>'}</div></div>` : ''}
    ${a.verdict ? `<div class="verdict ${a.verdict.passed ? 'ok' : 'bad'}"><b>Judge · ${a.verdict.passed ? 'pass' : 'fail'}</b><p>${esc(a.verdict.critique)}</p></div>` : ''}
    ${!a.passed ? `<div class="lbl">Feedback sent back to the builder</div><pre>${esc(a.feedback)}</pre>` : ''}
    ${a.narration ? block('<span class="lbl">Narration</span>', `<pre>${esc(a.narration)}</pre>`) : ''}
    ${block('<span class="lbl">Code</span>', `<pre class="code">${esc(a.code)}</pre>`, !a.passed)}
    ${a._g ? reviewBox(a._g.key, a._g.passed, 'Did the judge decide this attempt correctly?') : ''}`;
  const finalCheck = (title, e, g) => block(`${title} ${pill(e.passed)}${ncPill(e.number_check ? { ...e.number_check, against: 'golden label' } : g && g.number_check)}${g ? `<span data-your="${esc(g.key)}">${yourPill(g.key)}</span>` : ''}`,
    `<div class="evidence">
      ${e.inputs ? `<div class="ev"><div class="lbl">Inputs (original values)</div>${kvPre(e.inputs)}</div>` : ''}
      ${e.extracted ? `<div class="ev"><div class="lbl">Extracted by the small model</div>${kvPre(e.extracted)}</div>` : ''}
      ${e.expected ? `<div class="ev"><div class="lbl">Expected values</div>${kvPre(e.expected)}</div>` : ''}
      <div class="ev"><div class="lbl">Program output</div>${kvPre(e.output)}</div>
      ${S.detail.golden ? `<div class="ev"><div class="lbl">Answer key</div><pre>${esc(S.detail.golden)}</pre></div>` : ''}</div>
     <div class="verdict ${e.passed ? 'ok' : 'bad'}"><b>Judge · ${e.passed ? 'pass' : 'fail'}</b><p>${esc(e.critique)}</p></div>
     ${ncLine(e.number_check ? { ...e.number_check, against: 'golden label' } : g && g.number_check)}
     ${g ? reviewBox(g.key, g.passed) : ''}`, !e.passed);
  const e2eGroups = S.judgements.filter(g => (g.tag || '').endsWith(':judge:end_to_end'));
  const exGroups = r => S.judgements.find(g => (g.tag || '').endsWith(`:judge:extraction${r}`));
  $('#d-tab').innerHTML = `<div class="stack">
    ${explain('Step 4 · The build', 'The builder reads the transcript and turns it into small Python functions (the cache). <b>Phases 1–4</b> plan it: split question from answer, list the reasoning steps, find the values that can change (the <i>parameters</i>), and plan the functions.',
      '<b>Phase 5</b> writes each function: code check → run on test inputs → a judge compares the output with the transcript → retry with feedback if it fails. Two final checks then run the whole program.')}
    <div class="stats compact">
      ${stat('Result', b.success ? '<span class="pass">success</span>' : '<span class="fail">failed</span>', '', 'Whether the builder produced a cache that passed its own final checks.')}
      ${stat('Builder cost', money(b.cost), '', 'Everything spent on building: builder, judges and extractor calls.')}
      ${stat('Functions', m.functions ?? Object.keys(byFn).length, '', 'How many reusable Python functions the cache contains.')}
      ${stat('Phase-5 attempts', b.attempts.length, `${b.attempts.filter(a => !a.passed).length} failed`, 'Each function may take several tries; failed tries are fed back to the builder.')}
      ${stat('End-to-end', b.end_to_end.map(e => e.passed ? '✓' : '✗').join(' ') || '—', '', 'The whole program is run on the original question and a judge checks the final answer.')}
      ${stat('Extraction rounds', b.extraction_rounds.map(e => e.passed ? '✓' : '✗').join(' ') || '—', '', 'The small model pulls values from the raw question, the program runs, and a judge checks. Repeated a few times.')}
    </div>
    <h3>Phases 1–4: the plan</h3>
    ${Object.entries(b.phases || {}).map(([k, v]) => block(esc(k), `<pre class="code">${esc(JSON.stringify(v, null, 2))}</pre>`)).join('')}
    <h3>Phase 5: every attempt, by function</h3>
    ${Object.entries(byFn).map(([fn, list]) => block(
      `<span class="tag">${esc(fn)}</span>${list.length} attempt${list.length > 1 ? 's' : ''} ${pill(list[list.length - 1].passed)}`,
      list.map(a => block(`attempt ${a.attempt} ${pill(a.passed)} <span class="muted small">${esc(failedAt(a))}</span>${a._g ? `<span data-your="${esc(a._g.key)}">${yourPill(a._g.key)}</span>` : ''}`,
        attemptBody(a), !a.passed)).join(''))).join('')}
    <h3>Final checks on the whole program</h3>
    ${b.end_to_end.map((e, k) => finalCheck(`End-to-end check ${k + 1}`, e, e2eGroups[k])).join('')}
    ${b.extraction_rounds.map(r => finalCheck(`Extraction round ${r.round}`, r, exGroups(r.round))).join('')}
    ${b.reused && Object.keys(b.reused).length ? `<div class="card"><div class="lbl">Reused functions</div><pre>${esc(JSON.stringify(b.reused, null, 1))}</pre></div>` : ''}
  </div>`;
}

function drawEval() {
  const r = S.detailReports;
  const ev = r.eval || (r.eval_progress && r.eval_progress.report);
  if (!ev) { $('#d-tab').innerHTML = '<p class="muted">No eval report.</p>'; return; }
  const partial = !r.eval;
  const c = ev.costs || {};
  const modes = {};
  ev.variants.forEach(v => { const m = modes[v.mode] = modes[v.mode] || [0, 0]; m[1]++; if (v.outcome === 'pass') m[0]++; });
  let mode = 'all', outcome = 'all';
  const host = $('#d-tab');
  const cc = ev.correctness_checks || {};
  const corrRow = (key, label, step, val) => {
    const x = cc[key], g = S.judgements.find(j => j.step === step);
    const verdicts = x ? x.verdicts : g ? g.verdicts : [];
    const nc = x ? (x.number_check && { ...x.number_check, against: 'golden label' }) : g && g.number_check;
    return block(`<b>${esc(label)}</b> ${pill(val, 'yes', 'no')}${judgePills(verdicts)}${ncPill(nc)}${g ? `<span data-your="${esc(g.key)}">${yourPill(g.key)}</span>` : ''}`,
      `${x && x.evidence ? `<div class="evidence">${Object.entries(x.evidence).map(([k, v]) => `<div class="ev"><div class="lbl">${esc(k)}</div><pre>${esc(v)}</pre></div>`).join('')}</div>` : g ? evidence(g.sections) : ''}
       ${judgeNotes(verdicts)}${ncLine(nc)}${g ? reviewBox(g.key, g.passed) : ''}`, val === false);
  };
  const cands = ev.candidates || [];
  host.innerHTML = `<div class="stack">
    ${explain('Step 5 · The evaluation', 'First, three correctness checks on the original question. Then three kinds of test questions the cache has never seen: <b>phrasal</b> (reworded, same numbers), <b>parametric</b> (same wording, new numbers) and <b>combined</b> (both).',
      'Each test question is written by the generator model, accepted by two judges, run through the cache, then graded by two more judges. For new numbers there is no answer key: the reference is the generator model’s own solution, so it can be wrong too. Check those first.')}
    ${partial ? '<p class="note">Partial: only eval_progress.json was saved (the run did not finish).</p>' : ''}
    <div class="stats compact">
      ${stat('S Acc / C Acc / Fidelity', `${yn(ev.source_correct)} / ${yn(ev.cache_correct)} / ${yn(ev.fidelity)}`, 'source right · cache right · they agree', 'S Acc: is the source answer right? C Acc: is the cache answer right? Fidelity: does the cache give the same answer as the source?')}
      ${Object.entries(modes).map(([k, [p, t]]) => stat(k, `${p}/${t}`, 'passed', { phrasal: 'Reworded questions with the same numbers.', parametric: 'Same wording, new numbers.', combined: 'Reworded and new numbers.' }[k.replace(/-.*/, '')] || 'Test questions of this kind.')).join('')}
      ${stat('Source cost', money(c.source_cost), '', 'What one normal solve by the source model cost.')}
      ${stat('Builder cost', money(c.builder_cost), '', 'What it cost to build the cache.')}
      ${stat('Cache cost per use', money(c.cache_cost_per_invocation), `${c.invocations_measured ?? '—'} invocations measured`, 'Average cost of answering one question from the cache.')}
      ${stat('Build ×', mult(c.build_multiplier), 'build cost ÷ one solve', 'How many normal solves the build cost is worth. Lower is better.')}
      ${stat('Savings ×', mult(c.savings_multiplier), `break-even ${c.break_even_invocations ?? '—'} uses`, 'One normal solve ÷ one cached answer. Higher is better. Break-even is how many uses repay the build.')}
    </div>
    ${Object.keys(ev.errors || {}).length ? `<div class="note err"><b>Errors</b><pre>${esc(JSON.stringify(ev.errors, null, 1))}</pre></div>` : ''}
    <h3>Correctness on the original question</h3>
    ${corrRow('source_correct', 'Source answer correct (vs answer key)', 's_acc', ev.source_correct)}
    ${corrRow('cache_correct', 'Cache answer correct (vs answer key)', 'c_acc', ev.cache_correct)}
    ${corrRow('fidelity', 'Cache agrees with the source', 'fidelity', ev.fidelity)}
    <h3>Test questions</h3>
    <div class="filters">
      ${seg('mode', ['all', ...Object.keys(modes)].map(m => [m, m]), 'Mode')}
      ${seg('outcome', ['all', ...new Set(ev.variants.map(v => v.outcome))].map(o => [o, o]), 'Outcome')}
    </div>
    <div class="stack" id="variants"></div>
    <h3>Generated candidates ${cands.length ? `<span class="muted small">${cands.filter(x => x.accepted).length} accepted · ${cands.filter(x => !x.accepted).length} rejected</span>` : ''}</h3>
    ${cands.length ? cands.map((x, k) => block(`<span class="tag">${esc(x.mode)}</span>${pill(x.accepted, 'accepted', 'rejected')}${judgePills(x.verdicts)}<span>${esc(x.question.slice(0, 120))}${x.question.length > 120 ? '…' : ''}</span>`,
      `<div class="evidence"><div class="ev"><div class="lbl">Based on</div><pre>${esc(x.base_question)}</pre></div><div class="ev"><div class="lbl">Candidate</div><pre>${esc(x.question)}</pre></div>
       ${x.final_answer ? `<div class="ev"><div class="lbl">Generator's answer for it</div><pre>${esc(x.final_answer)}</pre></div>` : ''}</div>
       ${x.new_solution ? block('<span class="lbl">Generator\'s worked solution</span>', `<pre>${esc(x.new_solution)}</pre>`) : ''}
       ${judgeNotes(x.verdicts)}`)).join('')
      : `<p class="muted small">Not recorded for this run (older runs only kept the rejected count: ${Object.entries(ev.rejected_candidates || {}).map(([k, v]) => `${esc(k)} ${v}`).join(' · ') || '—'}). See the Review tab for every validation verdict.</p>`}
  </div>`;
  const draw = () => {
    $$('[data-mode]', host).forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === mode ? 'true' : 'false'));
    $$('[data-outcome]', host).forEach(b => b.setAttribute('aria-pressed', b.dataset.outcome === outcome ? 'true' : 'false'));
    $('#variants').innerHTML = ev.variants.map((v, k) => ({ v, k })).filter(({ v }) => (mode === 'all' || v.mode === mode) && (outcome === 'all' || v.outcome === outcome))
      .map(({ v, k }) => {
        const g = gradingGroup(v.question, `test:${v.mode}`);
        const nc = v.number_check && v.number_check.status ? { ...v.number_check, against: v.reference_source === 'golden label' ? 'answer key' : 'generator\'s answer' } : g && g.number_check;
        const refSrc = v.reference_source || (v.mode === 'phrasal' ? 'golden label' : 'generator model');
        return block(`<span class="muted mono">${k + 1}</span><span class="tag">${esc(v.mode)}</span>${pill(v.outcome === 'pass', 'PASS', v.outcome.toUpperCase())}${ncPill(nc)}${g ? `<span data-your="${esc(g.key)}">${yourPill(g.key)}</span>` : ''}<span>${esc(v.question.slice(0, 120))}${v.question.length > 120 ? '…' : ''}</span>`,
          `<div class="ev"><div class="lbl">Test question</div><pre>${esc(v.question)}</pre></div>
           <div class="evidence">
             <div class="ev"><div class="lbl">Cache answer</div><pre>${esc(v.cache_answer || '(none)')}</pre></div>
             <div class="ev"><div class="lbl">Reference answer · from the ${esc(refSrc)}</div><pre>${esc(v.reference_answer || v.reference.split('\n\n')[0])}</pre>
               ${refSrc !== 'golden label' ? '<p class="warn small">Written by the generator model, not an answer key. It can be wrong.</p>' : ''}</div></div>
           ${v.parameters && Object.keys(v.parameters).length ? `<div class="evidence"><div class="ev"><div class="lbl">Values the small model extracted</div>${kvPre(v.parameters)}</div>
             <div class="ev"><div class="lbl">Matched cache entry</div><pre>${esc(v.entry_point || '')}</pre><p class="muted small">similarity ${v.similarity != null ? v.similarity.toFixed(3) : '—'}</p></div></div>` : ''}
           <div class="lbl">Grading judges</div>${v.verdicts && v.verdicts.length ? judgeNotes(v.verdicts) : g ? judgeNotes(g.verdicts) : `<pre>${esc(v.detail)}</pre>`}
           ${ncLine(nc)}
           ${v.reference.includes('\n\n') ? block('<span class="lbl">Reference worked solution</span>', `<pre>${esc(v.reference)}</pre>`) : ''}
           ${v.narration ? block('<span class="lbl">Cache narration (how it got the answer)</span>', `<pre>${esc(v.narration)}</pre>`) : ''}
           ${g ? reviewBox(g.key, g.passed, 'Is the cache answer actually correct?') : ''}`, v.outcome !== 'pass' || (nc && nc.status === 'mismatch'));
      }).join('') || '<p class="muted">No variants match.</p>';
  };
  host.addEventListener('click', ev2 => {
    const m = ev2.target.closest('[data-mode]'), o = ev2.target.closest('[data-outcome]');
    if (m) { mode = m.dataset.mode; draw(); }
    if (o) { outcome = o.dataset.outcome; draw(); }
  });
  draw();
}

function drawCompiled() {
  $('#d-tab').innerHTML = `<div class="stack">
    ${explain('Step 6 · Compiled to plain Python', 'The cache turned into an ordinary Python file: no model, no embeddings, no API calls. It was checked against the live cache to make sure both give the same answers.')}
    <pre class="code">${esc(S.detailReports.compiled)}</pre></div>`;
}

function drawOutput() {
  const d = S.detail;
  $('#d-tab').innerHTML = `<div class="stack">
    ${explain('Console output', 'The raw text the command printed while it ran, exactly as it would appear in a terminal. Useful when something fails or you want the full log.')}
    <pre class="console tall" id="d-console"></pre></div>`;
  let offset = 0, text = '';
  const tick = async () => {
    const con = $('#d-console');
    if (!con) return;
    const o = await api(`/api/experiments/${encodeURIComponent(d.id)}/output?offset=${offset}`).catch(() => null);
    if (!o) return;
    if (o.text) {
      const atBottom = con.scrollHeight - con.scrollTop - con.clientHeight < 40;
      text += o.text; offset = o.offset; con.textContent = text;
      if (atBottom || offset === o.text.length) con.scrollTop = con.scrollHeight;
    }
  };
  tick();
  if (d.status === 'running') every(1000, tick);
}

function drawSettings() {
  const d = S.detail, m = S.meta;
  const rows = Object.keys(d.settings || {}).map(k => {
    const v = d.settings[k], def = m.settings[k];
    const over = (d.overrides || {})[k] != null;
    return `<tr${over ? ' class="sel"' : ''}><td class="mono">${esc(k)}</td><td class="mono">${esc(v)}</td>
      <td class="mono muted">${esc(def)}</td><td>${over ? '<span class="tag">override</span>' : String(v) !== String(def) ? '<span class="warn small">differs from current default</span>' : ''}</td></tr>`;
  }).join('');
  $('#d-tab').innerHTML = `<div class="stack">
    ${explain('Settings used', 'The models, effort levels and thresholds this run used, and how they differ from today’s defaults in <span class="mono">arc/config.py</span>. Overrides were set for this run only.')}
    ${d.settings_recorded ? '' : '<p class="note">Settings were not recorded for this run; shown are the current defaults from arc/config.py.</p>'}
    <dl class="kv card"><dt>Command</dt><dd class="mono">${esc(d.legacy ? 'several, by hand' : cmdLine(d))}</dd>
      <dt>Args</dt><dd class="mono">${esc(JSON.stringify(d.args))}</dd>
      <dt>Started from</dt><dd>${esc(d.legacy ? 'command line, before experiment IDs' : d.source === 'ui' ? 'this UI' : 'command line')}</dd>
      <dt>Saved files</dt><dd class="mono">${esc((d.files || []).join(', ') || '—')}</dd></dl>
    <div class="wide card"><table><thead><tr><th>Setting</th><th>This run</th><th>Current default</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

async function viewCompare(ids) {
  app.innerHTML = '<p class="muted">Loading…</p>';
  if (!S.meta) await refreshMeta();
  S.experiments = await api('/api/experiments');
  const picker = (chosen) => `<div class="card stack"><h3>Pick experiments to compare</h3>
    <div class="stack">${S.experiments.map(e => `<label class="check"><input type="checkbox" value="${esc(e.id)}"${chosen.includes(e.id) ? ' checked' : ''}>
      <span><span class="mono">${esc(e.id)}</span> <span class="muted small">${esc((e.argv || []).join(' '))} · ${money(e.totals.cost)}</span></span></label>`).join('')}</div>
    <div><button class="btn primary" type="button" id="cmp-go">Compare</button></div></div>`;
  if (ids.length < 2) {
    app.innerHTML = `<div class="view">${head('Compare', 'Put two or more experiments side by side: cost, accuracy, build effort and the settings that differ. Select at least two.')}${picker(ids)}</div>`;
    $('#cmp-go').addEventListener('click', () => {
      const sel = $$('.card input[type=checkbox]:checked').map(x => x.value);
      if (sel.length >= 2) location.hash = 'compare/' + sel.map(encodeURIComponent).join(',');
    });
    return;
  }
  let ds;
  try { ds = await Promise.all(ids.map(id => api('/api/experiments/' + encodeURIComponent(id)))); }
  catch (e) { app.innerHTML = `<p class="note err">${esc(e.message)}</p>`; return; }
  let showAll = false;
  const draw = () => {
    const rows = [];
    const sec = title => rows.push(`<tr class="sec"><th class="metric" colspan="${ds.length + 1}">${esc(title)}</th></tr>`);
    // kind: 'low' or 'high' marks the best numeric cell in bold.
    const row = (name, vals, fmt = x => esc(x ?? '—'), kind = null) => {
      const nums = vals.map(v => typeof v === 'number' && !isNaN(v) ? v : null);
      let best = null;
      if (kind && nums.filter(v => v != null).length > 1) best = kind === 'low' ? Math.min(...nums.filter(v => v != null)) : Math.max(...nums.filter(v => v != null));
      rows.push(`<tr><th class="metric">${esc(name)}</th>${vals.map((v, k) => `<td class="${nums[k] != null && nums[k] === best ? 'best' : ''}">${fmt(v)}</td>`).join('')}</tr>`);
    };
    sec('Overview');
    rows.push(`<tr><th class="metric">Experiment</th>${ds.map(d => `<td><a class="mono" href="#exp/${encodeURIComponent(d.id)}">${esc(d.id)}</a>${d.label ? `<br><span class="muted small">${esc(d.label)}</span>` : ''}</td>`).join('')}</tr>`);
    row('Command', ds.map(d => d.legacy ? 'legacy (steps 1–6)' : (d.argv || []).join(' ')), x => `<span class="mono small">${esc(x)}</span>`);
    row('Status', ds.map(d => d.status), status);
    row('Started', ds.map(d => d.started), when);
    row('Duration', ds.map(d => d.seconds), secs, 'low');
    row('Profile', ds.map(d => d.profile), x => esc(x ?? '—'));
    ['SOURCE_MODEL', 'BUILDER_MODEL', 'JUDGE_MODEL', 'GENERATOR_MODEL', 'TARGET_MODEL'].forEach(k =>
      row(k.replace('_MODEL', '').toLowerCase() + ' model', ds.map(d => (d.models || {})[k]), x => `<span class="mono small">${esc((x || '—').replace('claude-', ''))}</span>`));
    row('Reviewed by you (disagree)', ds.map(d => d.reviews ? `${d.reviews.reviewed} (${d.reviews.disagree})` : '—'));
    sec('Totals');
    row('Total cost', ds.map(d => d.totals.cost), money, 'low');
    row('Calls', ds.map(d => d.totals.calls), num);
    row('Input tokens', ds.map(d => d.totals.calls_with_tokens ? d.totals.input : null), num, 'low');
    row('Output tokens', ds.map(d => d.totals.calls_with_tokens ? d.totals.output : null), num, 'low');
    row('Cache read tokens', ds.map(d => d.totals.calls_with_tokens ? d.totals.cache_read : null), num);
    row('Cache write tokens', ds.map(d => d.totals.calls_with_tokens ? d.totals.cache_write : null), num);
    row('Errors / refusals', ds.map(d => `${d.totals.errors} / ${d.totals.refusals}`));
    sec('Cost by stage');
    const stages = [...new Set(ds.flatMap(d => Object.keys(d.by_stage)))];
    stages.forEach(s => row(stageName(s), ds.map(d => d.by_stage[s] ? d.by_stage[s].cost : null), x => x == null ? '—' : money(x), 'low'));
    sec('Cost and tokens by role');
    const roles = [...new Set(ds.flatMap(d => Object.keys(d.by_role)))];
    roles.forEach(r => row(r, ds.map(d => d.by_role[r] || null), t => t == null ? '—' :
      `${money(t.cost)} <span class="muted small">· ${t.calls} calls${t.calls_with_tokens ? ` · ${num(t.input + t.cache_read + t.cache_write)} in / ${num(t.output)} out` : ''}</span>`));
    if (ds.some(d => d.build)) {
      sec('Build');
      row('Build result', ds.map(d => d.build ? d.build.success : null), yn);
      row('Builder cost', ds.map(d => d.build ? d.build.cost : null), money, 'low');
      row('Functions', ds.map(d => d.build ? d.build.functions : null), num);
      row('Phase-5 attempts (failed)', ds.map(d => d.build ? `${d.build.attempts} (${d.build.failed_attempts})` : null));
    }
    if (ds.some(d => d.eval)) {
      sec('Evaluation');
      row('S Acc / C Acc / Fidelity', ds.map(d => d.eval ? `${yn(d.eval.source_correct)} / ${yn(d.eval.cache_correct)} / ${yn(d.eval.fidelity)}` : '—'), x => x);
      ['phrasal', 'parametric', 'combined'].forEach(m => row(m, ds.map(d => d.eval ? frac(d.eval.modes[m]) : null)));
      row('Pass rate', ds.map(d => {
        if (!d.eval) return null;
        const ms = Object.values(d.eval.modes), t = ms.reduce((a, x) => a + x.total, 0);
        return t ? ms.reduce((a, x) => a + x.passed, 0) / t : null;
      }), x => x == null ? '—' : (x * 100).toFixed(0) + '%', 'high');
      row('Source cost', ds.map(d => d.eval ? d.eval.costs.source_cost : null), money);
      row('Cache cost per use', ds.map(d => d.eval ? d.eval.costs.cache_cost_per_invocation : null), money, 'low');
      row('Build multiplier', ds.map(d => d.eval ? d.eval.costs.build_multiplier : null), mult, 'low');
      row('Savings multiplier', ds.map(d => d.eval ? d.eval.costs.savings_multiplier : null), mult, 'high');
      row('Break-even uses', ds.map(d => d.eval ? d.eval.costs.break_even_invocations : null), num, 'low');
    }
    if (ds.some(d => d.ask_path)) { sec('Serving'); row('Path', ds.map(d => d.ask_path)); }
    sec('Settings');
    const names = S.meta.setting_names;
    let shown = 0;
    names.forEach(n => {
      const vals = ds.map(d => d.settings[n]);
      const differs = new Set(vals.map(String)).size > 1;
      if (!differs && !showAll) return;
      shown++;
      rows.push(`<tr><th class="metric mono small">${esc(n)}</th>${ds.map((d, k) => `<td class="mono small${differs ? ' diff' : ''}">${esc(vals[k] ?? '—')}${(d.overrides || {})[n] != null ? ' <span class="tag">override</span>' : ''}${!d.settings_recorded ? ' <span class="muted">(assumed)</span>' : ''}</td>`).join('')}</tr>`);
    });
    if (!shown) rows.push(`<tr><td colspan="${ds.length + 1}" class="muted small">All settings are the same.</td></tr>`);
    app.innerHTML = `<div class="view">
      ${head(`Compare ${ds.length} experiments`, 'Each column is one experiment. The best value in a row is <b class="pass">bold green</b>; settings that differ between runs are tinted. Use it to see what a model or setting change did to cost and accuracy.',
        `<label class="check small"><input type="checkbox" id="show-all"${showAll ? ' checked' : ''}> show all settings</label><a class="btn sm" href="#compare">Change selection</a>`)}
      <div class="wide card"><table class="cmp">${rows.join('')}</table></div>
      </div>`;
    $('#show-all').addEventListener('change', ev => { showAll = ev.target.checked; draw(); });
  };
  draw();
}

// ---------------------------------------------------------------------------

refreshMeta().then(route).catch(e => { app.innerHTML = `<p class="note err">Cannot reach the server: ${esc(e.message)}</p>`; });
