"""Render one page showing everything a test run did, read straight from the run files.

    python -m arc.walkthrough gas_law   ->  runs/gas_law/walkthrough.html

Sources: runs/<id>/transcript.json, build_report.json, eval_report.json,
caches/<id>/ (our builder's entry), caches/paper_<id>/ (the paper's entry),
caches/master/compiled_<id>.py, and runs/llm_log.jsonl (every model call).
"""

from __future__ import annotations

import html
import json
import sys
from collections import defaultdict

from . import config


def esc(x) -> str:
    return html.escape(str(x))


def money(x: float) -> str:
    return f"${x:.4f}"


STYLE = """
<title>ARC Gas-Law Test Run</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap">
<style>
/* Layout: a lab-notebook column; summary first, then each stage of the run in order,
   with every artifact expandable in place. */
:root{
  --bg:#f6f7f9; --surface:#ffffff; --ink:#17191d; --muted:#5d6470; --rule:#dde1e7;
  --code:#eef1f5; --accent:#2f57c7; --accent-soft:#e5ecfb; --pass:#1d7a46; --fail:#b42318;
  --warn:#9a6700;
  --display:"Space Grotesk",ui-sans-serif,system-ui,sans-serif;
  --body:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#121419; --surface:#1a1d24; --ink:#e7e9ee; --muted:#9aa2b1; --rule:#2c313b;
  --code:#20242c; --accent:#8fa9ff; --accent-soft:#232c45; --pass:#5ccf8b; --fail:#ff8a7a;
  --warn:#e3b341; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121419; --surface:#1a1d24; --ink:#e7e9ee; --muted:#9aa2b1; --rule:#2c313b;
  --code:#20242c; --accent:#8fa9ff; --accent-soft:#232c45; --pass:#5ccf8b; --fail:#ff8a7a;
  --warn:#e3b341; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.6 var(--body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1040px;margin:0 auto;padding-block:32px 96px;display:flex;flex-direction:column;gap:14px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;line-height:1.25;margin:0}
h1{font-size:30px}h2{font-size:22px;padding-top:26px;border-top:1px solid var(--rule);margin-top:18px}
h3{font-size:16.5px;margin-top:10px}
p{margin:0;max-width:72ch}.muted{color:var(--muted)}
a{color:var(--accent)}
code,pre,.mono{font-family:var(--mono)}
pre{background:var(--code);border-radius:8px;padding:12px 14px;margin:0;overflow-x:auto;
  font-size:12.5px;line-height:1.55;white-space:pre;tab-size:4}
pre.wrap{white-space:pre-wrap;word-break:break-word}
.eyebrow{font:600 11.5px/1 var(--body);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:12px 14px;min-width:0}
.stat b{display:block;font:700 22px/1.3 var(--display);font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:13px}
nav.toc{display:flex;flex-wrap:wrap;gap:6px}
nav.toc a{background:var(--accent-soft);color:var(--accent);text-decoration:none;border-radius:6px;
  padding:4px 10px;font-size:13.5px}
nav.toc a:focus-visible,summary:focus-visible,button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:13px}
.flow span{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:3px 9px}
.flow i{color:var(--muted);font-style:normal}
.wide{overflow-x:auto;min-width:0}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--surface);
  border:1px solid var(--rule);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12.5px}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
details{background:var(--surface);border:1px solid var(--rule);border-radius:10px;min-width:0}
details>summary{cursor:pointer;padding:10px 14px;list-style:none;display:flex;gap:8px;
  align-items:baseline;flex-wrap:wrap}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸";color:var(--muted);font-size:12px}
details[open]>summary::before{content:"▾"}
details>.body{padding:0 14px 14px;display:flex;flex-direction:column;gap:10px;min-width:0}
.tag{font:600 11.5px/1.6 var(--mono);border-radius:999px;padding:0 8px;background:var(--accent-soft);color:var(--accent);white-space:nowrap}
.pass{color:var(--pass);font-weight:600}.fail{color:var(--fail);font-weight:600}
.note{border-left:3px solid var(--warn);padding:8px 12px;background:var(--surface);border-radius:0 8px 8px 0;font-size:14px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:13.5px}
.kv dt{color:var(--muted)}.kv dd{margin:0;min-width:0;overflow-wrap:anywhere}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.controls button{font:500 13px var(--body);border:1px solid var(--rule);background:var(--surface);
  color:var(--ink);border-radius:999px;padding:4px 12px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.controls input{font:14px var(--body);padding:6px 10px;border:1px solid var(--rule);border-radius:8px;
  background:var(--surface);color:var(--ink);min-width:0;flex:1 1 220px}
.calls{display:flex;flex-direction:column;gap:6px}
.lbl{font:600 12px var(--body);color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
"""

SCRIPT = """
<script>
(function(){
  var buttons = document.querySelectorAll('[data-stage-filter]');
  var search = document.getElementById('call-search');
  var calls = document.querySelectorAll('[data-call]');
  var count = document.getElementById('call-count');
  var stage = 'all';
  function apply(){
    var q = (search.value || '').toLowerCase(); var shown = 0;
    calls.forEach(function(el){
      var ok = (stage === 'all' || el.dataset.stage === stage) &&
               (!q || el.dataset.text.indexOf(q) !== -1);
      el.hidden = !ok; if (ok) shown++;
    });
    count.textContent = shown + ' of ' + calls.length + ' calls shown';
  }
  buttons.forEach(function(b){
    b.addEventListener('click', function(){
      stage = b.dataset.stageFilter;
      buttons.forEach(function(x){ x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      apply();
    });
  });
  search.addEventListener('input', apply);
  document.getElementById('expand-shown').addEventListener('click', function(){
    calls.forEach(function(el){ if (!el.hidden) el.open = true; });
  });
  document.getElementById('collapse-all').addEventListener('click', function(){
    calls.forEach(function(el){ el.open = false; });
  });
  apply();
})();
</script>
"""

STAGE_NAMES = {
    "step1": "Step 1: paper functions",
    "step2": "Step 2: serving",
    "source": "Step 3: source solve",
    "build": "Step 4: build",
    "eval": "Step 5: evaluation",
    "compile": "Step 6: compile check",
}


def _stage(index: int, e: dict, pid: str, first_source: int, first_eval: int, last_eval: int) -> str:
    tag = e.get("tag") or ""
    if tag == f"source:{pid}":
        return "source"
    rest = tag[len(pid) + 1:] if tag.startswith(pid + ":") else ""
    if rest.startswith(("phase", "judge", "build", "repair")):
        return "build"
    if rest:
        return "eval"
    if index < first_source:
        return "step1" if index < 4 else "step2"
    return "compile" if index > last_eval else "eval"


def _prompt_text(p) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        return "\n\n".join(b.get("text", json.dumps(b)) if isinstance(b, dict) else str(b) for b in p)
    return json.dumps(p, indent=2) if p is not None else ""


def _pretty_response(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (ValueError, TypeError):
        return text


def details(summary: str, body: str, open_: bool = False, attrs: str = "") -> str:
    return (f"<details{' open' if open_ else ''} {attrs}><summary>{summary}</summary>"
            f"<div class='body'>{body}</div></details>")


def render(pid: str) -> str:
    run = config.RUNS_DIR / pid
    problem = json.loads((config.PROBLEMS_DIR / f"{pid}.json").read_text())
    t = json.loads((run / "transcript.json").read_text())
    b = json.loads((run / "build_report.json").read_text())
    ev = json.loads((run / "eval_report.json").read_text())
    reg = json.loads((config.CACHES_DIR / pid / "registry.json").read_text())
    paper_dir = config.CACHES_DIR / f"paper_{pid}"
    paper_reg = json.loads((paper_dir / "registry.json").read_text()) if paper_dir.exists() else None
    compiled_path = config.CACHES_DIR / "master" / f"compiled_{pid}.py"
    compiled = compiled_path.read_text() if compiled_path.exists() else ""
    # Only the first, hand-run test: later calls carry a run_id and belong to experiments.
    log = [e for e in (json.loads(l) for l in open(config.RUNS_DIR / "llm_log.jsonl")) if not e.get("run_id")]

    first_source = next(i for i, e in enumerate(log) if e.get("tag") == f"source:{pid}")
    eval_idx = [i for i, e in enumerate(log) if (e.get("tag") or "").startswith(pid + ":") and
                not (e.get("tag") or "")[len(pid) + 1:].startswith(("phase", "judge", "build", "repair"))]
    first_eval, last_eval = min(eval_idx), max(eval_idx)
    stages = [_stage(i, e, pid, first_source, first_eval, last_eval) for i, e in enumerate(log)]
    stage_cost = defaultdict(float)
    stage_calls = defaultdict(int)
    for s, e in zip(stages, log):
        stage_cost[s] += e.get("cost", 0)
        stage_calls[s] += 1
    total = sum(e.get("cost", 0) for e in log)
    by_role = b["cost_by_role"]
    c = ev["costs"]
    builder_system = next((e.get("system") for e in log if e["role"] == "builder"), None)
    builder_system_text = _prompt_text(builder_system)

    o: list[str] = [STYLE, "<main>"]
    o.append(f"<div class='eyebrow'>CPU-ARC replication · SciBench, paper Appendix A</div>"
             f"<h1>Gas-law test run</h1><p class='muted'>Every artifact of the test: the source "
             f"solve, each build phase and attempt, the functions that were written, every "
             f"evaluation variant, the compiled file, and all {len(log)} model calls with their "
             f"exact prompts and responses. Read straight from the run files.</p>")
    o.append("<div class='stats'>"
             f"<div class='stat'><span>Whole test</span><b>{money(total)}</b><span>{len(log)} logged calls</span></div>"
             f"<div class='stat'><span>Source solve</span><b>{money(c['source_cost'])}</b><span>142 in · 1,136 out tokens</span></div>"
             f"<div class='stat'><span>Build</span><b>{money(c['builder_cost'])}</b><span>{sum(int(r['calls']) for r in by_role.values())} calls · "
             f"{c['build_multiplier']:.0f}× the source solve</span></div>"
             f"<div class='stat'><span>One cache use</span><b>${c['cache_cost_per_invocation']:.5f}</b><span>"
             f"{c['savings_multiplier']:.0f}× cheaper · break-even {c['break_even_invocations']} uses</span></div>"
             "</div>")
    o.append("<nav class='toc'><a href='#question'>Question</a><a href='#paper'>Paper's entry</a>"
             "<a href='#source'>Source solve</a><a href='#build'>Build</a><a href='#functions'>Final functions</a>"
             "<a href='#eval'>Evaluation</a><a href='#compiled'>Compiled</a><a href='#calls'>All model calls</a></nav>")

    # Question
    o.append(f"<h2 id='question'>The question</h2><pre class='wrap'>{esc(problem['question'])}</pre>"
             f"<p><span class='muted'>Golden label:</span> {esc(problem['golden_label'])}</p>")
    o.append("<div class='stats'>" + "".join(
        f"<div class='stat'><span>{esc(STAGE_NAMES[s])}</span><b>{money(stage_cost[s])}</b>"
        f"<span>{stage_calls[s]} calls</span></div>" for s in STAGE_NAMES if stage_calls[s]) + "</div>")

    # Paper's entry (steps 1-2)
    if paper_reg:
        o.append("<h2 id='paper'>Steps 1–2: the paper's own entry on our runtime</h2>"
                 "<p>The paper's printed functions (Figures 5–6; the pressure converter is described "
                 "but not printed, so it was written from its description) were run on our runtime. "
                 "Output: 5132.10 kPa = 5.13 MPa = 51.32 bar = 50.65 atm, with a narration matching "
                 "Figure 8. Serving then hit the cache for the original, a rewording (similarity 0.887) "
                 "and new values (2.5 mol N₂ → 646.07 kPa), and an unrelated question fell back to the "
                 "full model. Compare these 3 functions with the 8 our builder wrote, below.</p>")
        for f in paper_reg["functions"]:
            src = (paper_dir / "functions" / f"{f['id']}.py").read_text()
            o.append(details(f"<span class='tag'>{esc(f['origin'])}</span>{esc(f['description'])}",
                             f"<pre>{esc(src.strip())}</pre>"))

    # Source
    o.append("<h2 id='source'>Step 3: the source solve (the transcript)</h2>")
    o.append("<dl class='kv'>"
             f"<dt>Model</dt><dd class='mono'>{esc(t['model'])}, effort high</dd>"
             "<dt>Tokens</dt><dd>142 input · 1,136 output (about 790 visible; the rest is hidden thinking, "
             "billed but not shown)</dd>"
             f"<dt>Cost</dt><dd>{money(t['cost'])}</dd></dl>")
    src_call = log[first_source]
    o.append(details("Exact prompt sent", f"<pre class='wrap'>{esc(_prompt_text(src_call['prompt']))}</pre>"))
    o.append(details("Visible worked solution: what the builder reads", f"<pre class='wrap'>{esc(t['solution'])}</pre>", open_=True))
    o.append(details("Thinking summary: for humans only, not given to the builder",
                     f"<pre class='wrap'>{esc(t['thinking_summary'])}</pre>"))

    # Build
    o.append("<h2 id='build'>Step 4: the build</h2>")
    o.append("<div class='flow'><span>Phase 1 split question / answer</span><i>→</i><span>Phase 2 decompose</span>"
             "<i>→</i><span>Phase 3 parameters + template</span><i>→</i><span>Phase 4 plan</span><i>→</i>"
             "<span>Phase 5, per function: write → static check → run → judge → fix</span><i>→</i>"
             "<span>End-to-end check</span><i>→</i><span>3 extraction rounds</span><i>→</i>"
             "<span>Register 2 entry points</span></div>")
    o.append("<h3>Tokens and cost by role</h3><div class='wide'><table><tr><th>Role</th><th class='n'>Calls</th>"
             "<th class='n'>Input</th><th class='n'>Cache write</th><th class='n'>Cache read</th>"
             "<th class='n'>Output</th><th class='n'>Cost</th></tr>")
    tin = tout = 0
    for role, r in by_role.items():
        tin += r["input"] + r["cache_write"] + r["cache_read"]
        tout += r["output"]
        o.append(f"<tr><td>{esc(role)}</td><td class='n'>{int(r['calls'])}</td><td class='n'>{int(r['input']):,}</td>"
                 f"<td class='n'>{int(r['cache_write']):,}</td><td class='n'>{int(r['cache_read']):,}</td>"
                 f"<td class='n'>{int(r['output']):,}</td><td class='n'>{money(r['cost'])}</td></tr>")
    o.append(f"<tr><th>Total</th><th></th><th class='n' colspan='3'>{int(tin):,} prompt tokens</th>"
             f"<th class='n'>{int(tout):,}</th><th class='n'>{money(b['cost'])}</th></tr></table></div>")
    o.append("<p class='note'>Output tokens include hidden thinking. Token counts per call were not logged "
             "in this run (per-role totals above are exact, and per-call cost is exact). New runs log "
             "tokens for every call.</p>")
    o.append("<h3>Every build call, in order</h3><div class='wide'><table><tr><th>#</th><th>Role</th><th>Step</th>"
             "<th class='n'>Seconds</th><th class='n'>Cost</th></tr>")
    n = 0
    for i, (s, e) in enumerate(zip(stages, log)):
        if s != "build":
            continue
        n += 1
        o.append(f"<tr><td class='n'>{n}</td><td>{esc(e['role'])}</td><td class='mono'>{esc((e.get('tag') or '')[len(pid) + 1:])}</td>"
                 f"<td class='n'>{e.get('seconds', '')}</td><td class='n'>{money(e.get('cost', 0))}</td></tr>")
    o.append("</table></div>")
    if builder_system_text:
        o.append(details("The builder's system prompt (same for every builder call)",
                         f"<pre class='wrap'>{esc(builder_system_text)}</pre>"))
    o.append("<h3>Phases 1–4: what the builder decided</h3>")
    for name, val in b["phases"].items():
        o.append(details(esc(name), f"<pre>{esc(json.dumps(val, indent=2, ensure_ascii=False))}</pre>"))
    o.append("<h3>Phase 5: every attempt, failures included</h3>")
    for a in b["attempts"]:
        cls, word = ("pass", "passed") if a["passed"] else ("fail", "failed")
        fb = "" if a["passed"] else f"<div class='lbl'>Feedback sent back to the builder</div><pre class='wrap'>{esc(a['feedback'])}</pre>"
        o.append(details(f"<span class='tag'>{esc(a['function_id'])}</span>attempt {a['attempt']} · "
                         f"<span class='{cls}'>{word}</span>",
                         fb + f"<div class='lbl'>Code</div><pre>{esc(a['code'])}</pre>"))
    o.append("<h3>End-to-end checks</h3>")
    for e2e in b["end_to_end"]:
        o.append(details(f"Correctness on the original values · <span class='{'pass' if e2e['passed'] else 'fail'}'>"
                         f"{'pass' if e2e['passed'] else 'fail'}</span>",
                         f"<pre class='wrap'>{esc(json.dumps(e2e['output'], ensure_ascii=False, indent=1))}</pre>"
                         f"<p><span class='muted'>Judge:</span> {esc(e2e['critique'])}</p>"))
    for r in b["extraction_rounds"]:
        o.append(details(f"Extraction round {r['round']} · <span class='{'pass' if r['passed'] else 'fail'}'>"
                         f"{'pass' if r['passed'] else 'fail'}</span>",
                         f"<div class='lbl'>What the small model extracted from the raw question</div>"
                         f"<pre class='wrap'>{esc(json.dumps(r.get('extracted', {}), ensure_ascii=False, indent=1))}</pre>"
                         f"<p><span class='muted'>Judge:</span> {esc(r['critique'])}</p>"))

    # Final functions
    o.append("<h2 id='functions'>The cache entry the builder produced</h2>"
             f"<p>{len(reg['functions'])} functions and {len(reg['entry_points'])} entry points, as saved in "
             f"<span class='mono'>caches/{esc(pid)}/</span>.</p>")
    for f in reg["functions"]:
        src = (config.CACHES_DIR / pid / "functions" / f"{f['id']}.py").read_text()
        meta = (f"<dl class='kv'><dt>Inputs</dt><dd>{esc('; '.join(f['inputs']) or '—')}</dd>"
                f"<dt>Outputs</dt><dd>{esc('; '.join(f['outputs']) or '—')}</dd>"
                f"<dt>Also answers to</dt><dd>{esc('; '.join(f['aliases']) or '—')}</dd></dl>")
        o.append(details(esc(f["description"]), meta + f"<pre>{esc(src.strip())}</pre>"))
    o.append("<h3>Entry points</h3>")
    for ep in reg["entry_points"]:
        params = "".join(f"<tr><td>{esc(p['description'])}</td><td class='mono'>{esc(p['original_value'])}</td>"
                         f"<td>{esc(', '.join(p['format_examples']))}</td><td>{esc(p['validity_range'])}</td></tr>"
                         for p in ep["params"])
        o.append(details(f"<span class='tag'>{esc(ep['kind'])}</span>{esc(ep['text'][:120])}",
                         f"<pre class='wrap'>{esc(ep['text'])}</pre><div class='wide'><table><tr><th>Parameter</th>"
                         f"<th>Original</th><th>Format examples</th><th>Valid range</th></tr>{params}</table></div>"))

    # Evaluation
    o.append("<h2 id='eval'>Step 5: the evaluation</h2>")
    o.append("<div class='flow'><span>Correctness: 2 judges each for S Acc, C Acc, fidelity</span><i>→</i>"
             "<span>Generator writes 5 variants</span><i>→</i><span>2 judges validate each (both must accept)</span>"
             "<i>→</i><span>Variant runs through the cache</span><i>→</i><span>2 more judges grade the answer</span></div>")
    modes = defaultdict(lambda: [0, 0])
    for v in ev["variants"]:
        modes[v["mode"]][1] += 1
        modes[v["mode"]][0] += v["outcome"] == "pass"
    o.append("<div class='stats'>"
             f"<div class='stat'><span>Correctness</span><b>{'✓' if ev['cache_correct'] else '✗'} cache · "
             f"{'✓' if ev['fidelity'] else '✗'} fidelity</b><span>source correct: {ev['source_correct']}</span></div>"
             + "".join(f"<div class='stat'><span>{esc(m)}</span><b>{p}/{tot}</b><span>passed</span></div>"
                       for m, (p, tot) in modes.items()) + "</div>")
    o.append("<p class='note'>The first evaluation attempt stopped when the call that writes new-value "
             "variants was refused. That call was not logged (logging of refusals was added afterwards). "
             "The prompt was reworded from “reasoning trace / narration” to “worked solution”, and the "
             "run resumed without repeating the results already collected.</p>")
    eg = defaultdict(lambda: [0, 0.0])
    for s, e in zip(stages, log):
        if s == "eval":
            key = (e["role"], (e.get("tag") or "")[len(pid) + 1:] or "cache invocation")
            eg[key][0] += 1
            eg[key][1] += e.get("cost", 0)
    o.append("<h3>Evaluation cost by step</h3><div class='wide'><table><tr><th>Role</th><th>Step</th>"
             "<th class='n'>Calls</th><th class='n'>Cost</th></tr>" +
             "".join(f"<tr><td>{esc(r)}</td><td class='mono'>{esc(k)}</td><td class='n'>{nc}</td>"
                     f"<td class='n'>{money(cc)}</td></tr>" for (r, k), (nc, cc) in eg.items()) + "</table></div>")
    o.append(f"<h3>Original question through the cache</h3><pre class='wrap'>{esc(ev['original_cache_answer'])}</pre>")
    o.append("<h3>Every variant</h3>")
    for v in ev["variants"]:
        cls = "pass" if v["outcome"] == "pass" else "fail"
        o.append(details(f"<span class='tag'>{esc(v['mode'])}</span><span class='{cls}'>{esc(v['outcome'].upper())}</span>"
                         f"<span>{esc(v['question'][:120])}…</span>",
                         f"<div class='lbl'>Question</div><pre class='wrap'>{esc(v['question'])}</pre>"
                         f"<div class='lbl'>Cache answer</div><pre class='wrap'>{esc(v['cache_answer'])}</pre>"
                         f"<div class='lbl'>Reference answer</div><pre class='wrap'>{esc(v['reference'])}</pre>"
                         f"<div class='lbl'>Grading judges</div><p>{esc(v['detail'])}</p>"))

    # Compiled
    if compiled:
        o.append("<h2 id='compiled'>Step 6: the compiled cache</h2><p>The same 8 functions after static "
                 "resolution: each <span class='mono'>do()</span> became a direct call and each "
                 "<span class='mono'>load()</span> a parameter or dictionary key. It gave identical "
                 "output to the uncompiled cache on all 15 variants and never loaded the embedding model.</p>")
        o.append(details(f"<span class='mono'>caches/master/compiled_{esc(pid)}.py</span>", f"<pre>{esc(compiled)}</pre>"))

    # All calls
    o.append(f"<h2 id='calls'>All {len(log)} model calls</h2><p>Exact prompt and response for every call, in "
             "the order they ran. Filter by stage or search any text (a function name, a number, a judge's "
             "critique).</p>")
    o.append("<div class='controls'><button type='button' data-stage-filter='all' aria-pressed='true'>All</button>" +
             "".join(f"<button type='button' data-stage-filter='{s}' aria-pressed='false'>{esc(STAGE_NAMES[s])}</button>"
                     for s in STAGE_NAMES if stage_calls[s]) +
             "</div><div class='controls'><input id='call-search' type='search' placeholder='Search prompts and responses'>"
             "<button type='button' id='expand-shown'>Expand shown</button>"
             "<button type='button' id='collapse-all'>Collapse all</button></div>"
             "<p id='call-count' class='muted'></p><div class='calls'>")
    for i, (s, e) in enumerate(zip(stages, log), 1):
        prompt = _prompt_text(e.get("prompt"))
        system = _prompt_text(e.get("system"))
        response = _pretty_response(e.get("response", ""))
        sys_block = ""
        if system:
            sys_block = ("<div class='lbl'>System prompt</div><p class='muted'>The builder system prompt shown in "
                         "the build section.</p>") if system == builder_system_text else \
                        f"<div class='lbl'>System prompt</div><pre class='wrap'>{esc(system)}</pre>"
        tokens = (f" · {e['input_tokens']:,} in / {e['output_tokens']:,} out" if "input_tokens" in e else "")
        searchable = esc(f"{e['role']} {e.get('tag') or ''} {prompt} {response}".lower())
        o.append(details(
            f"<span class='mono muted'>#{i}</span><span class='tag'>{esc(e['role'])}</span>"
            f"<span class='mono'>{esc(e.get('tag') or STAGE_NAMES[s])}</span>"
            f"<span class='muted'>{e.get('seconds', '?')} s · {money(e.get('cost', 0))}{tokens}</span>",
            f"<dl class='kv'><dt>Stage</dt><dd>{esc(STAGE_NAMES[s])}</dd><dt>Model</dt><dd class='mono'>{esc(e.get('model'))}</dd></dl>"
            + sys_block +
            f"<div class='lbl'>Prompt</div><pre class='wrap'>{esc(prompt)}</pre>"
            f"<div class='lbl'>Response</div><pre class='wrap'>{esc(response)}</pre>",
            attrs=f"data-call data-stage='{s}' data-text=\"{searchable}\""))
    o.append("</div></main>")
    o.append(SCRIPT)
    return "".join(o)


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "gas_law"
    path = config.RUNS_DIR / pid / "walkthrough.html"
    path.write_text(render(pid))
    print(path, f"{path.stat().st_size / 1e6:.2f} MB")
