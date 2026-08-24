"""FastAPI dashboard for reviewing persisted Plan C evidence on localhost."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from harness.agent_runtime.service import MineFailureResearchService
from harness.pairing import PairedCaptureService, PairingError
from harness.persistence.store import ExperimentStore, ReviewState
from harness.research.campaign import ResearchCampaignStore


def _timeline(trajectory_path: str | None) -> list[dict[str, Any]]:
    if not trajectory_path or not Path(trajectory_path).is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in Path(trajectory_path).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("record_type") == "initial_observation":
            events.append({"kind": "experiment_start", "time": entry["observation"]["simulation_time"]})
        elif entry.get("record_type") == "step":
            result = entry["result"]
            events.append({"kind": "robot_action", "time": result["simulation_time"], "action": entry["action"]})
            for event in result.get("events", []):
                events.append({"kind": "environmental_event", "time": result["simulation_time"], "event": event})
            if result.get("done"):
                events.append({"kind": "termination", "time": result["simulation_time"]})
    return events


def build_pair_view(store: ExperimentStore, pair_id: str) -> dict[str, Any] | None:
    pair = store.get_pair(pair_id)
    if not pair:
        return None
    isaac = store.get_experiment(pair["isaac_run_id"])
    reactor = store.get_experiment(pair["reactor_run_id"])
    return {
        "pair": pair, "isaac": isaac, "reactor": reactor,
        "isaac_artifacts": store.artifacts_for("experiment", pair["isaac_run_id"]),
        "reactor_artifacts": store.artifacts_for("experiment", pair["reactor_run_id"]),
        "pair_artifacts": store.artifacts_for("pair", pair_id),
        "isaac_timeline": _timeline(isaac["trajectory_path"] if isaac else None),
        "reactor_timing_note": "Reactor chunks/sequences are shown independently; no precise clock synchronization is claimed.",
        "authority": {"isaac": "PHYSICS-GROUNDED SIMULATION", "reactor": "NEURAL WORLD VISUAL EVIDENCE"},
    }


def build_experiment_view(store: ExperimentStore, run_id: str) -> dict[str, Any] | None:
    experiment = store.get_experiment(run_id)
    if not experiment:
        return None
    authority = "PHYSICS-GROUNDED SIMULATION" if experiment["backend"] == "isaac_sim" else "NEURAL WORLD VISUAL EVIDENCE" if experiment["backend"].startswith("reactor/") else "RECORDED HARNESS EXPERIMENT"
    return {"experiment": experiment, "artifacts": store.artifacts_for("experiment", run_id), "timeline": _timeline(experiment["trajectory_path"]), "authority": authority}


class ReviewUpdate(BaseModel):
    review_state: ReviewState


class PairCaptureCreate(BaseModel):
    isaac_run_id: str
    objective: str = ""
    model: str | None = None


class AgentCampaignCreate(BaseModel):
    objective: str
    experiment_budget: int = 3


class AgentStepCreate(BaseModel):
    instruction: str


_STYLE = """
:root{--bg:#0a1020;--panel:#101a30;--line:#263552;--text:#edf3ff;--muted:#9dabc2;--isaac:#3ad8b0;--reactor:#ae8cff;--red:#ff6685;--amber:#ffbf5c;--green:#57d68d;--blue:#69b9ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 18% 0,#172948 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit;text-decoration:none}button,select{font:inherit}button{cursor:pointer}.shell{max-width:1500px;margin:auto;padding:28px}.topbar{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:25px}.eyebrow{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}.title{font-size:clamp(28px,4vw,46px);letter-spacing:-.04em;margin:3px 0 7px}.subtitle{margin:0;color:var(--muted);font-size:16px}.authority{display:inline-flex;gap:8px;align-items:center;border:1px solid var(--line);border-radius:999px;padding:7px 11px;color:var(--muted);font-size:12px;white-space:nowrap}.dot{width:7px;height:7px;border-radius:50%;background:var(--reactor)}.grid{display:grid;gap:16px}.summary-grid{grid-template-columns:repeat(4,1fr);margin:22px 0}.card,.metric,.viewport,.details-card{background:linear-gradient(145deg,rgba(25,38,65,.95),rgba(15,25,45,.95));border:1px solid var(--line);border-radius:16px;box-shadow:0 18px 45px rgba(0,0,0,.16)}.metric{padding:16px}.metric .number{font-size:27px;font-weight:800;letter-spacing:-.04em}.metric .label,.micro{color:var(--muted);font-size:12px}.section-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:30px 0 14px}.section-head h2{font-size:18px;margin:0}.pairs{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}.pair-card{padding:14px;transition:transform .16s,border-color .16s}.pair-card:hover{transform:translateY(-2px);border-color:#4d6391}.pair-card .visuals{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:13px 0}.thumb{height:116px;background:#0a1223;border-radius:10px;overflow:hidden;display:grid;place-items:center;color:var(--muted);font-size:12px}.thumb img{width:100%;height:100%;object-fit:cover}.pair-title{font-weight:750;font-size:17px;margin:2px 0}.pair-meta{color:var(--muted);font-size:13px}.row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.spread{justify-content:space-between}.badge{display:inline-flex;align-items:center;border:1px solid currentColor;border-radius:99px;padding:3px 8px;font-size:11px;font-weight:800;letter-spacing:.02em;text-transform:uppercase}.badge.red{color:var(--red);background:#ff668515}.badge.amber{color:var(--amber);background:#ffbf5c15}.badge.green{color:var(--green);background:#57d68d15}.badge.isaac{color:var(--isaac);background:#3ad8b015}.badge.reactor{color:var(--reactor);background:#ae8cff15}.primary-action{display:inline-flex;background:var(--blue);color:#071222;border-radius:8px;padding:8px 11px;font-size:13px;font-weight:800}.standalone{padding:0 15px 14px}.standalone summary{padding:14px 0;cursor:pointer;font-weight:700}.run-row{display:flex;justify-content:space-between;gap:14px;padding:12px 0;border-top:1px solid var(--line);color:var(--muted)}.run-row strong{color:var(--text)}.back{color:var(--muted);font-weight:700;font-size:13px}.pair-top{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:18px}.pair-name{font-weight:800;font-size:20px}.summary-strip{grid-template-columns:repeat(4,1fr);margin:0 0 18px}.summary-strip .metric{min-height:104px}.summary-strip strong{display:block;font-size:15px;margin:4px 0}.comparison-grid{grid-template-columns:1fr 1fr;gap:18px}.viewport{overflow:hidden;min-width:0}.viewport-head{padding:15px 16px 12px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.viewport h2{font-size:18px;margin:0}.viewport.isaac{border-top:3px solid var(--isaac)}.viewport.reactor{border-top:3px solid var(--reactor)}.media-stage{height:min(54vh,590px);min-height:350px;background:#050a14;display:grid;place-items:center;position:relative}.media-stage video,.media-stage img{height:100%;width:100%;object-fit:contain}.empty-media{color:var(--muted);max-width:250px;text-align:center}.media-controls{padding:10px 14px;display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--line);color:var(--muted);font-size:12px}.media-controls button{background:transparent;border:1px solid var(--line);color:var(--text);border-radius:7px;padding:5px 9px}.media-controls button:disabled{opacity:.35;cursor:not-allowed}.filmstrip{display:flex;gap:8px;overflow:auto;padding:10px 13px 14px;background:#0b1426}.filmstrip button{padding:0;border:2px solid transparent;border-radius:7px;background:#111;min-width:72px;height:53px;overflow:hidden}.filmstrip button.active{border-color:var(--blue)}.filmstrip img{width:100%;height:100%;object-fit:cover}.below-grid{grid-template-columns:1.05fr .95fr .8fr;margin-top:18px}.details-card{padding:16px}.details-card h2{font-size:16px;margin:0 0 10px}.details-card p{margin:6px 0;color:var(--muted)}.timeline{list-style:none;margin:0;padding:0}.timeline li{border-left:2px solid var(--line);padding:0 0 12px 13px;margin-left:5px;position:relative}.timeline li:before{content:"";position:absolute;width:9px;height:9px;border-radius:50%;left:-5.5px;top:5px;background:var(--blue)}.timeline li.environmental_event:before,.timeline li.termination:before{background:var(--red)}.timeline .time{color:var(--blue);font-size:12px;font-weight:800}.review select{width:100%;background:#0c1527;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px}.review button{background:var(--blue);color:#071222;border:0;border-radius:8px;padding:9px 12px;font-weight:800;margin-top:9px}.save-status{min-height:20px;font-size:12px;color:var(--green);margin:8px 0 0}.technical{margin-top:18px}.technical details{border-top:1px solid var(--line);padding:12px 0}.technical summary{cursor:pointer;font-weight:700}.technical pre{white-space:pre-wrap;overflow:auto;color:var(--muted);font-size:12px;margin:10px 0 0}.artifact-links a{display:inline-block;margin:0 7px 7px 0;color:var(--blue);font-size:12px}.notice{border-left:3px solid var(--amber);padding:8px 10px;background:#ffbf5c10;color:#ffd99a;font-size:13px;border-radius:0 8px 8px 0}@media(max-width:900px){.summary-grid,.summary-strip,.below-grid{grid-template-columns:repeat(2,1fr)}.comparison-grid{grid-template-columns:1fr}.media-stage{height:55vw;min-height:300px}.topbar{display:block}.authority{margin-top:14px}}@media(max-width:560px){.shell{padding:18px}.summary-grid,.summary-strip{grid-template-columns:1fr}.run-row{display:block}.run-row>*{display:block;margin:3px 0}.media-stage{min-height:240px}}
"""

_THEME_STYLE = """
/* Light is the presentation default; dark is intentionally charcoal/black. */
:root{--bg:#f4f7fb;--panel:#fff;--line:#d8e1ee;--text:#162235;--muted:#62728a;--blue:#256bc0}.authority{background:#fff}.card,.metric,.viewport,.details-card{background:linear-gradient(145deg,#fff,#f8fbff);box-shadow:0 16px 38px rgba(29,54,88,.09)}.thumb,.filmstrip{background:#edf2f8}.filmstrip button{background:#dce5f0}.media-controls{background:#fff}.media-controls button{color:var(--text);background:#fff;border-color:var(--line)}.review select{background:#fff;color:var(--text);border-color:var(--line)}body{background:radial-gradient(circle at 18% 0,#dcecff 0,transparent 35%),var(--bg)}#theme-toggle{position:fixed;z-index:10;right:18px;bottom:18px;border:1px solid var(--line);border-radius:999px;padding:7px 10px;background:var(--panel);color:var(--text);font-size:12px;box-shadow:0 8px 24px rgba(0,0,0,.13)}body.dark{--bg:#111315;--panel:#1a1d20;--line:#343a40;--text:#f0f2f4;--muted:#a7afb8;--blue:#7eb8ff;background:radial-gradient(circle at 18% 0,#24282c 0,transparent 35%),#111315}body.dark .authority,body.dark .media-controls,body.dark .media-controls button,body.dark .review select{background:#1a1d20;color:var(--text);border-color:var(--line)}body.dark .card,body.dark .metric,body.dark .viewport,body.dark .details-card{background:linear-gradient(145deg,#202428,#151719);box-shadow:0 18px 45px rgba(0,0,0,.3)}body.dark .thumb,body.dark .filmstrip{background:#0d0f11}body.dark .filmstrip button{background:#000}
"""

_REACTOR_STYLE = """
.reactor-layout{grid-template-columns:minmax(0,1.65fr) minmax(310px,.8fr);align-items:start}.reactor-controls{display:grid;gap:12px}.reactor-controls h2{margin-bottom:0}.reactor-controls label{display:grid;gap:6px;color:var(--muted);font-size:12px;font-weight:700}.reactor-controls textarea,.reactor-controls input[type=number],.reactor-controls input[type=file]{width:100%;font:inherit;color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}.reactor-controls textarea{resize:vertical}.reactor-controls button,.navigation button{border:1px solid var(--line);border-radius:8px;padding:9px 12px;background:var(--panel);color:var(--text);font-weight:750}.reactor-controls button:disabled{opacity:.45;cursor:not-allowed}.reactor-controls .live-primary,.navigation #stop-reactor{background:var(--blue);color:#071222;border-color:transparent}.navigation{margin-top:18px}.navigation .section-head{margin:0 0 12px}.nav-grids{display:flex;gap:32px;flex-wrap:wrap}.nav-group h3{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 8px}.control-pad{display:grid;grid-template-columns:repeat(3,52px);grid-template-rows:repeat(3,42px);gap:7px;margin-bottom:12px}.control-pad .north{grid-column:2;grid-row:1}.control-pad .west{grid-column:1;grid-row:2}.control-pad .east{grid-column:3;grid-row:2}.control-pad .south{grid-column:2;grid-row:3}.control-pad button.active{border-color:var(--reactor);box-shadow:0 0 0 2px #ae8cff35}.media-stage #reactor-video{display:none}.media-stage #reactor-video.active{display:block}.media-stage #video-placeholder.hidden{display:none}@media(max-width:900px){.reactor-layout{grid-template-columns:1fr}.reactor-controls{grid-template-columns:1fr 1fr}.reactor-controls h2,.reactor-controls p{grid-column:1/-1}}@media(max-width:560px){.reactor-controls{grid-template-columns:1fr}.nav-grids{gap:20px}}
"""

_RECORDING_SELECTOR_STYLE = """
.recording-switcher{margin:0 0 18px;padding:13px 16px;display:flex;align-items:center;gap:12px}.recording-switcher label{font-size:12px;font-weight:800;color:var(--muted);white-space:nowrap}.recording-switcher select{min-width:min(520px,100%);flex:1;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px 10px;font:inherit}@media(max-width:560px){.recording-switcher{display:grid;gap:6px}.recording-switcher select{min-width:0;width:100%}}
"""

_LIBRARY_BODY = """
<div class="shell"><header class="topbar"><div><div class="eyebrow">Plan C experiment review</div><h1 class="title">Physical AI Failure Harness</h1><p class="subtitle">Visual comparison of physics-grounded simulation and neural-world evidence.</p></div><div class="row"><a class="primary-action" href="/agent">Research agent</a><a class="primary-action" href="/reactor">Open live Reactor</a><div class="authority"><span class="dot"></span>Reactor evidence is not physics ground truth</div></div></header><section class="grid summary-grid" id="metrics"></section><section><div class="section-head"><h2>Paired experiment library</h2><span class="micro">Select a recording to review the comparison</span></div><div class="grid pairs" id="pairs"></div></section><details class="card standalone"><summary>Standalone experiments <span class="micro">(recorded without a paired comparison)</span></summary><div id="runs"></div></details></div>
"""
_AGENT_BODY = """
<div class="shell"><header class="topbar"><div><div class="eyebrow">OpenAI Agents SDK</div><h1 class="title">MineFailureResearcher</h1><p class="subtitle">One persistent tool-using researcher. Scientific evidence remains in the experiment store.</p></div><a class="back" href="/">← Experiment library</a></header><section class="grid comparison-grid"><article class="details-card reactor-controls"><h2>Campaign</h2><label>Existing campaign<select id="agent-campaign"><option value="">Create or select a campaign…</option></select></label><label>Objective<textarea id="agent-objective" rows="3">Investigate roof-support failure in the mine and compare Isaac with Reactor.</textarea></label><label>Experiment budget<input id="agent-budget" type="number" min="1" max="20" value="3"></label><button class="live-primary" id="agent-create">Create campaign</button><label>Research instruction<textarea id="agent-instruction" rows="5">Investigate roof-support failure in the mine. Inspect the world and previous evidence, then choose at most one useful experiment.</textarea></label><div class="row"><button class="live-primary" id="agent-step">Run research step</button><button id="agent-continue">Continue</button></div><p class="save-status" id="agent-message" role="status"></p></article><article class="details-card"><h2>Current research state</h2><div id="agent-summary" class="notice">Select or create a campaign.</div><div class="row" id="agent-links"></div><h2>Agent/tool events</h2><pre id="agent-events" style="white-space:pre-wrap;max-height:52vh;overflow:auto"></pre></article></section></div><script type="module" src="/static/agent-runtime.js"></script>
"""
_REACTOR_BODY = """
<div class="shell reactor-live"><header class="topbar"><div><div class="eyebrow">Live neural-world session</div><h1 class="title">Reactor control room</h1><p class="subtitle">Navigate a LingBot World 2 video world in real time. This is visual evidence, not a physics simulator.</p></div><div class="row"><a class="back" href="/">← Experiment library</a><span class="authority"><span class="dot"></span><span id="reactor-status">Checking configuration…</span></span></div></header><div class="grid reactor-layout"><section class="viewport reactor"><div class="viewport-head"><div><span class="badge reactor">Reactor / LingBot World 2</span><h2>Live video</h2></div><span class="micro" id="chunk-status">No session</span></div><div class="media-stage"><video id="reactor-video" autoplay playsinline muted></video><div class="empty-media" id="video-placeholder">Connect a session, choose a seed image, and start generating.</div></div></section><aside class="details-card reactor-controls"><h2>Paired capture</h2><label>Isaac recording<select id="paired-isaac-run"><option>Loading Isaac recordings…</option></select></label><label>Research objective<textarea id="paired-objective" rows="2" placeholder="Optional context for the world-model prompt."></textarea></label><div class="row"><button class="live-primary" id="prepare-pair">Prepare paired run</button><button id="finish-pair" disabled>Finish & save pair</button></div><p class="micro">Preparation sends the Isaac initial frame and structured context to Responses, then starts and records the Reactor session.</p><h2>Manual session</h2><label>Scene prompt<textarea id="reactor-prompt" rows="5" placeholder="Describe the world you want to explore."></textarea></label><label>Seed image<input id="reactor-image" type="file" accept="image/*"></label><label>Seed<input id="reactor-seed" type="number" min="0" value="42"></label><label>Look speed <span id="speed-value">5°/frame</span><input id="reactor-speed" type="range" min="0" max="30" value="5"></label><div class="row"><button id="connect-reactor">Connect</button><button id="start-reactor" disabled>Start manual world</button></div><div class="row"><button id="pause-reactor" disabled>Pause</button><button id="reset-reactor" disabled>Reset</button></div><p class="save-status" id="reactor-message" role="status"></p><p class="micro">Controls change at the next generated chunk. Stop controls are sent explicitly when a key is released.</p></aside></div><section class="details-card navigation"><div class="section-head"><h2>Navigation</h2><span class="micro">WASD moves · arrow keys look · release to stop</span></div><div class="nav-grids"><div class="nav-group"><h3>Move</h3><div class="control-pad"><button class="north" data-control="move_longitudinal" data-value="forward" aria-label="Move forward">W</button><button class="west" data-control="move_lateral" data-value="strafe_left" aria-label="Strafe left">A</button><button class="east" data-control="move_lateral" data-value="strafe_right" aria-label="Strafe right">D</button><button class="south" data-control="move_longitudinal" data-value="back" aria-label="Move backward">S</button></div></div><div class="nav-group"><h3>Look</h3><div class="control-pad"><button class="north" data-control="look_vertical" data-value="up" aria-label="Look up">↑</button><button class="west" data-control="look_horizontal" data-value="left" aria-label="Look left">←</button><button class="east" data-control="look_horizontal" data-value="right" aria-label="Look right">→</button><button class="south" data-control="look_vertical" data-value="down" aria-label="Look down">↓</button></div></div></div><button id="stop-reactor">Stop all motion</button></section><section class="technical details-card"><details><summary>Transport boundary</summary><p>A paired capture records browser-received Reactor video and writes a Plan C pair after you finish it. The comparison remains semantically aligned visual evidence, not synchronized physical replay.</p></details></section></div><script type="module" src="/static/reactor-live.js"></script>
"""
_PAIR_BODY = """
<div class="shell"><div class="pair-top"><a class="back" href="/">← Experiment library</a><div class="row"><span id="pair-name" class="pair-name"></span><span id="status"></span><span id="review-badge"></span></div></div><section class="details-card recording-switcher"><label for="pair-selector">Comparison recording</label><select id="pair-selector" aria-label="Choose a paired Isaac and Reactor recording"><option>Loading recordings…</option></select><span class="micro" id="pair-selector-count"></span></section><section class="grid summary-strip" id="summary"></section><section class="grid comparison-grid"><article class="viewport isaac"><div class="viewport-head"><div><span class="badge isaac">Isaac / PhysX</span><h2>Physics-grounded simulation — reference</h2></div></div><div id="isaac-media"></div></article><article class="viewport reactor"><div class="viewport-head"><div><span class="badge reactor">Reactor</span><h2>Neural-world visual evidence — not physics ground truth</h2></div></div><div id="reactor-media"></div></article></section><section class="grid below-grid"><article class="details-card" id="comparison-summary"></article><article class="details-card"><h2>Isaac event timeline</h2><ol class="timeline" id="timeline"></ol></article><article class="details-card review"><h2>Human review</h2><p>Stored as mutable review metadata; raw artifacts are unchanged.</p><select id="review-select" aria-label="Review state"></select><button id="save-review">Save review</button><p class="save-status" id="save-status" role="status"></p></article></section><section class="technical details-card"><details><summary>Experiment metadata</summary><pre id="metadata"></pre></details><details><summary>Comparison record</summary><pre id="comparison-record"></pre></details><details><summary>Artifact manifest</summary><pre id="artifact-manifest"></pre><div class="artifact-links" id="artifact-links"></div></details><details><summary>Raw event timeline</summary><pre id="raw-timeline"></pre></details></section></div><script type="module" src="/static/recording-selector.js"></script>
"""
_EXPERIMENT_BODY = """
<div class="shell"><div class="pair-top"><a class="back" href="/">← Experiment library</a><span id="run-name" class="pair-name"></span></div><section class="grid comparison-grid"><article class="viewport" id="standalone-viewport"><div class="viewport-head"><div><span id="standalone-authority" class="badge"></span><h2>Recorded experiment</h2></div></div><div id="standalone-media"></div></article><article class="details-card"><h2>Outcome</h2><div id="standalone-outcome"></div><h2>Event timeline</h2><ol class="timeline" id="standalone-timeline"></ol></article></section><section class="technical details-card"><details><summary>Experiment metadata</summary><pre id="standalone-metadata"></pre></details><details><summary>Artifact manifest</summary><pre id="standalone-artifacts"></pre></details><details><summary>Raw event timeline</summary><pre id="standalone-raw-timeline"></pre></details></section></div>
"""

_SCRIPT = """
const $=s=>document.querySelector(s),el=(t,c,x)=>{const n=document.createElement(t);if(c)n.className=c;if(x!==undefined)n.textContent=x;return n},url=a=>`/artifacts/${a.artifact_id}`,pretty=x=>String(x??'unknown').replaceAll('_',' '),status=x=>x==='candidate_discrepancy'||x==='not_comparable'?'red':x==='consistent_visual_evidence'?'green':'amber';
const themeToggle=$('#theme-toggle'),savedTheme=localStorage.getItem('harness-theme');if(savedTheme==='dark')document.body.classList.add('dark');themeToggle.textContent=document.body.classList.contains('dark')?'Light theme':'Dark theme';themeToggle.onclick=()=>{document.body.classList.toggle('dark');const dark=document.body.classList.contains('dark');localStorage.setItem('harness-theme',dark?'dark':'light');themeToggle.textContent=dark?'Light theme':'Dark theme'};
const redact=v=>Array.isArray(v)?v.map(redact):v&&typeof v==='object'?Object.fromEntries(Object.entries(v).map(([k,x])=>[k,(k.includes('path')||k==='trajectory_ref'||k==='evidence_refs')?'[server-side artifact reference]':redact(x)])):v;
const json=(id,v)=>$(id).textContent=JSON.stringify(redact(v),null,2),summary=a=>a.map(x=>({artifact_id:x.artifact_id,kind:x.kind,metadata:redact(x.metadata)})),playable=a=>a.filter(x=>['video','image','thumbnail'].includes(x.kind));
function browser(host,artifacts,label){const root=$(host),items=playable(artifacts);root.replaceChildren();if(!items.length){const stage=el('div','media-stage');stage.append(el('div','empty-media','No playable recording is available. Use the collapsed artifact manifest for available evidence.'));root.append(stage);return}let i=Math.max(0,items.findIndex(x=>x.kind==='video'));if(i<0)i=0;const stage=el('div','media-stage'),controls=el('div','media-controls'),prev=el('button','', '← Previous'),count=el('span',''),next=el('button','', 'Next →'),strip=el('div','filmstrip');controls.append(prev,count,next);function render(){stage.replaceChildren();const a=items[i],n=document.createElement(a.kind==='video'?'video':'img');n.src=url(a);n.controls=a.kind==='video';n.alt=`${label} recording ${i+1}`;stage.append(n);count.textContent=`Recording ${i+1} of ${items.length}`;prev.disabled=i===0;next.disabled=i===items.length-1;strip.replaceChildren();items.forEach((a,j)=>{const b=el('button',j===i?'active':'');b.type='button';b.setAttribute('aria-label',`Select ${label} recording ${j+1}`);b.onclick=()=>{i=j;render()};if(a.kind==='video')b.textContent='▶ Video';else{const img=document.createElement('img');img.src=url(a);img.alt='';b.append(img)}strip.append(b)})}prev.onclick=()=>{if(i){i--;render()}};next.onclick=()=>{if(i<items.length-1){i++;render()}};root.tabIndex=0;root.onkeydown=e=>{if(e.key==='ArrowLeft')prev.click();if(e.key==='ArrowRight')next.click()};render();root.append(stage,controls,strip)}
function timeline(host,events){const root=$(host);root.replaceChildren();if(!events.length){root.append(el('li','','No recorded timeline events.'));return}events.forEach(e=>{const li=el('li',e.kind);li.append(el('div','time',`t = ${e.time??'n/a'} s`));let text=pretty(e.kind);if(e.kind==='robot_action')text=`Action: ${e.action?.name||'recorded action'}`;if(e.kind==='environmental_event')text=`Environmental event: ${e.event?.event_type||'recorded event'}${e.event?.severity?` (${e.event.severity})`:''}`;if(e.kind==='termination')text='Experiment terminated';li.append(el('div','',text));root.append(li)})}
function outcome(x){const e=x?.evaluation;if(!e)return 'No evaluation record available.';return e.environmental_failure?`Environmental failure: ${pretty(e.failure_type)} (${pretty(e.severity)})`:e.task_success?'Task completed with no environmental failure recorded.':'No environmental failure recorded.'}
async function library(){const d=await fetch('/api/overview').then(r=>r.json()),pairs=d.pairs||[],all=(d.unpaired_experiments||[]).length+pairs.length*2;[['Paired experiments',pairs.length],['Candidate discrepancies',pairs.filter(p=>p.comparison_status==='candidate_discrepancy').length],['Inconclusive comparisons',pairs.filter(p=>p.comparison_status==='inconclusive').length],['Recorded runs',all]].forEach(([l,n])=>{const c=el('article','metric');c.append(el('div','number',n),el('div','label',l));$('#metrics').append(c)});for(const p of pairs){const card=document.createElement('a');card.href=`/pairs/${encodeURIComponent(p.pair_id)}`;card.className='card pair-card';const top=el('div','row spread');top.append(el('span','badge '+status(p.comparison_status),pretty(p.comparison_status)),el('span','micro',pretty(p.review_state)));card.append(top,el('div','pair-title',p.task),el('div','pair-meta',p.scenario_id||'Scenario unavailable'));const vs=el('div','visuals'),ia=el('div','thumb','Isaac replay'),re=el('div','thumb','Reactor evidence');vs.append(ia,re);card.append(vs);const bottom=el('div','row spread');bottom.append(el('span','micro',`Visual confidence: ${p.visual_confidence??'n/a'} · ${p.created_at||'Timestamp unavailable'}`),el('span','primary-action','Review comparison →'));card.append(bottom);$('#pairs').append(card);fetch(`/api/pairs/${encodeURIComponent(p.pair_id)}`).then(r=>r.json()).then(x=>[[ia,x.isaac_artifacts],[re,x.reactor_artifacts]].forEach(([target,arts])=>{const a=playable(arts).find(a=>a.kind==='thumbnail')||playable(arts)[0];if(a){target.replaceChildren();const image=document.createElement('img');image.src=url(a);image.alt='Recording preview';target.append(image)}})).catch(()=>{})}(d.unpaired_experiments||[]).forEach(r=>{const row=document.createElement('a');row.href=`/experiments/${encodeURIComponent(r.run_id)}`;row.className='run-row';row.append(el('strong','',r.run_id),el('span','',r.backend),el('span','',r.task),el('span','',outcome(r)));$('#runs').append(row)})}
async function pair(){const id=decodeURIComponent(location.pathname.split('/').pop()),d=await fetch(`/api/pairs/${encodeURIComponent(id)}`).then(r=>r.json()),p=d.pair;$('#pair-name').textContent=`Same experiment: ${p.task}`;$('#status').className='badge '+status(p.comparison_status);$('#status').textContent=pretty(p.comparison_status);$('#review-badge').className='badge amber';$('#review-badge').textContent=pretty(p.review_state);[['Physics outcome',outcome(d.isaac)],['Visual assessment',`${p.visual_observed===null?'Indeterminate':p.visual_observed?'Observed':'Not observed'} · ${p.visual_confidence??'n/a'} confidence`],['Action alignment',pretty(p.action_alignment)],['Comparison result',pretty(p.comparison_status)]].forEach(([l,v])=>{const c=el('article','metric');c.append(el('div','label',l),el('strong','',v));if(l==='Action alignment')c.append(el('div','micro',p.alignment_note));if(l==='Comparison result')c.append(el('div','micro',p.comparison_reason));$('#summary').append(c)});browser('#isaac-media',d.isaac_artifacts,'Isaac');browser('#reactor-media',d.reactor_artifacts,'Reactor');$('#comparison-summary').append(el('h2','','Comparison summary'),el('p','',p.comparison_reason));if(p.action_alignment!=='exact')$('#comparison-summary').append(el('div','notice',`Action alignment is ${pretty(p.action_alignment)}. The recordings are not frame- or clock-synchronized.`));$('#comparison-summary').append(el('p','micro',d.reactor_timing_note));timeline('#timeline',d.isaac_timeline);const select=$('#review-select');['unreviewed','valid_discrepancy','bad_world_model_generation','bad_scenario','simulator_artifact','inconclusive'].forEach(x=>{const o=el('option','',pretty(x));o.value=x;o.selected=x===p.review_state;select.append(o)});$('#save-review').onclick=async()=>{const s=$('#save-status');s.textContent='Saving…';try{const r=await fetch(`/api/pairs/${encodeURIComponent(id)}/review`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({review_state:select.value})});if(!r.ok)throw Error();s.textContent='Review state saved.';$('#review-badge').textContent=pretty(select.value)}catch{s.textContent='Unable to save review state. Please retry.'}};json('#metadata',{isaac:d.isaac,reactor:d.reactor});json('#comparison-record',p);json('#artifact-manifest',{isaac:summary(d.isaac_artifacts),reactor:summary(d.reactor_artifacts),pair:summary(d.pair_artifacts)});json('#raw-timeline',d.isaac_timeline);[...d.isaac_artifacts,...d.reactor_artifacts,...d.pair_artifacts].forEach(a=>{const link=el('a','',`Download ${a.kind} #${a.artifact_id}`);link.href=url(a);$('#artifact-links').append(link)})}
async function experiment(){const id=decodeURIComponent(location.pathname.split('/').pop()),d=await fetch(`/api/experiments/${encodeURIComponent(id)}`).then(r=>r.json());$('#run-name').textContent=d.experiment.run_id;const b=$('#standalone-authority');b.textContent=d.authority;b.classList.add(d.experiment.backend==='isaac_sim'?'isaac':'reactor');browser('#standalone-media',d.artifacts,'experiment');$('#standalone-outcome').append(el('p','',outcome(d.experiment)));timeline('#standalone-timeline',d.timeline);json('#standalone-metadata',d.experiment);json('#standalone-artifacts',summary(d.artifacts));json('#standalone-raw-timeline',d.timeline)}
if(document.body.dataset.page==='library')library();if(document.body.dataset.page==='pair')pair();if(document.body.dataset.page==='experiment')experiment();
"""


def _page(body: str, page: str) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Physical AI Failure Harness</title><style>{_STYLE}{_THEME_STYLE}{_REACTOR_STYLE}{_RECORDING_SELECTOR_STYLE}</style></head><body data-page='{page}'>{body}<button id='theme-toggle' type='button' aria-label='Toggle color theme'></button><script>{_SCRIPT}</script></body></html>")


def _mint_reactor_token(api_key: str, max_sessions: int) -> dict[str, Any]:
    """Mint a model-scoped JWT without exposing the long-lived API key."""
    payload = json.dumps({"authorization_details": [{"type": "session", "resources": {"models": {"match": ["reactor/lingbot-world-2"]}}, "constraints": {"max_sessions": max_sessions}}]}).encode()
    request = URLRequest(
        "https://api.reactor.inc/tokens",
        data=payload,
        headers={"Reactor-API-Key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            token = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise HTTPException(502, "Unable to obtain a Reactor session token") from exc
    if not isinstance(token.get("jwt"), str):
        raise HTTPException(502, "Reactor returned an invalid session token")
    return {"jwt": token["jwt"], "model": "reactor/lingbot-world-2"}


def create_app(store: ExperimentStore) -> FastAPI:
    app = FastAPI(title="Physical AI Failure Harness", docs_url=None, redoc_url=None)
    research = ResearchCampaignStore(store.database_path)
    paired_capture = PairedCaptureService(store, store.database_path.parent)
    app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return _page(_LIBRARY_BODY, "library")

    @app.get("/reactor", response_class=HTMLResponse)
    def reactor_page() -> HTMLResponse:
        return _page(_REACTOR_BODY, "reactor")

    @app.get("/agent", response_class=HTMLResponse)
    def agent_page() -> HTMLResponse:
        return _page(_AGENT_BODY, "agent")

    @app.get("/api/reactor/live-config")
    def reactor_live_config() -> dict[str, Any]:
        return {"enabled": bool(os.environ.get("REACTOR_API_KEY")), "model": "reactor/lingbot-world-2"}

    @app.post("/api/reactor/token")
    def reactor_token() -> dict[str, Any]:
        api_key = os.environ.get("REACTOR_API_KEY")
        if not api_key:
            raise HTTPException(503, "REACTOR_API_KEY is not configured on the dashboard server")
        try:
            max_sessions = int(os.environ.get("REACTOR_MAX_LIVE_SESSIONS", "1"))
        except ValueError as exc:
            raise HTTPException(500, "REACTOR_MAX_LIVE_SESSIONS must be an integer") from exc
        if max_sessions < 1:
            raise HTTPException(500, "REACTOR_MAX_LIVE_SESSIONS must be at least one")
        return _mint_reactor_token(api_key, max_sessions)

    @app.get("/api/isaac-recordings")
    def isaac_recordings() -> list[dict[str, Any]]:
        return [record for record in store.list_experiments() if record["backend"] == "isaac_sim"]

    @app.post("/api/pair-captures", status_code=201)
    def prepare_pair_capture(payload: PairCaptureCreate) -> dict[str, Any]:
        model = payload.model or os.environ.get("AGENT_MODEL") or os.environ.get("RESEARCH_MODEL")
        if not model:
            raise HTTPException(503, "AGENT_MODEL or RESEARCH_MODEL must be configured")
        try:
            return paired_capture.prepare(payload.isaac_run_id, objective=payload.objective, model=model)
        except (PairingError, RuntimeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/pair-captures/{pair_id}/seed")
    def pair_capture_seed(pair_id: str) -> FileResponse:
        try:
            return FileResponse(paired_capture.seed_image(pair_id))
        except PairingError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/pair-captures/{pair_id}")
    def prepared_pair_capture(pair_id: str) -> dict[str, Any]:
        try:
            return paired_capture.prepared_capture(pair_id)
        except PairingError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/pair-captures/{pair_id}/recording", status_code=201)
    async def finish_pair_capture(pair_id: str, request: Request) -> dict[str, Any]:
        try:
            return paired_capture.finalize(pair_id, await request.body(), content_type=request.headers.get("content-type", "video/webm"))
        except PairingError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/experiments/{run_id}", response_class=HTMLResponse)
    def experiment_page(run_id: str) -> HTMLResponse:
        if not store.get_experiment(run_id):
            raise HTTPException(404, "experiment not found")
        return _page(_EXPERIMENT_BODY, "experiment")

    @app.get("/pairs/{pair_id}", response_class=HTMLResponse)
    def pair_page(pair_id: str) -> HTMLResponse:
        if not store.get_pair(pair_id):
            raise HTTPException(404, "pair not found")
        return _page(_PAIR_BODY, "pair")

    @app.get("/api/experiments")
    def experiments() -> list[dict[str, Any]]:
        return store.list_experiments()

    def agent_service() -> MineFailureResearchService:
        model = os.environ.get("AGENT_MODEL") or os.environ.get("RESEARCH_MODEL")
        if not model:
            raise HTTPException(503, "AGENT_MODEL or RESEARCH_MODEL must be configured")
        return MineFailureResearchService(store.database_path.parent.parent, model=model)

    @app.post("/api/agent/campaigns", status_code=201)
    def create_agent_campaign(payload: AgentCampaignCreate) -> dict[str, Any]:
        try:
            model = os.environ.get("AGENT_MODEL") or os.environ.get("RESEARCH_MODEL")
            if not model:
                raise HTTPException(503, "AGENT_MODEL or RESEARCH_MODEL must be configured")
            campaign_id = research.create_campaign(
                payload.objective,
                experiment_budget=payload.experiment_budget,
                model_provider="openai_agents_sdk",
                model_name=model,
                capability_version="mine_v1-agents-sdk-v1",
                simulator_metadata={"container_image": "nvcr.io/nvidia/isaac-sim:6.0.1"},
            )
            return research.get_campaign(campaign_id) or {}
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/agent/campaigns")
    def agent_campaigns() -> list[dict[str, Any]]:
        return [
            row for row in research.list_campaigns()
            if row.get("model_provider") == "openai_agents_sdk"
        ]

    @app.get("/api/agent/campaigns/{campaign_id}")
    def agent_campaign(campaign_id: str) -> dict[str, Any]:
        result = research.get_campaign(campaign_id)
        if not result:
            raise HTTPException(404, "campaign not found")
        return {
            **result,
            "latest_iteration": research.latest_iteration(campaign_id),
            "events": research.list_events(campaign_id),
        }

    @app.get("/api/agent/campaigns/{campaign_id}/events")
    def agent_campaign_events(campaign_id: str) -> list[dict[str, Any]]:
        if not research.get_campaign(campaign_id):
            raise HTTPException(404, "campaign not found")
        return research.list_events(campaign_id)

    async def run_agent_step(campaign_id: str, instruction: str) -> dict[str, Any]:
        if not instruction.strip():
            raise HTTPException(400, "instruction must not be empty")
        try:
            result = await agent_service().run_step(campaign_id, instruction.strip())
            return result.model_dump(mode="json")
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/agent/campaigns/{campaign_id}/step")
    async def agent_step(campaign_id: str, payload: AgentStepCreate) -> dict[str, Any]:
        return await run_agent_step(campaign_id, payload.instruction)

    @app.post("/api/agent/campaigns/{campaign_id}/continue")
    async def continue_agent(campaign_id: str, payload: AgentStepCreate) -> dict[str, Any]:
        return await run_agent_step(campaign_id, payload.instruction)

    @app.get("/api/overview")
    def overview() -> dict[str, list[dict[str, Any]]]:
        pairs = store.list_pairs()
        paired_runs = {run_id for pair in pairs for run_id in (pair["isaac_run_id"], pair["reactor_run_id"])}
        return {"pairs": pairs, "unpaired_experiments": [item for item in store.list_experiments() if item["run_id"] not in paired_runs]}

    @app.get("/api/experiments/{run_id}")
    def experiment_detail(run_id: str) -> dict[str, Any]:
        detail = build_experiment_view(store, run_id)
        if not detail:
            raise HTTPException(404, "experiment not found")
        return detail

    @app.get("/api/pairs")
    def pairs() -> list[dict[str, Any]]:
        return store.list_pairs()

    @app.get("/api/pairs/{pair_id}")
    def pair_detail(pair_id: str) -> dict[str, Any]:
        detail = build_pair_view(store, pair_id)
        if not detail:
            raise HTTPException(404, "pair not found")
        return detail

    @app.put("/api/pairs/{pair_id}/review")
    def update_review(pair_id: str, update: ReviewUpdate) -> dict[str, str]:
        try:
            store.set_review_state(pair_id, update.review_state)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"pair_id": pair_id, "review_state": update.review_state.value}

    @app.get("/artifacts/{artifact_id}")
    def artifact(artifact_id: int) -> FileResponse:
        match = next((item for pair in store.list_pairs() for item in store.artifacts_for("pair", pair["pair_id"]) if item["artifact_id"] == artifact_id), None)
        if not match:
            match = next((item for record in store.list_experiments() for item in store.artifacts_for("experiment", record["run_id"]) if item["artifact_id"] == artifact_id), None)
        if not match or not Path(match["path"]).is_file():
            raise HTTPException(404, "artifact not available")
        return FileResponse(match["path"])

    return app
