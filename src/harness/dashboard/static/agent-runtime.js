const campaign = document.querySelector('#agent-campaign');
const message = document.querySelector('#agent-message');
const summary = document.querySelector('#agent-summary');
const events = document.querySelector('#agent-events');
const links = document.querySelector('#agent-links');

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

async function loadCampaigns(selectId) {
  const rows = await request('/api/campaigns');
  campaign.replaceChildren(new Option('Create or select a campaign…', ''));
  rows.filter(row => row.model_provider === 'openai_agents_sdk').forEach(row => {
    campaign.append(new Option(`${row.objective.slice(0, 60)} · ${row.status}`, row.campaign_id));
  });
  if (selectId) campaign.value = selectId;
  if (campaign.value) await refresh();
}

async function refresh() {
  if (!campaign.value) return;
  const state = await request(`/api/agent/campaigns/${encodeURIComponent(campaign.value)}`);
  summary.textContent = `${state.status} · ${state.experiments_used}/${state.experiment_budget} experiments · ${state.objective}`;
  events.textContent = state.events.slice(-30).map(event => {
    const tool = event.payload?.tool_name ? ` ${event.payload.tool_name}` : '';
    return `${event.created_at}  ${event.event_type}${tool}`;
  }).join('\n');
  links.replaceChildren();
  const iteration = state.latest_iteration;
  if (iteration?.isaac_run_id) {
    const anchor = document.createElement('a'); anchor.className = 'primary-action';
    anchor.href = `/experiments/${encodeURIComponent(iteration.isaac_run_id)}`; anchor.textContent = 'Open latest Isaac run'; links.append(anchor);
  }
  if (iteration?.plan_c_pair_id) {
    const anchor = document.createElement('a'); anchor.className = 'primary-action';
    anchor.href = iteration.reactor_run_id ? `/pairs/${encodeURIComponent(iteration.plan_c_pair_id)}` : '/reactor';
    anchor.textContent = iteration.reactor_run_id ? 'Open pair comparison' : 'Complete Reactor capture'; links.append(anchor);
  }
}

document.querySelector('#agent-create').onclick = async () => {
  try {
    message.textContent = 'Creating…';
    const row = await request('/api/agent/campaigns', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({objective: document.querySelector('#agent-objective').value, experiment_budget: Number(document.querySelector('#agent-budget').value)})});
    await loadCampaigns(row.campaign_id); message.textContent = 'Campaign created.';
  } catch (error) { message.textContent = error.message; }
};

async function run(path, fallback) {
  if (!campaign.value) { message.textContent = 'Select or create a campaign first.'; return; }
  try {
    message.textContent = 'Agent is running; an Isaac tool call can take several minutes…';
    const instruction = document.querySelector('#agent-instruction').value || fallback;
    const result = await request(`/api/agent/campaigns/${encodeURIComponent(campaign.value)}/${path}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({instruction})});
    summary.textContent = `${result.status}: ${result.evidence_summary}`;
    message.textContent = result.user_action_required || 'Research step completed.';
    await refresh();
  } catch (error) { message.textContent = error.message; await refresh(); }
}

document.querySelector('#agent-step').onclick = () => run('step', 'Run one research step.');
document.querySelector('#agent-continue').onclick = () => run('continue', 'Continue the research campaign.');
campaign.onchange = refresh;
loadCampaigns().catch(error => { message.textContent = error.message; });
