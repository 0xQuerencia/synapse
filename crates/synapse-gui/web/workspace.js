'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let mode = location.pathname === '/contribute' ? 'worker' : 'host';
  let current = null, socket = null, socketPeer = null, working = false, apiRequest = null, localGenerating = false;
  let lastError = '', pollRunning = false;
  const bytes = value => value == null ? '—' : value < 1048576 ? `${(value / 1024).toFixed(1)} KiB` : `${(value / 1048576).toFixed(1)} MiB`;
  const range = item => !item ? 'Not loaded' : item.start === item.end ? 'Coordinator · I/O only' : item.end === item.start + 1 ? `Layer ${item.start}` : `Layers ${item.start}–${item.end - 1}`;
  const integer = id => $(id).value === '' ? null : Number($(id).value);
  function notice(message, error = false) { $('notice').textContent = message; $('notice').classList.toggle('error', error); $('notice').hidden = !message; }
  async function request(path, body, signal) {
    const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin', headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body), signal});
    if (response.status === 401) { location.assign('/login'); throw new Error('Sign in to continue.'); }
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : data.error?.message || `Request failed (${response.status})`);
    return data;
  }
  function selectMode(next) {
    mode = next;
    for (const tab of document.querySelectorAll('[data-mode]')) { const selected = tab.dataset.mode === mode; tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1; }
    $('role-form').setAttribute('aria-labelledby', mode === 'host' ? 'host-tab' : 'worker-tab');
    $('join-fields').hidden = mode !== 'worker'; $('duration-field').hidden = mode !== 'worker';
    $('output-field').hidden = mode !== 'host'; $('host-options').hidden = mode !== 'host';
    $('master-address').required = mode === 'worker'; $('pairing-code').required = mode === 'worker';
    $('max-tokens').required = mode === 'host'; $('start').textContent = mode === 'host' ? 'Start host' : 'Connect this device';
    $('listen').value = mode === 'host' ? '0.0.0.0:4001' : '0.0.0.0:0';
  }
  document.querySelectorAll('[data-mode]').forEach(tab => {
    tab.addEventListener('click', () => selectMode(tab.dataset.mode));
    tab.addEventListener('keydown', event => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) { event.preventDefault(); selectMode(event.key === 'Home' ? 'host' : event.key === 'End' ? 'worker' : mode === 'host' ? 'worker' : 'host'); $(mode === 'host' ? 'host-tab' : 'worker-tab').focus(); } });
  });
  function render(data) {
    current = data;
    const session = data.session, node = data.node, active = session.role !== 'idle';
    $('connection').textContent = 'Device connected'; $('connection').classList.add('connected');
    $('role').textContent = active ? (session.running ? session.role === 'master' ? 'Host' : session.role === 'worker' ? 'Worker' : 'Relay' : 'Stopped') : 'Idle';
    $('backend').textContent = active ? session.backend === 'ollama_local' ? 'Local Ollama · no workers' : 'Candle · private membership' : 'Ready to configure';
    $('peers').textContent = node?.authenticated_count ?? 0;
    $('budget').textContent = bytes(node?.memory_budget_bytes || null); $('resident').textContent = bytes(node?.resident_weight_bytes);
    $('session-badge').textContent = active ? session.running ? 'Running' : 'Ended' : 'Idle';
    $('role-form').hidden = active; document.querySelector('.tabs').hidden = active; $('running-controls').hidden = !active;
    $('session-description').textContent = active ? `${session.model_id} · ${session.running ? 'session is running' : 'session ended; stop to reset'}` : '';
    $('worker-controls').hidden = session.role !== 'worker';
    $('worker-state').textContent = node?.worker_state ? `Worker ${node.worker_state.state} · ${node.worker_state.active_jobs} active job(s)${session.drain_at_unix ? ' · drains at ' + new Date(session.drain_at_unix * 1000).toLocaleString() : ''}` : '';
    for (const button of document.querySelectorAll('[data-action]')) button.disabled = working || !session.running;
    $('stop').disabled = working; $('start').disabled = working;
    $('peer-id').textContent = session.peer_id || data.peer_id || '—'; $('digest').textContent = node?.model_digest || '—';
    $('owned-range').textContent = range(node?.owned_layer_range); $('reserved').textContent = bytes(node?.memory_estimate?.total_bytes);
    $('share').hidden = !(session.running && session.pairing_code);
    $('share-code').textContent = session.pairing_code || '';
    $('share-addresses').textContent = (node?.listen_addresses || []).map(address => address.includes('/p2p/') ? address : `${address}/p2p/${session.peer_id}`).join('\n') || 'Waiting for an observed listen address…';
    const route = node?.route_plan;
    $('route-list').replaceChildren(); $('route-empty').hidden = !!route;
    $('route-empty').textContent = !active ? 'Start a session to see authenticated peers and the active layer route.' : node?.authenticated_count ? 'Peers authenticated. The route is planned when you submit a prompt.' : 'Waiting for authenticated peers. Use the host connection details and check both model manifests.';
    if (route) {
      const participants = [{layer_range: route.master_layer_range, peer_id: session.peer_id, host: true}, ...route.worker_hops];
      for (const participant of participants) { const item = document.createElement('li'); const title = document.createElement('strong'); const peer = document.createElement('span'); title.textContent = range(participant.layer_range); peer.textContent = `${participant.host ? 'This host' : 'Worker'} · ${participant.peer_id.slice(0, 14)}…`; item.append(title, peer); $('route-list').append(item); }
    }
    const apiOption = $('inference-source').querySelector('[value="api"]');
    apiOption.disabled = !data.api_model; apiOption.textContent = data.api_model ? `Local API · ${data.api_model}` : 'Local API model (not configured)';
    $('chat-mode').textContent = $('inference-source').value === 'api' ? 'Local full model' : session.role === 'master' ? 'Host session' : 'No host';
    $('generate').disabled = localGenerating || ( $('inference-source').value === 'api' ? !data.api_model : session.role !== 'master' || !session.running || session.generating );
    if (session.last_error && session.last_error !== lastError) { lastError = session.last_error; notice(session.last_error, true); }
    if (!(session.role === 'master' && session.running) && socket) { socket.close(); socket = null; socketPeer = null; }
  }
  async function poll() {
    if (pollRunning || working) return;
    pollRunning = true;
    try { const data = await request('/api/status', undefined, AbortSignal.timeout(10000)); render(data);
      if (!$('models').children.length) for (const model of data.cached_models || []) { const option = document.createElement('option'); option.value = model; $('models').append(option); }
    } catch (error) { $('connection').textContent = 'Status unavailable'; $('connection').classList.remove('connected'); notice(error.message, true); }
    finally { pollRunning = false; }
  }
  async function action(callback) {
    if (working) return; working = true; if (current) render(current);
    try { await callback(); } catch (error) { notice(error.message, true); }
    finally { working = false; await poll(); }
  }
  $('role-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
    notice('Verifying model artifacts and starting the session…');
    const body = {model: $('model').value.trim(), listen: $('listen').value.trim(), memory_mib: integer('memory')};
    if (mode === 'host') Object.assign(body, {max_tokens: integer('max-tokens'), raw_prompt: $('raw-prompt').checked, coordinator_only: $('coordinator-only').checked});
    else Object.assign(body, {code: $('pairing-code').value.trim(), master_addr: $('master-address').value.trim(), duration_minutes: integer('duration')});
    await request(`/api/session/${mode === 'host' ? 'host' : 'join'}`, body);
    $('pairing-code').value = ''; lastError = ''; notice('Session started. Authentication and shard readiness are shown in live status.');
  }); });
  $('stop').addEventListener('click', () => action(async () => { await request('/api/session/stop', {}); if (socket) socket.close(); socket = null; notice('Session stopped and local tasks released.'); }));
  document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => action(async () => { const result = await request('/api/worker/control', {action: button.dataset.action}); notice(`Worker acknowledged: ${result.state}; ${result.active_jobs} active job(s).`); })));
  $('logout').addEventListener('click', () => action(async () => { await request('/api/logout', {}); location.assign('/login'); }));
  $('inference-source').addEventListener('change', () => { $('chat-help').textContent = $('inference-source').value === 'api' ? 'This request runs on the explicitly configured local API model. It does not use the distributed route.' : 'Start a host, then connect a trusted worker. Exact layer coverage and readiness are checked before inference.'; if (current) render(current); });
  function socketReady() {
    if (socket?.readyState === WebSocket.OPEN && socketPeer === current?.session.peer_id) return Promise.resolve();
    if (socket) socket.close();
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`); socket = ws; socketPeer = current.session.peer_id;
      let ready = false;
      const timeout = setTimeout(() => { ws.close(); reject(new Error('Live stream did not become ready.')); }, 5000);
      ws.addEventListener('message', event => {
        let message; try { message = JSON.parse(event.data); } catch { notice('Invalid live stream message.', true); return; }
        if (message.type === 'hello') { ready = true; clearTimeout(timeout); resolve(); }
        if (message.type === 'error') { clearTimeout(timeout); notice(message.message, true); if (!ready) reject(new Error(message.message)); $('generation-status').textContent = 'Output incomplete · stream error'; }
        const item = message.event;
        if (item?.type === 'token') { $('answer').textContent += item.text; $('answer').scrollTop = $('answer').scrollHeight; }
        if (item?.type === 'generation_done') { $('generation-status').textContent = `${item.tokens_generated} output tokens · ${item.reason === 'Error' ? 'failed' : item.reason === 'Eos' ? 'end of sequence' : 'output limit reached'}`; if (item.reason === 'Error') notice('Generation failed. Check the model, available memory, and worker connection. Partial output is incomplete.', true); poll(); }
      });
      ws.addEventListener('close', () => { clearTimeout(timeout); if (!ready) reject(new Error('Live stream closed before readiness.')); if (current?.session.generating) { $('generation-status').textContent = 'Stream disconnected · output may be incomplete'; } if (socket === ws) { socket = null; socketPeer = null; } });
      ws.addEventListener('error', () => { clearTimeout(timeout); if (!ready) reject(new Error('Could not open the authenticated live stream.')); });
    });
  }
  async function generateLocal(prompt) {
    apiRequest = new AbortController(); $('cancel').hidden = false;
    const response = await fetch('/v1/chat/completions', {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'}, signal: apiRequest.signal,
      body: JSON.stringify({model: current.api_model, messages: [{role: 'user', content: prompt}], max_tokens: integer('max-tokens') || 8, temperature: 0, stream: true, stream_options: {include_usage: true}})});
    if (!response.ok) { const data = await response.json(); throw new Error(data.error?.message || `API request failed (${response.status})`); }
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '', finish = null, usage = null, done = false;
    try {
      while (true) {
        const chunk = await reader.read(); if (chunk.done) break;
        buffer += decoder.decode(chunk.value, {stream: true});
        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
          const value = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
          if (!value) continue; if (value === '[DONE]') { done = true; continue; }
          const data = JSON.parse(value); if (data.error) throw new Error(data.error.message || 'Generation failed');
          const choice = data.choices?.[0]; if (choice?.delta?.content) $('answer').textContent += choice.delta.content;
          if (choice?.finish_reason) finish = choice.finish_reason; if (data.usage) usage = data.usage;
        }
      }
      if (!done || !finish) throw new Error('The stream ended without a successful completion. Partial output is incomplete.');
      $('generation-status').textContent = `${usage?.completion_tokens ?? 'Unreported'} output tokens · ${finish === 'length' ? 'output limit reached' : 'completed'}`;
    } finally { reader.releaseLock(); }
  }
  $('prompt-form').addEventListener('submit', event => { event.preventDefault(); if (localGenerating) return;
    (async () => {
      const prompt = $('prompt').value; localGenerating = true; $('answer').textContent = ''; $('generation-status').textContent = 'Waiting for the first token…'; notice(''); if (current) render(current);
      try { if ($('inference-source').value === 'api') await generateLocal(prompt); else { await socketReady(); await request('/api/session/prompt', {text: prompt}); } }
      catch (error) { const cancelled = error.name === 'AbortError'; notice(cancelled ? 'Local request cancelled. Partial output is incomplete.' : error.message, !cancelled); $('generation-status').textContent = cancelled ? 'Cancelled' : 'Request failed'; }
      finally { localGenerating = false; apiRequest = null; $('cancel').hidden = true; await poll(); }
    })();
  });
  $('cancel').addEventListener('click', () => apiRequest?.abort());
  window.addEventListener('pagehide', () => { apiRequest?.abort(); socket?.close(); });
  selectMode(mode); poll(); setInterval(poll, 2000);
})();
