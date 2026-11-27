import { chooseRoute, defaultDevices, MiB } from './simulator.mjs';
const form = document.querySelector('#explorer-form');
form.querySelectorAll('fieldset').forEach(fieldset => { fieldset.disabled = false; });
const status = document.querySelector('#route-status');
const output = document.querySelector('#route-output');
const labels = { local: 'Run on your laptop', remote: 'Use one full-model host', sharded: 'Split across trusted devices' };
const formatGiB = bytes => (bytes / (1024 * MiB)).toFixed(2);

function text(tag, value, className = '') {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

function update() {
  const devices = defaultDevices.map(device => ({ ...device, enabled: form.elements[device.id + '-enabled'].checked, budgetGiB: Number(form.elements[device.id + '-budget'].value) }));
  devices.forEach(device => {
    document.querySelector('#' + device.id + '-budget-label').textContent = `${device.budgetGiB} GiB`;
    form.elements[device.id + '-budget'].disabled = !device.enabled;
  });
  const result = chooseRoute({ profileKey: form.elements.profile.value, networkKey: form.elements.network.value, context: Number(form.elements.context.value), devices });
  output.replaceChildren();
  document.querySelector('#weight-value').textContent = formatGiB(result.totalWeightsBytes) + ' GiB';
  if (!result.route) {
    status.textContent = 'No feasible example route';
    output.append(text('p', 'The enabled devices cannot fit a complete route under these individual budgets. Try more memory, a smaller profile or a shorter context.', 'empty-state'));
    output.append(text('p', 'No request was sent and no credits were reserved.', 'muted'));
    return;
  }
  const route = result.route;
  status.textContent = labels[route.kind];
  const stages = document.createElement('ol');
  stages.className = 'route-stages';
  route.placements.forEach(device => {
    const stage = document.createElement('li');
    stage.append(text('strong', device.label));
    stage.append(text('span', device.end > device.start ? `Layers ${device.start}–${device.end - 1}` : 'Coordinator · no layers', 'stage-range'));
    stage.append(text('span', `${formatGiB(device.bytes)} / ${device.budgetGiB} GiB`, 'stage-memory'));
    const meter = document.createElement('meter');
    meter.min = 0;
    meter.max = device.budgetGiB * 1024 * MiB;
    meter.value = device.bytes;
    meter.setAttribute('aria-label', `${device.label} estimated inference memory`);
    stage.append(meter);
    stage.append(text('small', device.ownsIO ? 'Includes input/output weights, KV and scratch.' : 'Includes assigned weights, KV and scratch.'));
    stages.append(stage);
  });
  output.append(stages);
  const estimate = text('p', `${route.estimatedMs.toFixed(1)} ms/token in this fictional cost model.`, 'estimate');
  output.append(estimate);
  output.append(text('p', route.kind === 'sharded' ? 'The coordinator keeps embedding and output weights. Every transformer layer is covered once; each device fits its own budget.' : 'One device can hold all the required weights and state. A full-model remote host streams text; it does not send activations across a multi-stage route.', 'result-explanation'));
}
form.addEventListener('input', update);
form.addEventListener('change', update);
update();
