const trace = document.querySelector('.trace');
const controls = trace.querySelector('.trace-controls');
const steps = Array.from(controls.querySelectorAll('[data-trace-step]'));
const panels = trace.querySelectorAll('[data-trace-panel]');

for (const button of steps) {
  button.addEventListener('click', () => {
    const step = button.dataset.traceStep;
    trace.dataset.step = step;

    for (const option of steps) {
      option.setAttribute('aria-pressed', String(option === button));
    }
    for (const panel of panels) {
      panel.hidden = panel.dataset.tracePanel !== step;
    }
  });
}

trace.querySelector('.trace-fallback').hidden = true;
controls.hidden = false;
