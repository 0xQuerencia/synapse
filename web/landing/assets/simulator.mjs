// Pure, deterministic resource model. These are fictional profiles, not benchmarks.
export const MiB = 1024 * 1024;
export const profiles = Object.freeze({
  small: { label: 'Small profile', layers: 24, layerMiB: 128, fixedMiB: 512, kvBytesPerLayerToken: 2048, scratchMiB: 256, activationBytes: 8192 },
  medium: { label: 'Medium profile', layers: 32, layerMiB: 256, fixedMiB: 1024, kvBytesPerLayerToken: 4096, scratchMiB: 384, activationBytes: 16384 },
  large: { label: 'Large profile', layers: 40, layerMiB: 384, fixedMiB: 1536, kvBytesPerLayerToken: 8192, scratchMiB: 512, activationBytes: 24576 },
});
export const networks = Object.freeze({
  wired: { label: 'Wired LAN', rttMs: 2, mibPerSecond: 100 },
  wifi: { label: 'Home Wi-Fi', rttMs: 8, mibPerSecond: 25 },
  wan: { label: 'Inter-city internet', rttMs: 60, mibPerSecond: 8 },
});
export const defaultDevices = Object.freeze([
  { id: 'laptop', label: 'Your laptop', budgetGiB: 4, layerMs: 5, enabled: true },
  { id: 'desktop', label: 'Trusted desktop', budgetGiB: 8, layerMs: 2.2, enabled: true },
  { id: 'gpu', label: 'Remote GPU host', budgetGiB: 12, layerMs: 0.7, enabled: false },
]);

export function memoryBytes(profile, layers, context, ownsIO = false) {
  return layers * (profile.layerMiB * MiB + context * profile.kvBytesPerLayerToken)
    + profile.scratchMiB * MiB + (ownsIO ? profile.fixedMiB * MiB : 0);
}

function permutations(values) {
  if (values.length === 0) return [[]];
  return values.flatMap((value, i) => permutations(values.filter((_, j) => i !== j)).map(rest => [value, ...rest]));
}

export function chooseRoute({ profileKey = 'medium', networkKey = 'wifi', context = 2048, devices = defaultDevices } = {}) {
  const profile = profiles[profileKey];
  const network = networks[networkKey];
  if (!profile || !network || ![512, 2048, 4096].includes(context)) throw new Error('Unsupported example configuration');
  if (devices.length > 3 || new Set(devices.map(d => d.id)).size !== devices.length) throw new Error('Example requires at most three unique devices');
  for (const device of devices) {
    if (!['laptop', 'desktop', 'gpu'].includes(device.id) || !Number.isFinite(device.budgetGiB) || device.budgetGiB < 1 || device.budgetGiB > 24 || !Number.isFinite(device.layerMs) || device.layerMs <= 0) throw new Error('Invalid example device');
  }
  const enabled = devices.filter(device => device.enabled);
  const candidates = [];
  const boundaryMs = network.rttMs + 2 * profile.activationBytes / MiB / network.mibPerSecond * 1000;
  for (const device of enabled) {
    const bytes = memoryBytes(profile, profile.layers, context, true);
    if (bytes <= device.budgetGiB * 1024 * MiB) {
      candidates.push({ kind: device.id === 'laptop' ? 'local' : 'remote', placements: [{ ...device, start: 0, end: profile.layers, bytes, ownsIO: true }], estimatedMs: profile.layers * device.layerMs + 3 });
    }
  }
  const host = enabled.find(device => device.id === 'laptop');
  const workers = enabled.filter(device => device.id !== 'laptop');
  if (host) {
    // Coordinator owns embedding/head, even when assigned zero transformer layers.
    // Enumerate a bounded ordered subset of workers; each gets a contiguous slice.
    const routes = workers.map(device => [device]);
    if (workers.length === 2) routes.push(...permutations(workers));
    for (const workerRoute of routes) {
      for (let hostLayers = 0; hostLayers < profile.layers; hostLayers += 1) {
        const hostBytes = memoryBytes(profile, hostLayers, context, true);
        if (hostBytes > host.budgetGiB * 1024 * MiB) continue;
        const visit = (index, nextLayer, placements) => {
          if (index === workerRoute.length) {
            if (nextLayer !== profile.layers) return;
            const computeMs = placements.reduce((sum, d) => sum + (d.end - d.start) * d.layerMs, 0);
            candidates.push({ kind: 'sharded', placements, estimatedMs: computeMs + 3 + workerRoute.length * boundaryMs });
            return;
          }
          const worker = workerRoute[index];
          const remainingWorkers = workerRoute.length - index - 1;
          const maxEnd = profile.layers - remainingWorkers;
          for (let end = nextLayer + 1; end <= maxEnd; end += 1) {
            if (remainingWorkers === 0 && end !== profile.layers) continue;
            const bytes = memoryBytes(profile, end - nextLayer, context, false);
            if (bytes > worker.budgetGiB * 1024 * MiB) break;
            visit(index + 1, end, [...placements, { ...worker, start: nextLayer, end, bytes, ownsIO: false }]);
          }
        };
        visit(0, hostLayers, [{ ...host, start: 0, end: hostLayers, bytes: hostBytes, ownsIO: true }]);
      }
    }
  }
  candidates.sort((a, b) => a.estimatedMs - b.estimatedMs || a.placements.length - b.placements.length || a.placements.map(d => d.id).join().localeCompare(b.placements.map(d => d.id).join()));
  return { profile, network, context, route: candidates[0] ?? null, totalWeightsBytes: (profile.layers * profile.layerMiB + profile.fixedMiB) * MiB, candidateCount: candidates.length };
}
