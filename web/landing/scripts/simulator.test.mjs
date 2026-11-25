import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseRoute, defaultDevices, memoryBytes, MiB, profiles } from '../assets/simulator.mjs';

test('medium profile has exact layer coverage and individual budgets', () => {
  const result = chooseRoute();
  assert.equal(result.route.kind, 'sharded');
  let next = 0;
  for (const stage of result.route.placements) {
    assert.equal(stage.start, next);
    next = stage.end;
    assert.ok(stage.bytes <= stage.budgetGiB * 1024 * MiB);
  }
  assert.equal(next, result.profile.layers);
  assert.equal(result.route.placements.filter(d => d.ownsIO).length, 1);
});

test('fitting fast full-model host beats slow internet sharding', () => {
  const devices = defaultDevices.map(device => ({ ...device, enabled: true }));
  const result = chooseRoute({ devices, networkKey: 'wan' });
  assert.equal(result.route.kind, 'remote');
  assert.equal(result.route.placements[0].id, 'gpu');
  assert.equal(result.route.placements.length, 1);
});

test('aggregate memory is insufficient evidence of a feasible route', () => {
  // 9.9 GiB total exceeds the medium full model footprint, but every node
  // must also fit its own scratch, and a coordinator must fit fixed weights.
  const devices = defaultDevices.map(d => ({ ...d, enabled: true, budgetGiB: 3.3 }));
  assert.ok(devices.reduce((sum, d) => sum + d.budgetGiB * 1024 * MiB, 0) > memoryBytes(profiles.medium, 32, 2048, true));
  assert.equal(chooseRoute({ devices }).route, null);
  const profile = profiles.medium;
  assert.equal(memoryBytes(profile, 0, 2048, true), (1024 + 384) * MiB);
  assert.equal(chooseRoute({ devices: defaultDevices.map(d => ({ ...d, enabled: d.id === 'laptop', budgetGiB: 1 })) }).route, null);
});

test('context has a real KV memory cost and cannot improve fit', () => {
  const short = memoryBytes(profiles.medium, 32, 512, true);
  const long = memoryBytes(profiles.medium, 32, 4096, true);
  assert.equal(long - short, 32 * (4096 - 512) * profiles.medium.kvBytesPerLayerToken);
  const budgetGiB = short / (1024 * MiB);
  const devices = [{ ...defaultDevices[0], budgetGiB }];
  assert.ok(chooseRoute({ devices, context: 512 }).route);
  assert.equal(chooseRoute({ devices, context: 4096 }).route, null);
});

test('slower link increases sharded decode estimate', () => {
  const fast = chooseRoute({ networkKey: 'wired' });
  const slow = chooseRoute({ networkKey: 'wan' });
  assert.equal(fast.route.kind, 'sharded');
  assert.equal(slow.route.kind, 'sharded');
  assert.ok(slow.route.estimatedMs > fast.route.estimatedMs);
});

test('no devices means no route; disabled devices cannot be used', () => {
  assert.equal(chooseRoute({ devices: defaultDevices.map(d => ({ ...d, enabled: false })) }).route, null);
  assert.ok(chooseRoute().route.placements.every(d => d.id !== 'gpu'));
});

test('all sample profiles and caps retain coverage and fit invariants', () => {
  for (const profileKey of Object.keys(profiles)) for (const context of [512, 2048, 4096]) for (const networkKey of ['wired', 'wifi', 'wan']) for (const budget of [1, 4, 8, 12, 24]) {
    const result = chooseRoute({ profileKey, context, networkKey, devices: defaultDevices.map(d => ({ ...d, enabled: true, budgetGiB: budget })) });
    if (!result.route) continue;
    let next = 0;
    for (const stage of result.route.placements) {
      assert.equal(stage.start, next);
      assert.ok(stage.end >= stage.start);
      assert.ok(stage.bytes <= stage.budgetGiB * 1024 * MiB);
      next = stage.end;
    }
    assert.equal(next, result.profile.layers);
  }
});

test('invalid or unbounded examples are rejected', () => {
  assert.throws(() => chooseRoute({ context: Infinity }));
  assert.throws(() => chooseRoute({ devices: [{ ...defaultDevices[0], budgetGiB: NaN }] }));
  assert.throws(() => chooseRoute({ devices: [defaultDevices[0], defaultDevices[0]] }));
});
