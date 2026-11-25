#!/usr/bin/env python3
"""Run local transport fault cases and retain unperformed physical/WAN gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('baseline_tcp', []),
    ('delayed_tcp', ['--proxy-delay-ms', '20', '--proxy-jitter-ms', '5']),
    ('bandwidth_tcp', ['--proxy-delay-ms', '10', '--proxy-bytes-per-second', '65536']),
    ('worker_disconnect', ['--proxy-delay-ms', '25', '--interrupt-worker-after-token', '2']),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default=str(Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target')) / 'debug/synapse'))
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / f'transport-matrix-{time.time_ns()}')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, extra in CASES:
        directory = args.output / name
        command = [sys.executable, str(ROOT / 'scripts/local_cluster.py'), '--binary', args.binary,
            '--workers', '1', '--transport', 'tcp', '--host-memory-bytes', '52000',
            '--worker-memory-bytes', '45000', '--timeout', '30', '--output', str(directory)] + extra
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45)
        run_path = directory / 'run.json'
        run = json.loads(run_path.read_text()) if run_path.exists() else {'success':False, 'error':'harness did not retain a run record'}
        fault = name == 'worker_disconnect'
        accepted = (result.returncode == 0 and run.get('success')) if not fault else (
            result.returncode != 0 and not run.get('success') and run.get('injected_fault')
            and run.get('partial_output_contiguous') and any(event.get('reason') == 'error' for event in run.get('terminal_events', []))
            and run.get('wall_seconds', 999) < 35)
        records.append({'case':name, 'expected_fault':fault, 'check_passed':bool(accepted),
                        'inference_success':run.get('success'), 'run_directory':str(directory), 'error':run.get('error')})
        print(json.dumps(records[-1]), flush=True)
    report = {'schema_version':1, 'environment':'one physical host, loopback TCP proxy', 'cases':records,
        'all_local_checks_passed':all(r['check_passed'] for r in records),
        'physical_lan':'not_tested', 'wifi':'not_tested', 'direct_wan':'not_tested', 'relayed_wan':'not_tested',
        'inter_city':'not_tested', 'ip_packet_loss':'not_tested'}
    (args.output / 'matrix.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['all_local_checks_passed'] else 1

if __name__ == '__main__': raise SystemExit(main())
