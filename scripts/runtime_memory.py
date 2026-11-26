#!/usr/bin/env python3
"""Measure repeated local GUI host lifecycles with the original tiny fixture."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

from gui_smoke import GuiServer, check_compatible_api, digest, require
from low_memory import group_snapshot

FIXTURE = Path(__file__).resolve().parents[1] / 'crates/synapse-core/tests/fixtures/tiny-llama'


def idle_snapshot(pid):
    proc = Path('/proc') / str(pid)
    fields = dict(line.split(':', 1) for line in (proc / 'smaps_rollup').read_text().splitlines()[1:])
    return {
        'rss_kib': int(fields['Rss'].split()[0]),
        'anonymous_kib': int(fields['Anonymous'].split()[0]),
        'open_fds': sum(1 for _ in (proc / 'fd').iterdir()),
        'threads': sum(1 for _ in (proc / 'task').iterdir()),
    }


def profile(binary, cycles, root, result):
    server = GuiServer('memory-profile', binary, FIXTURE, root)
    pid = server.process.pid
    try:
        server.ready()
        result['idle_after_ready'] = idle_snapshot(pid)
        for index in range(1, cycles + 1):
            started = server.json('POST', '/api/session/host', {
                'model': str(FIXTURE), 'max_tokens': 8, 'listen': '127.0.0.1:0',
                'memory_mib': 64, 'raw_prompt': True,
            })
            require(started['session']['running'], 'host did not start')
            server.wait_status(lambda status: bool((status.get('node') or {}).get('listen_addresses')),
                               'loopback host listener')
            check_compatible_api(server)
            server.json('POST', '/api/session/stop', {})
            server.wait_status(lambda status: status['session']['role'] == 'idle', 'stopped host')
            time.sleep(0.1)
            result['idle_after_cycles'].append({'cycle': index, **idle_snapshot(pid)})
    finally:
        status = Path(f'/proc/{pid}/status')
        try:
            for line in status.read_text().splitlines():
                if line.startswith('VmHWM:'):
                    result['gui_vm_hwm_kib'] = int(line.split()[1])
                    break
        except FileNotFoundError:
            pass  # Preserve completed cycles if the GUI exited before sampling.
        server.stop()
        server.log_handle.close()
        result['gui_exit_code'] = server.process.returncode
        result['surviving_group_pids'], _ = group_snapshot(pid)
        require(server.cleanup_result == 'exited' and not result['surviving_group_pids'],
                'GUI process group survived cleanup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=24)
    args = parser.parse_args()
    if sys.platform != 'linux' or args.cycles < 1:
        parser.error('requires Linux and a positive cycle count')
    binary = args.binary.resolve(strict=True)
    result = {
        'scope': 'loopback, repeated host start/stop and local gateway inference',
        'binary_sha256': digest(binary),
        'fixture_manifest_sha256': digest(FIXTURE / 'manifest.json'),
        'idle_after_cycles': [],
        'status': 'failed',
    }
    try:
        with tempfile.TemporaryDirectory(prefix='synapse-runtime-memory-') as directory:
            profile(binary, args.cycles, Path(directory), result)
        require('gui_vm_hwm_kib' in result, 'GUI peak RSS was unavailable')
        result['status'] = 'passed'
    finally:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
