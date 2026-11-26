#!/usr/bin/env python3
"""Run one locked Cargo command with Linux process-group memory evidence."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SECONDS = 0.1
GRACE_SECONDS = 5


def group_snapshot(group):
    members = []
    rss_kib = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / 'stat').read_text(errors='replace').rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == group and int(fields[3]) == group:
            members.append(int(entry.name))
            rss_kib += max(0, int(fields[21])) * (os.sysconf('SC_PAGE_SIZE') // 1024)
    return sorted(members), rss_kib


def signal_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def reap_adopted_children():
    # The original Cargo child is reaped separately with wait4 for its rusage.
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'build', 'test', 'clippy'))
    options, cargo_args = parser.parse_known_args()
    build_args = cargo_args[:cargo_args.index('--')] if '--' in cargo_args else cargo_args
    if any(arg.startswith(('-j', '--jobs', '--target-dir', '--config')) for arg in build_args):
        parser.error('use the fixed one-job configuration and shared low-memory target directory')
    if sys.platform != 'linux':
        parser.error('memory and process checks require Linux /proc')

    # Adopt and reap orphaned descendants if Cargo exits before its children.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')

    cancellation = None

    def cancel(sig, _frame):
        nonlocal cancellation
        if cancellation is None:
            cancellation = sig

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cancel)

    target = ROOT / 'target' / 'low-memory'
    env = os.environ | {
        'CARGO_BUILD_JOBS': '1',
        'CARGO_INCREMENTAL': '0',
        'CARGO_PROFILE_DEV_DEBUG': '0',
        'CARGO_PROFILE_TEST_DEBUG': '0',
        'CARGO_TARGET_DIR': str(target),
    }
    command = ['cargo', options.command, '--locked', '--jobs', '1', *cargo_args]
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    peak_group_kib = 0
    usage = None
    lingered = []
    stopping_at = None
    killed_at = None
    while True:
        if usage is None:
            pid, status, result = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                usage = result
        if usage is not None:
            reap_adopted_children()
        survivors, rss_kib = group_snapshot(process.pid)
        peak_group_kib = max(peak_group_kib, rss_kib)
        if usage is not None and not survivors:
            break

        now = time.monotonic()
        if stopping_at is None and (cancellation is not None or usage is not None):
            if cancellation is None:
                lingered = survivors
            signal_group(process.pid, cancellation or signal.SIGTERM)
            stopping_at = now
        elif stopping_at is not None and killed_at is None and now - stopping_at >= GRACE_SECONDS:
            signal_group(process.pid, signal.SIGKILL)
            killed_at = now
        elif killed_at is not None and now - killed_at >= GRACE_SECONDS:
            break
        time.sleep(SAMPLE_SECONDS)

    print(json.dumps({
        'cargo_exit_code': process.returncode,
        'cancellation_signal': cancellation,
        'elapsed_seconds': round(time.monotonic() - start, 3),
        'sample_interval_seconds': SAMPLE_SECONDS,
        'sampled_peak_group_rss_kib': peak_group_kib,
        'wait4_max_rss_kib': usage.ru_maxrss if usage is not None else None,
        'pids_after_cargo_exit': lingered,
        'surviving_group_pids': survivors,
        'target_dir': str(target),
    }), file=sys.stderr, flush=True)
    if survivors or lingered or process.returncode is None:
        return 1
    if cancellation is not None:
        return 128 + cancellation
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == '__main__':
    sys.exit(main())
