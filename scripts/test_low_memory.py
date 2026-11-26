import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

RUNNER = Path(__file__).with_name('low_memory.py').resolve()


@unittest.skipUnless(sys.platform == 'linux', 'Linux process supervision')
class LowMemoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.ready = root / 'ready'
        cargo = root / 'cargo'
        cargo.write_text(f'#!{sys.executable}\n' + '''import os
from pathlib import Path
import subprocess
import sys
import time

assert sys.argv[1:5] == ['check', '--locked', '--jobs', '1']
assert os.environ['CARGO_INCREMENTAL'] == '0'
assert os.environ['CARGO_BUILD_JOBS'] == '1'
assert Path(os.environ['CARGO_TARGET_DIR']) == Path.cwd() / 'target' / 'low-memory'
mode = sys.argv[-1]
if mode == 'pass':
    sys.exit(0)
if mode == 'fail':
    sys.exit(42)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path(os.environ['SYNAPSE_TEST_READY']).write_text(str(child.pid))
if mode == 'cancel':
    time.sleep(30)
''')
        cargo.chmod(0o700)
        self.env = os.environ | {
            'PATH': f"{root}:{os.environ['PATH']}", 'CARGO_BUILD_JOBS': '9',
            'CARGO_INCREMENTAL': '1', 'CARGO_TARGET_DIR': str(root),
            'SYNAPSE_TEST_READY': str(self.ready),
        }

    def run_mode(self, mode):
        with subprocess.Popen([sys.executable, RUNNER, 'check', mode], env=self.env,
                              text=True, stderr=subprocess.PIPE) as process:
            if mode == 'cancel':
                deadline = time.monotonic() + 5
                while not self.ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=15)
            result = json.loads(stderr.splitlines()[-1])
        self.assertEqual(result['surviving_group_pids'], [])
        self.assertGreater(result['wait4_max_rss_kib'], 0)
        return process.returncode, result

    def test_build_settings_and_cargo_status(self):
        for mode, expected in [('pass', 0), ('fail', 42)]:
            code, result = self.run_mode(mode)
            self.assertEqual(code, expected)
            self.assertEqual(result['cargo_exit_code'], expected)

    def test_leftover_child_is_cleaned_up_and_fails_the_build(self):
        code, result = self.run_mode('linger')
        self.assertEqual(code, 1)
        self.assertEqual(result['pids_after_cargo_exit'], [int(self.ready.read_text())])

    def test_cancellation_cleans_up_running_child(self):
        code, result = self.run_mode('cancel')
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertEqual(result['cancellation_signal'], signal.SIGTERM)
        self.assertFalse(Path(f'/proc/{self.ready.read_text()}').exists())
