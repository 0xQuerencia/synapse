import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

import install as installer
import package as builder
from package_format import canonical, digest, version_key


class InstallationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keys = tempfile.TemporaryDirectory(prefix='synapse-test-keys-')
        cls.key = Path(cls.keys.name) / 'signing.pem'
        cls.public = Path(cls.keys.name) / 'public.pem'
        subprocess.run(['openssl', 'genpkey', '-algorithm', 'ED25519', '-out', str(cls.key)], check=True, capture_output=True)
        cls.key.chmod(0o600)
        subprocess.run(['openssl', 'pkey', '-in', str(cls.key), '-pubout', '-out', str(cls.public)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.keys.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='synapse-install-test-')
        self.root = Path(self.temporary.name)
        self.prefix = self.root / 'application'
        self.packages = self.root / 'packages'
        self.count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def package(self, version='1.0.0', signed=False, binary='/bin/true', target=None):
        self.count += 1
        return builder.make_package(Path(binary), version, target or installer.host_target(), self.packages / str(self.count), signing_key=self.key if signed else None, channel='candidate' if signed else 'development')

    def install(self, paths, **kwargs):
        archive, statement, signature = paths
        if signature:
            return installer.install(archive, statement, self.prefix, signature=signature, public_key=self.public, **kwargs)
        return installer.install(archive, statement, self.prefix, allow_development=True, expected_sha256=digest(archive), **kwargs)

    def evil_package(self, members):
        self.count += 1
        directory = self.packages / str(self.count)
        directory.mkdir(parents=True)
        version, target = '9.0.0', installer.host_target()
        filename = f'synapse-{version}-{target}.tar.gz'
        archive = directory / filename
        with tarfile.open(archive, 'w:gz', format=tarfile.USTAR_FORMAT) as output:
            for member, content in members:
                output.addfile(member, io.BytesIO(content) if content else None)
        statement = {'schema_version': 1, 'filename': filename, 'sha256': digest(archive), 'size_bytes': archive.stat().st_size, 'manifest_sha256': '0' * 64, 'version': version, 'target': target, 'channel': 'development'}
        statement_path = directory / (filename + '.statement.json')
        statement_path.write_bytes(canonical(statement))
        return archive, statement_path, None

    def test_signed_install_creates_executable_without_startup(self):
        release = self.install(self.package(signed=True))
        self.assertEqual(installer.verify_release(release)['version'], '1.0.0')
        self.assertEqual(os.readlink(self.prefix / 'bin/synapse'), '../current/bin/synapse')
        self.assertEqual(subprocess.run([str(self.prefix / 'bin/synapse')]).returncode, 0)
        self.assertEqual(installer.running_processes(self.prefix), [])

    def test_unsigned_install_requires_explicit_development_and_independent_digest(self):
        archive, statement, _ = self.package()
        with self.assertRaisesRegex(ValueError, 'development'):
            installer.install(archive, statement, self.prefix)
        with self.assertRaisesRegex(ValueError, 'signature'):
            installer.install(archive, statement, self.prefix, allow_development=True, expected_sha256='0' * 64)
        self.assertFalse(self.prefix.exists())

    def test_bad_signature_does_not_create_installation(self):
        archive, statement, signature = self.package(signed=True)
        signature.write_bytes(b'0' * 64)
        with self.assertRaises(ValueError):
            self.install((archive, statement, signature))
        self.assertFalse(self.prefix.exists())

    def test_modified_signed_statement_is_rejected(self):
        paths = self.package(signed=True)
        data = json.loads(paths[1].read_text())
        data['sha256'] = '0' * 64
        paths[1].write_bytes(canonical(data))
        with self.assertRaises(ValueError):
            self.install(paths)
        self.assertFalse(self.prefix.exists())

    def test_tampered_archive_leaves_current_release_unchanged(self):
        old = self.install(self.package())
        paths = self.package('1.1.0', signed=True)
        raw = bytearray(paths[0].read_bytes())
        raw[len(raw) // 2] ^= 1
        paths[0].write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.install(paths, update=True)
        self.assertEqual((self.prefix / 'current').resolve(), old)
        self.assertFalse(list(self.prefix.glob('.archive-*')))
        self.assertFalse(list((self.prefix / 'releases').glob('.stage-*')))

    def test_update_and_rollback_are_verified_pointer_changes(self):
        old = self.install(self.package('1.0.0', signed=True))
        new = self.install(self.package('1.1.0', signed=True), update=True)
        self.assertEqual((self.prefix / 'current').resolve(), new)
        self.assertEqual((self.prefix / 'previous').resolve(), old)
        self.assertEqual(installer.rollback(self.prefix), old)
        self.assertEqual((self.prefix / 'current').resolve(), old)
        self.assertEqual((self.prefix / 'previous').resolve(), new)

    def test_downgrade_needs_explicit_permission(self):
        current = self.install(self.package('2.0.0'))
        older = self.package('1.9.0')
        with self.assertRaisesRegex(ValueError, 'downgrade'):
            self.install(older, update=True)
        self.assertEqual((self.prefix / 'current').resolve(), current)
        self.install(older, update=True, allow_downgrade=True)
        self.assertEqual(installer.verify_release((self.prefix / 'current').resolve())['version'], '1.9.0')

    def test_corrupt_previous_is_never_activated(self):
        previous = self.install(self.package())
        current = self.install(self.package('1.1.0'), update=True)
        (previous / 'bin/synapse').write_bytes(b'broken')
        with self.assertRaises(ValueError):
            installer.rollback(self.prefix)
        self.assertEqual((self.prefix / 'current').resolve(), current)

    def test_corrupt_current_can_be_recovered_with_verified_previous(self):
        previous = self.install(self.package())
        current = self.install(self.package('1.1.0'), update=True)
        (current / 'bin/synapse').write_bytes(b'broken')
        self.assertEqual(installer.rollback(self.prefix), previous)
        self.assertTrue(current.exists())

    def test_traversal_absolute_and_link_members_are_rejected(self):
        for name, kind, target in [('../escape', tarfile.REGTYPE, ''), (str(self.root / 'outside'), tarfile.REGTYPE, ''), ('bin/synapse', tarfile.SYMTYPE, '/bin/sh'), ('bin/synapse', tarfile.LNKTYPE, '/bin/sh'), ('bin/synapse', tarfile.FIFOTYPE, '')]:
            with self.subTest(name=name, kind=kind):
                member = tarfile.TarInfo(name)
                member.type, member.linkname = kind, target
                member.size = 1 if kind == tarfile.REGTYPE else 0
                paths = self.evil_package([(member, b'x' if member.size else b'')])
                with self.assertRaises(ValueError):
                    self.install(paths)
                self.assertFalse((self.root / 'outside').exists())
                self.assertFalse((self.root / 'escape').exists())
                self.assertFalse((self.prefix / 'current').exists())

    def test_duplicate_archive_member_is_rejected(self):
        member = tarfile.TarInfo('manifest.json')
        member.size = 2
        paths = self.evil_package([(member, b'{}'), (member, b'{}')])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.install(paths)

    def test_symlink_prefix_and_managed_directory_are_rejected(self):
        actual = self.root / 'actual'
        actual.mkdir()
        self.prefix.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.install(self.package())
        self.prefix.unlink()
        self.install(self.package())
        releases = self.prefix / 'releases'
        shutil.rmtree(releases)
        releases.symlink_to(actual, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.install(self.package('1.2.0'))
        self.assertEqual(list(actual.iterdir()), [])

    def test_uninstall_preserves_unmanaged_files_by_refusing(self):
        self.install(self.package())
        personal = self.prefix / 'notes.txt'
        personal.write_text('keep this')
        with self.assertRaisesRegex(ValueError, 'unmanaged'):
            installer.uninstall(self.prefix)
        self.assertEqual(personal.read_text(), 'keep this')
        self.assertTrue((self.prefix / 'bin/synapse').exists())

    def test_uninstall_preserves_configuration_outside_install_prefix(self):
        self.install(self.package())
        data = self.root / 'runtime-data'
        data.mkdir()
        (data / 'identity.key').write_text('synthetic local fixture')
        installer.uninstall(self.prefix)
        self.assertFalse(self.prefix.exists())
        self.assertEqual((data / 'identity.key').read_text(), 'synthetic local fixture')

    def test_keep_releases_removes_entrypoint_without_deleting_versions(self):
        release = self.install(self.package())
        installer.uninstall(self.prefix, keep_releases=True)
        self.assertTrue(release.exists())
        self.assertFalse(os.path.lexists(self.prefix / 'current'))
        self.assertFalse((self.prefix / 'bin').exists())

    def test_concurrent_installer_refuses_before_changing_pointer(self):
        current = self.install(self.package())
        with installer.locked(self.prefix):
            with self.assertRaisesRegex(ValueError, 'another installer'):
                self.install(self.package('1.2.0'), update=True)
        self.assertEqual((self.prefix / 'current').resolve(), current)

    def test_live_installed_process_blocks_update_and_uninstall(self):
        self.install(self.package(binary='/bin/sleep'))
        process = subprocess.Popen([str(self.prefix / 'bin/synapse'), '30'])
        try:
            self.assertIn(process.pid, installer.running_processes(self.prefix))
            with self.assertRaisesRegex(ValueError, 'running'):
                self.install(self.package('1.1.0'), update=True)
            with self.assertRaisesRegex(ValueError, 'running'):
                installer.uninstall(self.prefix)
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_wrong_target_is_rejected(self):
        other = 'aarch64-unknown-linux-gnu' if installer.host_target().startswith('x86_64') else 'x86_64-unknown-linux-gnu'
        with self.assertRaisesRegex(ValueError, 'target'):
            self.install(self.package(target=other))
        self.assertFalse(self.prefix.exists())

    def test_foreign_directory_is_not_adopted(self):
        self.prefix.mkdir()
        sentinel = self.prefix / 'personal-file'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'unmanaged'):
            self.install(self.package())
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_package_is_reproducible_and_does_not_overwrite(self):
        first, second = self.package(), self.package()
        self.assertEqual(digest(first[0]), digest(second[0]))
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            builder.make_package(Path('/bin/true'), '1.0.0', installer.host_target(), first[0].parent)

    def test_semantic_versions_compare_numeric_prereleases_correctly(self):
        self.assertLess(version_key('1.0.0-rc.2'), version_key('1.0.0-rc.10'))
        self.assertLess(version_key('1.0.0-rc.10'), version_key('1.0.0'))
        self.assertLess(version_key('1.9.0'), version_key('1.10.0'))
        for value in ['../version', '01.0.0', '1.0.0-01', '1.0.0-rc..1']:
            with self.assertRaises(ValueError):
                version_key(value)


if __name__ == '__main__':
    unittest.main()
