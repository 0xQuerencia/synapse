#!/usr/bin/env python3
"""Install/update a verified local Linux package without sudo or background startup."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import tarfile
import tempfile
import uuid

from package_format import MAX_ARCHIVE, MAX_DOCUMENT, MAX_EXPANDED, canonical, digest, read_document, regular_file, validate_manifest, validate_statement, verify_signature, version_key

MARKER = '.synapse-install.json'
RELEASE_METADATA = {'release-statement.json', 'release.sig', 'release.pub', 'verification.json'}


def host_target():
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine().lower(), platform.machine().lower())
    if platform.system() != 'Linux' or machine not in {'x86_64', 'aarch64'}:
        raise ValueError('this installer currently supports Linux x86_64/aarch64 only')
    return f'{machine}-unknown-linux-gnu'


def safe_path(path):
    path = Path(os.path.abspath(os.path.expanduser(path)))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise ValueError('installation path must not contain symlinks')
    return path


def fsync_dir(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_owned(path, content, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fchmod(fd, mode)
            os.fsync(fd)
    finally:
        os.close(fd)


def prepare_prefix(prefix, create=False):
    prefix = safe_path(prefix)
    if prefix == Path('/'):
        raise ValueError('root directory cannot be an install prefix')
    if not prefix.exists():
        if not create:
            raise ValueError('installation does not exist')
        prefix.mkdir(parents=True, mode=0o700)
    info = prefix.stat()
    if info.st_uid != os.geteuid() or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o022:
        raise ValueError('prefix must be a directory owned and writable only by this user')
    marker = prefix / MARKER
    if not marker.exists():
        if not create or any(prefix.iterdir()):
            raise ValueError('refusing to use a nonempty unmanaged directory')
        write_owned(marker, canonical({'schema_version': 1, 'owner_uid': os.geteuid()}))
    value, _ = read_document(marker)
    if value != {'schema_version': 1, 'owner_uid': os.geteuid()}:
        raise ValueError('unrecognized install ownership marker')
    return prefix


@contextmanager
def locked(prefix):
    lock_path = prefix / '.install.lock'
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise ValueError('unsafe install lock')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('another installer operation is active') from error
        yield
    finally:
        os.close(fd)


def ensure_directory(path):
    if not path.exists():
        path.mkdir(mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError('managed directory is not a normal owned directory')


def running_processes(prefix):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.geteuid():
                continue
            executable = os.readlink(entry / 'exe').removesuffix(' (deleted)')
            if Path(executable).is_relative_to(prefix / 'releases'):
                found.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def stopped(prefix):
    if running_processes(prefix):
        raise ValueError('an installed process is running; stop or drain it before changing versions')


def pointer(prefix, name):
    value = prefix / name
    if not os.path.lexists(value):
        return None
    if not value.is_symlink():
        raise ValueError('refusing to replace a foreign install pointer')
    target = Path(os.readlink(value))
    if target.is_absolute() or len(target.parts) != 2 or target.parts[0] != 'releases' or target.parts[1].startswith('.'):
        raise ValueError('install pointer escapes managed releases')
    destination = prefix / target
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError('install pointer has no normal release directory')
    return target.as_posix()


def replace_pointer(prefix, name, target):
    temporary = prefix / ('.pointer-' + uuid.uuid4().hex)
    try:
        os.symlink(target, temporary)
        os.replace(temporary, prefix / name)
        fsync_dir(prefix)
    finally:
        temporary.unlink(missing_ok=True)


class BoundedExpandedReader:
    def __init__(self, source):
        self.source, self.total = source, 0

    def read(self, size):
        if size < 0 or size > 1024 * 1024:
            raise ValueError('unsafe archive read request')
        chunk = self.source.read(size)
        self.total += len(chunk)
        if self.total > MAX_EXPANDED:
            raise ValueError('expanded archive exceeds limit')
        return chunk


def extract_verified(snapshot, stage, statement):
    seen = set()
    with snapshot.open('rb') as compressed:
        with gzip.GzipFile(fileobj=compressed) as expanded:
            with tarfile.open(fileobj=BoundedExpandedReader(expanded), mode='r|') as archive:
                for member in archive:
                    if member.name not in {'manifest.json', 'bin/synapse', 'LICENSE', 'sbom.cdx.json'} or member.name in seen:
                        raise ValueError('archive has an unsupported or duplicate path')
                    if not member.isfile() or member.pax_headers or member.size <= 0 or member.size > MAX_ARCHIVE:
                        raise ValueError('archive contains an unsafe member')
                    if member.name == 'manifest.json' and member.size > MAX_DOCUMENT:
                        raise ValueError('manifest too large')
                    if member.mode & 0o7000:
                        raise ValueError('special permission bits are forbidden')
                    seen.add(member.name)
                    target = stage / member.name
                    if target.parent != stage:
                        ensure_directory(target.parent)
                    source = archive.extractfile(member)
                    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    try:
                        with os.fdopen(fd, 'wb', closefd=False) as output:
                            remaining = member.size
                            while remaining:
                                chunk = source.read(min(65536, remaining))
                                if not chunk:
                                    raise ValueError('truncated archive member')
                                output.write(chunk)
                                remaining -= len(chunk)
                            output.flush()
                            os.fsync(fd)
                    finally:
                        os.close(fd)
    if not {'manifest.json', 'bin/synapse', 'LICENSE'}.issubset(seen):
        raise ValueError('incomplete package')
    manifest, raw = read_document(stage / 'manifest.json')
    validate_manifest(manifest)
    if hashlib.sha256(raw).hexdigest() != statement['manifest_sha256']:
        raise ValueError('manifest does not match authenticated statement')
    if any(manifest[key] != statement[key] for key in ['version', 'target', 'channel']):
        raise ValueError('package metadata mismatch')
    if {item['path'] for item in manifest['files']} != seen - {'manifest.json'}:
        raise ValueError('manifest does not cover every package file')
    for item in manifest['files']:
        file = stage / item['path']
        if file.stat().st_size != item['size_bytes'] or digest(file) != item['sha256']:
            raise ValueError('package content hash/size mismatch')
        file.chmod(item['mode'])
    return manifest


def verify_release(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('release directory is unsafe')
    manifest, raw = read_document(directory / 'manifest.json')
    validate_manifest(manifest)
    statement, _ = read_document(directory / 'release-statement.json')
    validate_statement(statement)
    verification, _ = read_document(directory / 'verification.json')
    if hashlib.sha256(raw).hexdigest() != statement['manifest_sha256']:
        raise ValueError('installed manifest integrity mismatch')
    if any(manifest[key] != statement[key] for key in ['version', 'target', 'channel']):
        raise ValueError('installed metadata mismatch')
    if verification == {'method': 'signature'}:
        verify_signature(directory / 'release-statement.json', directory / 'release.sig', directory / 'release.pub')
    elif verification != {'method': 'explicit-development-checksum'} or statement['channel'] != 'development':
        raise ValueError('unrecognized package verification policy')
    expected = {item['path'] for item in manifest['files']} | {'manifest.json', 'release-statement.json', 'verification.json'}
    if verification['method'] == 'signature':
        expected |= {'release.sig', 'release.pub'}
    actual = set()
    for item in directory.rglob('*'):
        if item.is_symlink() or (not item.is_dir() and not item.is_file()):
            raise ValueError('unexpected link or special file in installation')
        if item.is_file():
            actual.add(item.relative_to(directory).as_posix())
    if actual != expected:
        raise ValueError('release contains unexpected or missing files')
    for item in manifest['files']:
        file = directory / item['path']
        info = regular_file(file)
        if info.st_size != item['size_bytes'] or digest(file) != item['sha256'] or stat.S_IMODE(info.st_mode) != item['mode']:
            raise ValueError('installed package was modified')
    return manifest


def install(archive, statement_path, prefix, signature=None, public_key=None, allow_development=False, expected_sha256=None, allow_downgrade=False, update=False):
    statement, statement_raw = read_document(statement_path)
    validate_statement(statement)
    if statement['target'] != host_target():
        raise ValueError('package target does not match this host')
    if Path(archive).name != statement['filename']:
        raise ValueError('archive filename does not match statement')
    if statement['channel'] == 'development' and not allow_development:
        raise ValueError('development package requires explicit --allow-development')
    signed = signature is not None and public_key is not None
    if (signature is None) != (public_key is None):
        raise ValueError('both signature and trusted public key are required')
    if signed:
        verify_signature(statement_path, signature, public_key)
    elif not (allow_development and statement['channel'] == 'development' and expected_sha256 == statement['sha256']):
        raise ValueError('trusted signature required; explicit development mode also requires independent --sha256')
    prefix = prepare_prefix(prefix, create=not update)
    with locked(prefix):
        stopped(prefix)
        old = pointer(prefix, 'current')
        if update and old is None:
            raise ValueError('no current installation to update')
        if old:
            current = verify_release(prefix / old)
            if not allow_downgrade and version_key(statement['version']) < version_key(current['version']):
                raise ValueError('downgrade requires explicit --allow-downgrade or rollback')
        ensure_directory(prefix / 'releases')
        ensure_directory(prefix / 'bin')
        shim = prefix / 'bin/synapse'
        if os.path.lexists(shim) and (not shim.is_symlink() or os.readlink(shim) != '../current/bin/synapse'):
            raise ValueError('refusing to overwrite a foreign executable')
        if shutil.disk_usage(prefix).free < statement['size_bytes'] * 2:
            raise ValueError('insufficient staging space')
        stage = Path(tempfile.mkdtemp(prefix='.stage-', dir=prefix / 'releases'))
        snapshot = prefix / ('.archive-' + uuid.uuid4().hex)
        try:
            source_fd = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                info = os.fstat(source_fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size != statement['size_bytes']:
                    raise ValueError('archive size/type mismatch')
                output_fd = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                observed = hashlib.sha256()
                try:
                    with os.fdopen(source_fd, 'rb', closefd=False) as source, os.fdopen(output_fd, 'wb', closefd=False) as output:
                        count = 0
                        while chunk := source.read(65536):
                            count += len(chunk)
                            if count > MAX_ARCHIVE or count > statement['size_bytes']:
                                raise ValueError('archive exceeds expected size')
                            observed.update(chunk)
                            output.write(chunk)
                        output.flush()
                        os.fsync(output_fd)
                finally:
                    os.close(output_fd)
                if count != statement['size_bytes'] or observed.hexdigest() != statement['sha256']:
                    raise ValueError('archive checksum mismatch')
            finally:
                os.close(source_fd)
            manifest = extract_verified(snapshot, stage, statement)
            write_owned(stage / 'release-statement.json', statement_raw)
            write_owned(stage / 'verification.json', canonical({'method': 'signature' if signed else 'explicit-development-checksum'}))
            if signed:
                write_owned(stage / 'release.sig', Path(signature).read_bytes())
                write_owned(stage / 'release.pub', Path(public_key).read_bytes())
            verify_release(stage)
            fsync_dir(stage / 'bin')
            fsync_dir(stage)
            destination_name = f"{manifest['version']}-{manifest['target']}-{statement['sha256'][:12]}"
            destination = prefix / 'releases' / destination_name
            if destination.exists():
                verify_release(destination)
                if digest(destination / 'release-statement.json') != digest(stage / 'release-statement.json'):
                    raise ValueError('release identity collision')
            else:
                os.replace(stage, destination)
                fsync_dir(prefix / 'releases')
            new = 'releases/' + destination_name
            if old and old != new:
                pointer(prefix, 'previous')
                replace_pointer(prefix, 'previous', old)
            replace_pointer(prefix, 'current', new)
            if not os.path.lexists(shim):
                os.symlink('../current/bin/synapse', shim)
                fsync_dir(prefix / 'bin')
            return destination
        finally:
            snapshot.unlink(missing_ok=True)
            if stage.exists():
                shutil.rmtree(stage)


def rollback(prefix):
    prefix = prepare_prefix(prefix)
    with locked(prefix):
        stopped(prefix)
        current, previous = pointer(prefix, 'current'), pointer(prefix, 'previous')
        if not current or not previous or current == previous:
            raise ValueError('no previous release available')
        verify_release(prefix / previous)
        # Preserve a damaged current package for inspection; only the target
        # of rollback must pass verification before it is activated.
        replace_pointer(prefix, 'current', previous)
        replace_pointer(prefix, 'previous', current)
        return prefix / previous


def uninstall(prefix, keep_releases=False):
    prefix = prepare_prefix(prefix)
    with locked(prefix):
        stopped(prefix)
        pointer(prefix, 'current')
        pointer(prefix, 'previous')
        allowed_root = {MARKER, '.install.lock', 'releases', 'bin', 'current', 'previous'}
        if any(item.name not in allowed_root for item in prefix.iterdir()):
            raise ValueError('unmanaged files present; refusing destructive cleanup')
        releases, binary_dir = prefix / 'releases', prefix / 'bin'
        ensure_directory(releases)
        ensure_directory(binary_dir)
        for directory in releases.iterdir():
            verify_release(directory)
        children = list(binary_dir.iterdir())
        if any(item.name != 'synapse' or not item.is_symlink() or os.readlink(item) != '../current/bin/synapse' for item in children):
            raise ValueError('unmanaged executable present')
        for name in ['current', 'previous']:
            (prefix / name).unlink(missing_ok=True)
        for item in children:
            item.unlink()
        binary_dir.rmdir()
        if not keep_releases:
            for directory in releases.iterdir():
                shutil.rmtree(directory)
            releases.rmdir()
            (prefix / MARKER).unlink()
            (prefix / '.install.lock').unlink()
            prefix.rmdir()
        else:
            fsync_dir(prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path.home() / '.local/share/synapse')
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ['install', 'update']:
        command = commands.add_parser(action)
        command.add_argument('archive', type=Path)
        command.add_argument('--statement', type=Path, required=True)
        command.add_argument('--signature', type=Path)
        command.add_argument('--public-key', type=Path)
        command.add_argument('--allow-development', action='store_true')
        command.add_argument('--sha256')
        command.add_argument('--allow-downgrade', action='store_true')
    commands.add_parser('rollback')
    commands.add_parser('verify')
    remove = commands.add_parser('uninstall')
    remove.add_argument('--keep-releases', action='store_true')
    args = parser.parse_args()
    try:
        if args.action in {'install', 'update'}:
            location = install(args.archive, args.statement, args.prefix, args.signature, args.public_key, args.allow_development, args.sha256, args.allow_downgrade, args.action == 'update')
            print(f'Installed verified local package at {location}')
            print(f'Executable: {args.prefix}/bin/synapse')
            print('No process or service was started. Add the bin directory to PATH yourself if desired.')
        elif args.action == 'rollback':
            print(f'Activated previous verified package: {rollback(args.prefix)}')
        elif args.action == 'verify':
            prefix = prepare_prefix(args.prefix)
            with locked(prefix):
                current = pointer(prefix, 'current')
                if not current:
                    raise ValueError('no active package')
                print(json.dumps(verify_release(prefix / current), indent=2))
        else:
            uninstall(args.prefix, args.keep_releases)
            print('Removed installer-owned executables. Model, identity and configuration data were preserved.')
    except (OSError, ValueError, tarfile.TarError, EOFError) as error:
        parser.exit(1, f'Installer refused: {error}\n')


if __name__ == '__main__':
    main()
