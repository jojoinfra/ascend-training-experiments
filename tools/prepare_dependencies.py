"""Materialize a model's pinned Git dependencies and print their cache directory."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def git(repo, *args):
    return subprocess.check_output(['git', '-C', repo, *args], text=True).strip()


def verify(repo, spec):
    if git(repo, 'rev-parse', 'HEAD') != spec['commit']:
        raise ValueError(f'{repo}: HEAD differs from the locked commit')
    expected_tree = spec.get('patched_tree') or git(repo, 'rev-parse', 'HEAD^{tree}')
    if git(repo, 'write-tree') != expected_tree:
        raise ValueError(f'{repo}: source tree differs from the lock')
    # Unpacking an unchanged cache alters filesystem timestamps and inode numbers.
    git(repo, 'update-index', '-q', '--refresh')
    git(repo, 'diff-files', '--quiet', '--')
    if git(repo, 'ls-files', '--others', '--exclude-standard'):
        raise ValueError(f'{repo}: unexpected untracked files')


def prepare(lock, cache):
    contents = lock.read_bytes()
    specs = json.loads(contents)
    fingerprint = hashlib.sha256(contents)
    for name, spec in specs.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
            raise ValueError(f'invalid dependency directory: {name}')
        if not re.fullmatch(r'[a-f0-9]{40}', spec['commit']):
            raise ValueError(f'{name}: expected a full Git commit SHA')
        if 'patch' in spec:
            if not re.fullmatch(r'[a-f0-9]{40}', spec['patched_tree']):
                raise ValueError(f'{name}: expected a patched Git tree SHA')
            fingerprint.update((lock.parent / spec['patch']).read_bytes())

    cache.mkdir(parents=True, exist_ok=True)
    bundle = cache / fingerprint.hexdigest()
    with bundle.with_suffix('.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if bundle.exists():
            for name, spec in specs.items():
                verify(bundle / name, spec)
            return bundle

        with tempfile.TemporaryDirectory(dir=cache) as staging:
            pending = Path(staging) / 'dependencies'
            pending.mkdir()
            for name, spec in specs.items():
                repo = pending / name
                repo.mkdir()
                git(repo, 'init', '--quiet')
                git(repo, 'remote', 'add', 'origin', spec['url'])
                print(f"Fetching {name} at {spec['commit']}", file=sys.stderr, flush=True)
                git(repo, 'fetch', '--quiet', '--depth=1', 'origin', spec['commit'])
                git(repo, 'checkout', '--quiet', '--detach', spec['commit'])
                if 'patch' in spec:
                    git(repo, 'apply', '--index', lock.parent / spec['patch'])
                verify(repo, spec)
            pending.rename(bundle)
    return bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lock', type=Path)
    parser.add_argument('--cache-dir', type=Path,
                        default=os.getenv('ASCEND_BENCH_DEPS_CACHE', '~/.cache/ascend-training-benchmark'))
    args = parser.parse_args()
    try:
        print(prepare(args.lock.resolve(), args.cache_dir.expanduser().resolve()))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')
