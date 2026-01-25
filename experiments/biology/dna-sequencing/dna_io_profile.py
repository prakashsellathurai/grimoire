import os
import shutil
import time
import mmap
import gzip
import bz2
import tracemalloc
from time import perf_counter
from pathlib import Path
import tempfile
import argparse
import sys


def generate_sequence(num_bytes):
    # simple repeating ACGT pattern to reach approximate size
    bases = b'ACGT'
    rep = num_bytes // len(bases) + 1
    data = (bases * rep)[:num_bytes]
    return data

def removedirs(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' and all its contents have been removed.")
        except OSError as e:
            print(f"Error: {directory_path} : {e.strerror}")
    else:
        print(f"Directory '{directory_path}' does not exist.")

def fsync_file(f):
    try:
        f.flush()
        os.fsync(f.fileno())
    except Exception:
        pass


def measure(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = perf_counter()
    result = fn(*args, **kwargs)
    dur = perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dur, peak, result


def write_text(path: Path, data: bytes):
    with path.open('w', encoding='ascii') as f:
        f.write(data.decode('ascii'))
        fsync_file(f)


def read_text(path: Path):
    with path.open('r', encoding='ascii') as f:
        return f.read()


def write_binary(path: Path, data: bytes):
    with path.open('wb') as f:
        f.write(data)
        fsync_file(f)


def read_binary(path: Path):
    with path.open('rb') as f:
        return f.read()


def write_gzip(path: Path, data: bytes):
    with gzip.open(path, 'wb') as f:
        f.write(data)


def read_gzip(path: Path):
    with gzip.open(path, 'rb') as f:
        return f.read()


def write_bz2(path: Path, data: bytes):
    with bz2.open(path, 'wb') as f:
        f.write(data)


def read_bz2(path: Path):
    with bz2.open(path, 'rb') as f:
        return f.read()


def read_mmap(path: Path):
    with path.open('rb') as f:
        size = os.path.getsize(path)
        if size == 0:
            return b''
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            return mm[:]
        finally:
            mm.close()


def read_in_chunks(path: Path, chunk_size=1 << 20):
    total = 0
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
    return total


def run_all_tests(target_dir: Path, size_bytes: int):
    target_dir.mkdir(parents=True, exist_ok=True)
    results = []
    data = generate_sequence(size_bytes)

    tests = [
        ("text_write", write_text, target_dir / 'seq.txt'),
        ("text_read", read_text, target_dir / 'seq.txt'),
        ("binary_write", write_binary, target_dir / 'seq.bin'),
        ("binary_read", read_binary, target_dir / 'seq.bin'),
        ("gzip_write", write_gzip, target_dir / 'seq.gz'),
        ("gzip_read", read_gzip, target_dir / 'seq.gz'),
        ("bz2_write", write_bz2, target_dir / 'seq.bz2'),
        ("bz2_read", read_bz2, target_dir / 'seq.bz2'),
        ("mmap_read", read_mmap, target_dir / 'seq.bin'),
        ("chunked_read", read_in_chunks, target_dir / 'seq.bin')
    ]

    # ensure files removed before starting
    for _, _, p in tests:
        try:
            p.unlink()
        except Exception:
            pass

    # run write tests first to ensure reads have files
    for name, fn, path in tests:
        if name.endswith('_write'):
            if 'text' in name:
                dur, peak, _ = measure(fn, path, data)
            else:
                dur, peak, _ = measure(fn, path, data)
            results.append((name, dur, peak, path))

    # run read tests
    for name, fn, path in tests:
        if not name.endswith('_write'):
            dur, peak, out = measure(fn, path)
            # avoid keeping large result in memory; report length
            out_len = len(out) if isinstance(out, (bytes, str)) else out
            results.append((name, dur, peak, path, out_len))
    removedirs(target_dir)
    return results


def main():
    p = argparse.ArgumentParser(description='Profile DNA file I/O methods')
    p.add_argument('--size', type=int, default=5_000_000, help='approx bytes to generate')
    p.add_argument('--dir', type=str, default=None, help='target directory for test files')
    args = p.parse_args()

    if args.dir:
        target = Path(args.dir)
    else:
        target = Path(__file__).parent / 'profile_tmp'

    print(f"Running I/O profile in {target} with ~{args.size} bytes")
    results = run_all_tests(target, args.size)
    results.sort(key=lambda tup: (tup[2],tup[1]))
    print('\nResults')
    for r in results:
        if len(r) == 4:
            name, dur, peak, path = r
            print(f"- {name}: {dur:.3f}s, peak_mem={peak/1024/1024:.3f} MiB")
        else:
            name, dur, peak, path, out_len = r
            print(f"- {name}: {dur:.3f}s, peak_mem={peak/1024/1024:.3f} MiB")


if __name__ == '__main__':
    main()
